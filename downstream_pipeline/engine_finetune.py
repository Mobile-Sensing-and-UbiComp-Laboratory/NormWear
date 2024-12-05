# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional
import torch
import sklearn
import analysis
from timm.data import Mixup
import numpy as np
import misc as misc
import modules.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 1,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = batch['input'].to(device, non_blocking=True)
        targets = batch['label'].to(device, non_blocking=True)
        #print(targets)
        #print(samples.shape)

        with torch.cuda.amp.autocast():
            outputs,z = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        #torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

       # loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

            

    # gather the stats from all processes
    #metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(args,data_loader, model, device,criterion,print_analyzer=True):
    analyzer = analysis.Analyzer(print_conf_mat=print_analyzer)
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    per_batch = {'targets': [], 'predictions': [], 'metrics': []}
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch['input']
        target = batch['label']
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
    
        # compute output
        with torch.cuda.amp.autocast():
            output,z = model(images)
            loss = criterion(output, target)

        per_batch['targets'].append(target.cpu().numpy())
        per_batch['predictions'].append(output.float().cpu().numpy())
        per_batch['metrics'].append([loss.cpu().numpy()])

        metric_logger.update(loss=loss.item())

    epoch_loss = metric_logger.loss.global_avg
    predictions = torch.from_numpy(np.concatenate(per_batch['predictions'], axis=0))
    probs = torch.nn.functional.softmax(predictions,dim=1)
    predictions = torch.argmax(probs, dim=1).cpu().numpy()
    probs = probs.cpu().numpy()
    targets = np.concatenate(per_batch['targets'], axis=0).flatten()
    class_names = np.arange(probs.shape[1])
    metrics_dict = analyzer.analyze_classification(predictions, targets, class_names)
    # epoch_metrics['loss'] = epoch_loss
    metrics_dict['loss'] = epoch_loss
    
    # epoch_metrics['accuracy'] = metrics_dict['total_accuracy']  # same as average recall over all classes
    # epoch_metrics['precision'] = metrics_dict['prec_avg']  # average precision over all classes
     # FIXME: Add Auroc and AP:
    if len(list(set(targets))) <= 2:
        roc_auc = sklearn.metrics.roc_auc_score(targets, probs[:, 1])
        ap = sklearn.metrics.average_precision_score(targets, probs[:, 1])
    else:
        roc_auc = sklearn.metrics.roc_auc_score(targets, probs, multi_class="ovo", average="macro", labels=list(set(targets)))
        ap = sklearn.metrics.average_precision_score(targets, probs, average="macro")

    metric_logger.meters['accuracy'].update(metrics_dict['total_accuracy'])
    metric_logger.meters['roc_auc'].update(roc_auc)
    metric_logger.meters['ap'].update(ap)
    
    #print(f"* AUCROC {epoch_metrics['roc_auc']:.5f};  * loss {epoch_loss:.3f};")
    #return epoch_metrics,metrics_dict
    print(f"* AUCROC {metric_logger.roc_auc.global_avg:.5f};  * loss {metric_logger.loss.global_avg:.3f};")


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}