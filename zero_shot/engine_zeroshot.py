# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import os
import math
import random
import pickle
import numpy as np
from typing import Iterable

import torch
from torch.utils.data import Dataset

from ..pretrain_pipeline import misc

# ===== LOAD DATA =======================================================
def read_pickle(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError) as e:
        print(f"Error reading file: {file_path}")
        print(f"Error details: {str(e)}")
        print(f"File size: {os.path.getsize(file_path)} bytes")
        return None
    
class ZerShotDataset(Dataset):
    def __init__(self, args, nvar=4):
        self.img_dir = args.data_path
        self.nvar = nvar

    def __len__(self): # 195910 if data not in pod, else 144715
        return 195910 #len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        # check data condition
        data_in_pod = os.path.isfile("../data_in_pod.pkl")

        # fetch file path
        img_path = os.path.join(self.img_dir,f'{idx}.pkl')
        if data_in_pod:
            img_path = img_path.replace("../", "") # switch the root data path to the one in pod
        else:
            os.makedirs(self.img_dir.replace("../", ""), exist_ok=True)
        
        # read data
        if not os.path.isfile(img_path):
            return None
        data_dict = read_pickle(img_path)

        if data_dict.get('label') is None:
            return None

        # if data_dict is None:
        #     return None
            # # Skip this item and get the next one
            # print(f"Skipping item {idx} due to pickle error")
            # return self.__getitem__((idx + 1) % len(self))

        # # if use spectrum
        # x = torch.from_numpy(data_dict['spec'])
        # # shape adjust
        # ch_idx, F_idx, L_idx = 0, 0, 0
        # for s_i in range(1, 4):
        #     if x.shape[s_i] == 3:
        #         ch_idx = s_i
        #     elif x.shape[s_i] == 65:
        #         F_idx = s_i 
        #     else:
        #         L_idx = s_i
        # x = torch.permute(x, (0, ch_idx, L_idx, F_idx)) # nvar, 3, L, F

        if torch.is_tensor(data_dict['tss']):
            x = data_dict['tss'] # nvar, L
        else:
            x = torch.from_numpy(data_dict['tss']) # nvar, L
            x = torch.stack([ts for ts in x if (torch.isnan(ts).sum() / x.shape[1]) < 0.4])

        # ========== for saving to pod ========================================
        if not data_in_pod:
            # write
            with open(img_path.replace("../", ""), 'wb') as f:
                #print("Check filepath before write to avoid overwrite:", img_path.replace("../", ""))
                saved_dict = {
                    'tss': data_dict['tss'],
                    'label': data_dict['label']
                }
                pickle.dump(saved_dict, f)
        # ======================================================================
        
        # sample one task
        task_key = np.random.choice([k for k in data_dict['label'].keys()], 1)[0]

        # packaging for return
        return_pack = {
            # 'spec': x,
            'tss': x.float(),
            'task': task_key, # str
            'label': str(data_dict['label'][task_key]) # str
        }

        return return_pack

def collate_fn(batch, pad_nvar=4):
    # Filter out None values from the batch
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        # return {'spec': torch.empty(0), 'task': torch.empty(0), 'label': torch.empty(0)}
        return {'tss': torch.empty(0), 'task': torch.empty(0), 'label': torch.empty(0)}
    
    # Collect strings into a list
    task = [item['task'] for item in batch]  
    label = [item['label'] for item in batch]

    # Find the maximum sequence length L in the batch
    # max_L = 387 # +2 if it is tss
    max_L = 389
    
    batch_size = len(batch)
    # input_dim_1 = batch[0]['spec'].size(1)
    # input_dim_4 = batch[0]['spec'].size(3)
    
    # Initialize padded tensors for inputs and targets
    # padded_inputs = torch.zeros((batch_size, pad_nvar, input_dim_1, max_L, input_dim_4))
    padded_inputs = torch.zeros((batch_size, pad_nvar, max_L))

    for i, item in enumerate(batch):
        # print('item[input] shape: ',item['input'].shape)
        # exit()

        # n_var = item['spec'].size(0)
        # L = item['spec'].size(2)

        # if use tss
        n_var = item['tss'].size(0)
        L = item['tss'].size(1)
        
        # If the number of variables is greater than pad_nvar, randomly sample pad_nvar variables
        if n_var > pad_nvar:
            indices = random.sample(range(n_var), pad_nvar)
            # item['spec'] = item['spec'][indices]
            item['tss'] = item['tss'][indices]
            n_var = pad_nvar


        # Pad the inputs and targets
        # padded_inputs[i, :n_var, :, :L, :] = item['spec']
        padded_inputs[i, :n_var, :L] = item['tss']

        # Copy last valid sequence to pad remaining variables up to pad_nvar
        for j in range(n_var, pad_nvar):
            padded_inputs[i, j] = padded_inputs[i, n_var - 1]
        

    # return {'spec': padded_inputs, 'task': item['task'], 'label': item['label']}
    return {'tss': padded_inputs, 'task': task, 'label': label}

# ===== TRAIN HELPER FUNCTION =======================================================
def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
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

    # TODO: loading samples
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # batch = [b for b in batch if b is not None]
        # samples = batch['spec'].to(args.device)

        # edge case: skip if have less than 2 samples in the batch (unluckly sampled too many samples with no ground truth available)
        if len(batch['tss']) < 2:
            continue

        samples = batch['tss'].to(args.device)

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            misc.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            # input is x, task, label
            loss = model(samples, batch['task'], label=batch['label'])

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print('samples', torch.mean(samples))
            
            # for param in model.parameters():
            #     if param.grad is not None:
            #         param.grad.detach_()
            #         param.grad.zero_()
            
            # Replace the non-finite loss with zero to avoid issues in backward pass
            # loss = torch.tensor(1.0, requires_grad=True).to(loss.device)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),clip_grad=args.clip_grad,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        # free memory
        del samples

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}