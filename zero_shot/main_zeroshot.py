# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

# basic module
import argparse
import datetime
import pickle
import json
import numpy as np
import os
import time
from pathlib import Path

# torch related modules
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from ..pretrain_pipeline.misc import get_next_run_number

import timm.optim.optim_factory as optim_factory
import torch.nn as nn
from ..pretrain_pipeline import misc
from ..pretrain_pipeline.misc import NativeScalerWithGradNormCount as NativeScaler
from torch.utils.data import DataLoader

# model and data specific module
from .engine_zeroshot import train_one_epoch, collate_fn, ZerShotDataset
from .msitf_fusion import *

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training for zero-shot', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--remark', default='model_mae',
                        help='model_remark')
    parser.add_argument('--save_every_epoch', default=2, type=int,
                        help='default: save every 10 epoches')

    # Model parameters
    parser.add_argument('--nvar', type=int, default=4)
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--norm_pix_loss', type=bool, default=False)
    

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='max norm of gradient')
    

    # Dataset parameters
    parser.add_argument('--data_path', default='../data/wearable_zeroshot_pretrain', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default="../data/audio_results", type=str)
    parser.add_argument('--log_dir', default='../data/audio_results/log',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    # parser.add_argument('--resume', default='../../data/audio_results/audio_only_normwear/audio_only_normwear_checkpoint-10.pth',
    #                     help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    return parser


def main(args):
    misc.init_distributed_mode(args)
    print('======================= starting pretrain =======================')
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # TODO: add dataset_train
    dataset_train = ZerShotDataset(args)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        log_dir = os.path.join(args.log_dir, args.remark)
        os.makedirs(log_dir,exist_ok=True)
        run_number = get_next_run_number(log_dir, resume=args.resume)
        log_run_dir = os.path.join(log_dir, f'run_{run_number}')
        os.makedirs(log_run_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=log_run_dir)
    else:
        log_writer = None


    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, 
                        collate_fn=lambda x: collate_fn(x, pad_nvar=args.nvar),
                                sampler=sampler_train,drop_last=False,
                                num_workers=args.num_workers,pin_memory=True)
    print("Number of Samples:", len(dataset_train))
    
    # define the model
    print("Loading model...")
    model = NormWearZeroShot()

    model.to(device)

    model_without_ddp = model
    #print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        print('use distributed!!')
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95),amsgrad=True)
    # optimizer = torch.optim.SGD(param_groups, lr=args.lr,momentum=0.9,nesterov=True)
    #print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    ####### Check model has grad ########
    # for name, param in model.named_parameters():
    #     if not param.requires_grad:
    #         print(f"Warning: Parameter {name} has requires_grad=False")
    # print("###############Finish checking grad################")
    ########################################
    print(model.module.aggregator)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % args.save_every_epoch == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model.module.aggregator, model_without_ddp=model_without_ddp.aggregator, 
                optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch,
                model_only=True)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        # flag for whehther 1st epoch in this pod
        data_in_pod = os.path.isfile("../data_in_pod.pkl")
        if not data_in_pod:
            with open("../data_in_pod.pkl", 'wb') as f:
                pickle.dump("", f)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        args.output_dir = os.path.join(args.output_dir, args.remark)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

# python3 -m src.zero_shot.main_zeroshot --remark zs_test --batch_size 4
# python3 -m src.zero_shot.main_zeroshot --remark zs_test --resume ../data/audio_results/normwear_audio/normwear_audio_checkpoint-20.pth

# main new
# torchrun --nproc_per_node=4 -m src.zero_shot.main_zeroshot --remark normwear_msitf_clean --blr 1e-3 --weight_decay 1e-4 --epochs 50 --clip_grad 1.0 --save_every_epoch 5 --batch_size 64 --accum_iter 1