import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from config import DATASET_CONFIG
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from modules.pos_embed import interpolate_pos_embed

from modules.normwear import NormWear
from pretrain_pipeline.dataset import dataset_class



from tqdm import tqdm
from modules.head import RegressionHead, ClassificationHead
from torch.utils.tensorboard import SummaryWriter

import misc as misc
from misc import NativeScalerWithGradNormCount as NativeScaler
from engine_finetune import train_one_epoch,evaluate


def parse_tuple(arg_str):
    # Remove parentheses if they exist
    if arg_str.startswith('(') and arg_str.endswith(')'):
        arg_str = arg_str[1:-1]
    # Split the string by comma and convert to a tuple of floats
    return tuple(map(float, arg_str.split(',')))

def get_args_parser():
    parser = argparse.ArgumentParser('PhysioModel linear probing', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=4, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--y_range', default=None, type=parse_tuple)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--remark', default='model_mae_t6f5',
                        help='model_remark')
    # Model parameters
    parser.add_argument('--model_weight_dir', default='../data/results', type=str,
                        help='path of model weight')
    parser.add_argument('--model_name', default='model_mae', type=str,
                        help='Name of model to train')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--eval', default=False,type=bool,
                        help='Perform evaluation only')
    parser.add_argument('--checkpoint', default='../data/results/model_mae_checkpoint-140.pth', 
                        type=str,help='model checkpoint for evaluation')
    parser.add_argument('--task', default='reg', 
                        type=str,help='reg/cls')
    parser.add_argument('--log_dir', default='../data/results/log',
                        help='path where to tensorboard log')
    parser.add_argument('--new_size', default=(43,13),type=parse_tuple,
                        help='new number of patches')
    parser.add_argument('--use_meanpooling', default=0,type=int,
                        help='meanpooling cls fusion')
       

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')

    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--seed', default=42, type=int)

    # Dataset parameters
    parser.add_argument('--is_pretrain',default=False,type=bool)
    parser.add_argument('--data_dir', default='data', type=str,
                        help='dataset path')
    parser.add_argument('--ds_name', default='maus', type=str,
                        help='dataset name')
    
    parser.add_argument('--num_classes', type=int, default=1,
                    help='number of the classification types')
    parser.add_argument('--num_channel', default=1, type=int,
                        help='number of the input chennels')

    
    parser.add_argument('--output_dir', default='.././data/results',
                        help='path where to save, empty for no saving')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    

    return parser

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class LinearProb(nn.Module):
    def __init__(self, backbone,
                 num_classes,num_channel,
                 embed_size=768,task='reg',
                 y_range=None):
        
        super().__init__()
        # init backbone
        self.backbone = backbone
        #freeze_model(self.backbone)

        if task == 'reg':
            self.head = RegressionHead(n_vars=num_channel,
                            d_model=embed_size,y_range=y_range,
                            output_dim=num_classes)

        else:
            self.head = ClassificationHead(n_vars=num_channel,
                            d_model=embed_size,
                            n_classes=num_classes)

        self.embed_size = embed_size

        #self.analyzer = analysis.Analyzer(print_conf_mat=True) #Issue when doing model_ddp, this will cause isse
    def forward(self, x):
        '''Input
        x: bs x nvar, ch, L, F

        Output:
        cls: bs x num_class
        reg: bs
        '''
        
        z = self.backbone.feature_extractor(x) # bs,nvar,L+1,E
        x_out = self.head(z)

        return x_out, z
    

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    
    if args.log_dir is not None:
        path = os.path.join(args.log_dir, args.ds_name)
        os.makedirs(path, exist_ok=True)
        #run_number = get_next_run_number(path)
        log_run_dir = os.path.join(path, f'run_{args.remark}')
        os.makedirs(log_run_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=log_run_dir)

    train_dataset = dataset_class(data_dir=args.data_dir, is_train=True,max_len = args.max_len,use_spec=True)
    test_dataset = dataset_class(data_dir=args.data_dir, is_train=False,max_len = args.max_len,use_spec=True)

    print("Number of Training Samples:", len(train_dataset))
    print("Number of Testing Samples:", len(test_dataset))

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if True:#args.dist_eval:
            if len(test_dataset) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                test_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(test_dataset)
            
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train) # shuffle = True
        sampler_val = torch.utils.data.SequentialSampler(dataset_val) # shuffle = False
    

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                num_workers=4,
                                sampler = sampler_train,
                                drop_last=True,pin_memory=True,)
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                                num_workers=4,
                                drop_last=False,pin_memory=True,
                                sampler = sampler_val,)




    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    max_accuracy = 0.0
    best_epoch = 0
    # Reload everytime
    print('Loading pre-trained checkpoint from',args.checkpoint)
    backbone = NormWear(img_size=args.img_size,patch_size=(9,5),nvar=args.num_channel,
                                    is_pretrain=False,use_meanpooling=args.use_meanpooling,
                                    mask_prob=0,)
    checkpoint = torch.load(args.checkpoint,map_location='cpu')
    checkpoint_model = checkpoint['model']

    # TODO: Interpolating position embedding
    print('new size: ',args.new_size)
    interpolate_pos_embed(backbone, checkpoint_model,orig_size=(43,13),new_size=args.new_size)

    msg = backbone.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    model = LinearProb(backbone,
                num_classes=args.num_classes,
                num_channel=args.num_channel,
                task=args.task,
                y_range=args.y_range)

    # print("Model = %s" % str(model))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of training params : %.2f' % (n_parameters))
    model_without_ddp = model
    model.to(device)
    
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("lr: %.3e" % args.lr)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    optimizer =  torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay,)

    print(optimizer)
    loss_scaler = NativeScaler()

    criterion = torch.nn.L1Loss() if args.task == 'reg' else torch.nn.CrossEntropyLoss()
    print("criterion = %s" % str(criterion))

    # Individual experiement
    #epoch_metrics = OrderedDict()
    
    for epoch in tqdm(range(args.epochs)):
        train_stats = train_one_epoch(
            model,criterion,train_loader,
            optimizer,device,epoch,loss_scaler,
            max_norm=None,
            log_writer=log_writer,
            args=args,)
        
        # training accuracy
        #epoch_metrics,metrics_dict = evaluate(args,test_loader, model, device,criterion,analyzer)
        #print(f"AUCROC of the network on the {len(test_loader)} test images: {epoch_metrics['roc_auc']:.3f}, AP: {epoch_metrics['ap']:.5f}, Accuracy: {epoch_metrics['accuracy']:.5f}")
        test_stats = evaluate(args,test_loader, model, device,criterion,print_analyzer=False)
        print(f"AUCROC of the network on the {len(test_loader)} test images: {test_stats['roc_auc']:.3f}, AP: {test_stats['ap']:.5f}, Accuracy: {test_stats['accuracy']:.5f}")
        if test_stats["roc_auc"] > max_accuracy:
            max_accuracy = test_stats["roc_auc"]
            best_epoch = epoch
            if args.output_dir:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=f"best_run")
                
        print(f'Max roc_auc: {max_accuracy:.5f}')
        if log_writer is not None:
                log_writer.add_scalar('perf/test_acc1', test_stats['roc_auc'], epoch)
                log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)
      
    #### ############ ############ ############ ############
    print('####### Loading the best model ######')
    print(f'Loading epoch {best_epoch}')
    args.resume = os.path.normpath(os.path.join(args.output_dir,f'{args.remark}_checkpoint-best_run.pth'))
    misc.load_model(args,model_without_ddp=model_without_ddp,optimizer=optimizer,loss_scaler=loss_scaler)
    test_stats = evaluate(args,test_loader, model, device,criterion,print_analyzer=True)
    print(f"AUC_ROC of the network on the {len(test_loader)} test images: {test_stats['roc_auc']:.5f}, AP: {test_stats['ap']:.5f}, Accuracy: {test_stats['accuracy']:.5f}")
    ############ ############ ############ ############ ############
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    # Need mean accuracy and standard deviation of mult_exp_acc
    #print(f'{args.ds_name} has mean accuracy of {np.mean(mult_exp_acc)} with standard deviation of {np.std(mult_exp_acc)}')
    print('Done !!')

    # if args.output_dir:
    #     misc.save_model(
    #         args=args, model=model, model_without_ddp=model, optimizer=optimizer,
    #         loss_scaler=loss_scaler, epoch=epoch)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    args.data_dir = os.path.join(args.data_dir,args.ds_name)
    args.num_channel = DATASET_CONFIG[args.ds_name]["n_ch"]
    args.num_classes = DATASET_CONFIG[args.ds_name]["n_cl"]
   
    if "max_len" in DATASET_CONFIG[args.ds_name]:
        args.max_len = DATASET_CONFIG[args.ds_name]["max_len"]
    else:
        args.max_len = 390

    print('max len is: ',args.max_len)

    args.img_size = (args.max_len-3,65)
    args.task = DATASET_CONFIG[args.ds_name]["task"]
    #args.lr = DATASET_CONFIG[args.ds_name]["lr"]
    args.batch_size = DATASET_CONFIG[args.ds_name]["bs"]
    args.remark = args.remark + args.ds_name
    args.new_size = (int(args.img_size[0]//9),13)
    
    if "y_range" in DATASET_CONFIG[args.ds_name]:
        args.y_range = DATASET_CONFIG[args.ds_name]["y_range"]
    else:
        args.y_range = None 

    initial_timestamp = datetime.datetime.now()
    args.output_dir = os.path.join(args.output_dir,'Normwear_Downstream',args.ds_name,initial_timestamp.strftime("%Y-%m-%d_%H-%M"))
    args.log_dir = os.path.join(args.output_dir,'log')
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
