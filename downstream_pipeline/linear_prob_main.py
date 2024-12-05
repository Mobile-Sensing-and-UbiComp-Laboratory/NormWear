import os
import json
import pickle
import argparse
from tqdm import tqdm

from .task_specification import *
from .corrected_linear_prob import *

# # basic setting
# np.random.seed(42)
# torch.manual_seed(42)

# DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
# print("DEVICE:", DEVICE)

# get command line arguments
def get_args_parser():
    parser = argparse.ArgumentParser('HealthMAE linear probing', add_help=False)
    
    # basic parameters
    parser.add_argument('--remark', default='',
                        help='model_remark')
    parser.add_argument('--prepare_embed', type=int, default=0)
    parser.add_argument('--group', type=int, default=0)
    
    # Model parameters
    parser.add_argument('--model_weight_dir', default='../data/results', type=str,
                        help='path of model weight')
    parser.add_argument('--model_name', type=str,
                        help='Name of model to train')
    parser.add_argument('--eval', default=False,type=bool,
                        help='Perform evaluation only')
    parser.add_argument('--log_dir', default='../../data/audio_results/log',
                        help='path where to tensorboard log')
    parser.add_argument('--num_runs', default=1, type=int,
                        help='number of runs to perform')

    # Optimizer parameters
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')

    # Dataset parameters
    parser.add_argument('--data_path', default='../data', type=str,
                        help='dataset path')
    parser.add_argument('--ds_name', type=str,
                        help='dataset name')
    parser.add_argument('--output_dir', default='../data/audio_results/linprob_ckpt',
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

# def launch_linear_prob(model_name, dataset, repeat_num=3):
def launch_linear_prob(args, embed_root, all_fns):
    # get task types
    # print("Fetching basic information...")
    with open(os.path.join(embed_root, all_fns[0]), 'rb') as f:
        sample = pickle.load(f)

    label_types = [list(k.items())[0][0] for k in sample['label']] # ['class', 'class', 'reg', ...]
    feature_size = len(sample['embed'])

    del sample # clear cache

    # get split
    # print("Fetching data split...")
    split = json.load(open("../data/{}/train_test_split.json".format(args.ds_name)))

    # load data
    x_train, y_train, x_test, y_test = list(), dict(), list(), dict()
    print("Loading {} Embedding for {} ...".format(args.model_name, args.ds_name))
    for fn in tqdm(os.listdir(embed_root)):
        with open(os.path.join(embed_root, fn), 'rb') as f:
            data = pickle.load(f)

        if fn in split['test']:
            x_test.append(data['embed'])
            for task_idx in range(len(label_types)):
                if y_test.get(task_idx) is None:
                    y_test[task_idx] = list()
                y_test[task_idx].append(data['label'][task_idx][label_types[task_idx]])
            # y_test.append(data['label'][task_idx][task_type])
        else:
            x_train.append(data['embed'])
            for task_idx in range(len(label_types)):
                if y_train.get(task_idx) is None:
                    y_train[task_idx] = list()
                y_train[task_idx].append(data['label'][task_idx][label_types[task_idx]])
            # y_train.append(data['label'][task_idx][task_type])
    
    x_train = np.nan_to_num(np.stack(x_train))
    x_test = np.nan_to_num(np.stack(x_test))
    # y_train = np.stack(y_train)
    # y_test = np.stack(y_test)
    for task_idx in range(len(label_types)):
        y_train[task_idx] = np.stack(y_train[task_idx])
        y_test[task_idx] = np.stack(y_test[task_idx])

    # for score log
    task_scores = dict()

    for type_i in range(len(label_types)):
        print("{}: Task {}/{}".format(args.ds_name, type_i+1, len(label_types)))
        for n_ in tqdm(range(args.num_runs)):
            # print("{}: Training for task {}/{}, run {}/{}...".format(args.ds_name, type_i+1, len(label_types), n_+1, args.num_runs))
            # score = linear_prob_one_task(
            #     args,
            #     task_idx=type_i,
            #     task_type=label_types[type_i],
            #     train_fns=split['train'],
            #     val_fns=split['test'],
            #     root_path=embed_root,
            #     feature_size=feature_size,
            #     num_classes=CLASS_NUM[args.ds_name]["nums"][type_i],
            # )

            # TODO if args.group != 0: use old pipeline

            score = linear_prob(
                x_train,
                y_train[type_i],
                x_test,
                y_test[type_i],
                task_type=label_types[type_i], 
                random_state=n_
            )

            # save the score
            if task_scores.get(CLASS_NUM[args.ds_name]["names"][type_i]) is None:
                task_scores[CLASS_NUM[args.ds_name]["names"][type_i]] = list()
            task_scores[CLASS_NUM[args.ds_name]["names"][type_i]].append(score)

        print("Curr scores:", [round(100*s[0], 3) for s in task_scores[CLASS_NUM[args.ds_name]["names"][type_i]]])
    
    # log
    for task in task_scores:
        # task_scores[task]: [n, num_score]
        all_scores_m = 100*np.mean(np.array(task_scores[task]), axis=0) # num_score
        all_scores_s = 100*np.std(np.array(task_scores[task]), axis=0) # num_score

        scores = ""
        if len(all_scores_m) > 1:
            print("Scores following order: AUC ROC, AP, Accuracy, Precision, Recall, F1-score")
        else:
            print("Mean relative accuracy:")
        
        # print
        for s_i in range(len(all_scores_m)):
            scores += "{} +- {}, ".format(
                round(all_scores_m[s_i], 3),
                round(all_scores_s[s_i], 3),
            )
        
        print(task, scores)
    
    return task_scores

# def linear_prob_one_task(args, task_idx=0, task_type='class', train_fns=list(), val_fns=list(), root_path="", feature_size=512, num_classes=2):
#     # init linear prob
#     lp_model = LinearProb(
#         lr=args.blr if task_type == 'class' else 5e-3, # 1e-3
#         weight_decay=1e-5, # 1e-5
#         epochs=args.epochs, # 50
#         batch_size=args.batch_size, # 64
#         step_size=10, # 10
#         # data related
#         num_classes=num_classes,
#         feature_size=feature_size,
#         out_range = None if task_type == 'class' else CLASS_NUM[args.ds_name]["ranges"][task_idx]
#     )

#     # fit model
#     record = lp_model.fit(
#         task_idx,
#         train_fns, 
#         val_fns=val_fns, 
#         task_type=task_type, 
#         root_path=root_path, 
#         device=DEVICE
#     )

#     return record["score"]

    # save record

if __name__ == '__main__':
    # get command line arguments
    args = get_args_parser()
    args = args.parse_args()

    # python -m src.downstream.linear_prob_main --ds_name wearable_downstream/uci_har --model_name normwear --num_runs 100
    
    launch_linear_prob(args)