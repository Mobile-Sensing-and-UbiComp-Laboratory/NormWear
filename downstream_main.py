from .downstream_pipeline.prepare_embeds import *
from .downstream_pipeline.linear_prob_main import *
import gc

from pathlib import Path

# reset print format
import sys
sys.stdout = sys.__stdout__

# get command line arguments
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    save_remark = args.remark if len(args.remark) > 0 else args.model_name
    save_fname = "{}_results_all.pkl".format(save_remark)
    save_name = "../data/results/downstream_results/{}".format(save_fname) # full path
    pod_name = "physio-model-5d98486f85-ngv5t"

    DATA_GROUPS = {
        0: [ # all downstream
            "wearable_downstream/PPG_HTN",
            "wearable_downstream/PPG_DM",
            "wearable_downstream/PPG_CVA",
            "wearable_downstream/PPG_CVD",
            "wearable_downstream/non_invasive_bp", 
            "wearable_downstream/ppg_hgb", 
            "wearable_downstream/indian-fPCG",
            "wearable_downstream/ecg_heart_cat", # **
            "wearable_downstream/emg-tfc",
            "wearable_downstream/gameemo", # **
            "wearable_downstream/Epilepsy",
            "wearable_downstream/drive_fatigue", # *
            "wearable_downstream/uci_har", # ***
            "wearable_downstream/wesad", # ***
        ],
        1: [ # if test a subset of the tasks, specified them here, e.g.:
            "wearable_downstream/PPG_CVD",
            "wearable_downstream/ppg_hgb", 
        ]
    }

    # main iteration
    for dataset in tqdm(DATA_GROUPS[args.group]):

        # manually update the dataset's name
        args.ds_name = dataset

        # prepare embedding (comment out if embedding was ready)
        embed_root = '{}/{}/{}_wav_embed'.format(args.data_path, args.ds_name, save_remark)
        sample_root = '{}/{}/sample_for_downstream'.format(args.data_path, args.ds_name)
        all_sample_fns = os.listdir(sample_root)

        # check if embedding ready
        embedding_ready = True
        if not os.path.isdir(embed_root):
            embedding_ready = False
        else:
            all_fns = sorted(os.listdir(embed_root))
            if abs(len(all_fns) - len(all_sample_fns)) > 1:
                embedding_ready = False

        # prepare embed if not ready
        if args.prepare_embed or not embedding_ready:
            print("Preparing Embedding for {} ...".format(args.ds_name))
            root_prefix = "" if args.data_path == "data" else "../"
            audio_embedding_prepare(model_name=args.model_name, data_rootpath=dataset, root_prefix=root_prefix, remark=save_remark)
            # combine_normwear_ast(data_rootpath=dataset, root_prefix=root_prefix)

            # fetch all file names
            all_fns = sorted(os.listdir(embed_root))

        # linear prob
        # curr_res = main(args)
        curr_res = launch_linear_prob(args, embed_root, all_fns)

        # save the results
        # fetch previous results
        all_results = dict()
        if Path(save_name).is_file():
            try:
                with open(save_name, 'rb') as f:
                    all_results = pickle.load(f)
            except:
                all_results = dict()
        
        all_results[dataset] = curr_res

        # write new results
        with open(save_name, 'wb') as f:
            pickle.dump(all_results, f)
        
        # break
        # cleanup
        gc.collect()
        torch.cuda.empty_cache()

        # log for file transfer
    print("kubectl cp {}:data/results/downstream_results/{} data/results/{}".format(pod_name, save_fname, save_fname))

    # python3 -m downstream_all --model_name healthmae --num_runs 2 --group 2 --prepare_embed 1 --batch_size 256
    # CUDA_VISIBLE_DEVICES=1 python3 -m downstream_all --model_name healthmae --num_runs 2 --group 2 --prepare_embed 1 --batch_size 256

    # kubectl cp physio-model-zs-589f54b445-b6vrs:data/results/downstream_results/stats_results_all.pkl data/results/stats_results_all.pkl