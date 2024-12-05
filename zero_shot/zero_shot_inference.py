import os
import gc
import sys
import json
import pickle
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# from src.downstream.model_apis import *
from .msitf_fusion import *
from ..downstream_pipeline.task_specification import *

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
print("DEVICE:", DEVICE)

# ============= helper functions ================================================
def load_model(model_name='normwear'):
    # all models should follows the function structure of AST_API
    if model_name == 'normwear':
        # model = NormWearZeroShot() # random init
        # msitf_ckpt = "../data/audio_results/normwear_msitf/normwear_msitf_checkpoint-correct-incorrect.pth"
        # msitf_ckpt = "../data/audio_results/normwear_msitf/normwear_msitf_checkpoint-ctr.pth"
        # msitf_ckpt = "../data/audio_results/normwear_msitf/normwear_msitf_checkpoint-1.pth"
        msitf_ckpt = "../data/audio_results/normwear_msitf_clean/normwear_msitf_clean_checkpoint-5.pth"
        # msitf_ckpt = "../data/audio_results/normwear_msitf_clean/normwear_msitf_clean_checkpoint-15.pth"
        # msitf_ckpt = ""
        model = NormWearZeroShot(msitf_ckpt=msitf_ckpt)
    # elif model_name == 'clap':
    #     model = CLAP_API()
    else:
        print("Model not supported. ")
        exit()
    
    # return
    model = model.to(DEVICE)
    model.eval()

    # # check number of parameters
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"{model_name} Number of parameters: {total_params}")
    # exit()

    return model

def zs_inference(
    ds_name="wearable_downstream/PPG_HTN", 
    model_name='normwear-msitf', 
    task_idx=0,
    root_prefix="../", 
    batch_size=1024
):
    
    # construct model
    model = load_model(model_name=model_name)

    split = json.load(open("../data/{}/train_test_split.json".format(ds_name)))

    # get embedding for each sample
    tasks_type = dict()
    embeds_all, labels_all = list(), dict()
    curr_samples, batch_num = list(), 0
    for fn in tqdm(sorted(os.listdir('{}data/{}/sample_for_downstream'.format(root_prefix, ds_name)))):
        # edge case
        if fn[0] == '.' or fn not in split['test']:
            continue
        
        # load sample
        # read data
        with open(os.path.join('{}data/{}/sample_for_downstream'.format(root_prefix, ds_name), fn), 'rb') as f:
            sample = pickle.load(f) # ['uid', 'data', 'label', 'sampling_rate']
        
        # expand 1 dimension if only single dimension
        if len(sample['data'].shape) == 1:
            sample['data'] = np.expand_dims(sample['data'], axis=0) # nvar, L
        
        # store for batch operation
        task_name = CLASS_NUM[ds_name]['names'][task_idx]
        if labels_all.get(task_name) is None:
            labels_all[task_name] = list()
            tasks_type[task_name] = [k for k in sample['label'][task_idx].keys()][0]
        labels_all[task_name].append(sample['label'][task_idx][tasks_type[task_name]])
        curr_samples.append(sample['data'])
        batch_num += 1
        if batch_num < batch_size:
            continue

        # batch operation

        # test clap pipeline
        with torch.no_grad():
            embeds_all += model(
                torch.from_numpy(np.stack(curr_samples)).float().to(DEVICE), 
                [(ds_name, CLASS_NUM[ds_name]['names'][task_idx])],
                label=None
            ).cpu().detach().numpy().tolist() # bn, E
        
        # refresh vars
        curr_samples, batch_num = list(), 0
    
    # calculate last batch
    if len(curr_samples) > 0:
        with torch.no_grad():
            embeds_all += model(
                torch.from_numpy(np.stack(curr_samples)).float().to(DEVICE), 
                [(ds_name, CLASS_NUM[ds_name]['names'][task_idx])],
                label=None
            ).cpu().detach().numpy().tolist() # bn, E
    
    # calculate predictions
    embeds_all = torch.tensor(embeds_all).to(DEVICE) # N, E

    for k in labels_all:
        # if has more than 1 dimensions
        if isinstance(labels_all[k][0], np.ndarray):
            if len(labels_all[k][0].shape) >= 1:
                labels_all[k] = [tuple(l) for l in labels_all[k]]
            else:
                labels_all[k] = [int(l) for l in labels_all[k]]
    
        # get choices
        label_name_map = [l for l in set(labels_all[k])]

        # text encoding
        choice_embeds = txt_encode(
            task=[(ds_name, CLASS_NUM[ds_name]['names'][task_idx])], 
            label=label_name_map, 
            model=model, 
            task_type=tasks_type[k]
        ) # num_label, E

        # label map, y_true, distance, task_type
        scores = zs_evaluate(
            sensor_embeds=embeds_all, # tensor
            choice_embeds=choice_embeds, # tensor
            label_name_map=label_name_map, # dict
            task_type=tasks_type[k], # str
            y_trues=np.array(labels_all[k]) # np array
        )

        print("Evaluation scores on {}:".format(ds_name, k))
        scores = [round(s*100, 3) for s in scores]
        print(scores)

def zs_evaluate(
    sensor_embeds=None,
    choice_embeds=None,
    label_name_map=None,
    task_type=None,
    y_trues=None
):  
    # L1 distance
    distances = torch.abs(sensor_embeds[:, None, :] - choice_embeds[None, :, :]).sum(dim=-1) # bn, num_choice
    # distances = 1 / torch.matmul(sensor_embeds, choice_embeds.T)  # bn, num_choice
    # distances = distances + (0.5*dt_distances)

    # # check
    # print(distances.shape)
    # exit()

    if task_type == "reg":
        y_preds = np.array([label_name_map[idx] for idx in torch.argmin(distances, dim=1).cpu().numpy()]) # bn
        return [1 - np.mean(np.absolute(y_trues - y_preds) / y_trues)]
    else:
        sims = distances
        sims = 1 - (sims / torch.sum(sims, dim=1, keepdim=True))
        sims = torch.nan_to_num(sims) + 1e-8 # bn, num_choice
        
        y_preds = nn.functional.softmax(sims.float(), dim=1).detach().cpu().numpy() # bn, num_choice
    
        print("Classes in Test:", set(y_trues))
        if len(set(y_trues)) <= 2:
            return [roc_auc_score(y_trues, y_preds[:, 1])]
        else:
            # for i in range(len(y_trues)):
            #     print(y_trues[i], np.argmax(y_preds[i]))
            # print(y_trues, y_preds)
            return [roc_auc_score(y_trues, y_preds, multi_class="ovo", average="macro")]

if __name__ == '__main__':
    # python3 -m src.zero_shot.zero_shot_inference clap
    # python3 -m src.zero_shot.zero_shot_inference normwear

    # input model name
    model_name = sys.argv[1] # normwear

    # gc.collect()
    # torch.cuda.empty_cache()

    ds_names = [
        "wearable_downstream/PPG_HTN",
        "wearable_downstream/PPG_DM",
        "wearable_downstream/PPG_CVA",
        "wearable_downstream/PPG_CVD",
        "wearable_downstream/non_invasive_bp", 
        "wearable_downstream/ppg_hgb", 
        "wearable_downstream/indian-fPCG",
        "wearable_downstream/ecg_heart_cat", # **
        "wearable_downstream/drive_fatigue", # *
        "wearable_downstream/gameemo", # **
        "wearable_downstream/uci_har", # ***
        "wearable_downstream/wesad", # ***
        "wearable_downstream/emg-tfc",
        "wearable_downstream/Epilepsy",
    ]
    for d_i in range(len(ds_names)):
        task_idx = 0
        if ds_names[d_i] == "wearable_downstream/Epilepsy":
            ds_names[d_i] = (ds_names[d_i], 0)
            for d_j in range(1, 5):
                ds_names.append(("wearable_downstream/Epilepsy", d_j))
        else:
            ds_names[d_i] = (ds_names[d_i], 0)
    
    # launch zero shot evaluation
    for ds in ds_names:
        print(ds)
        ds_name, task_idx = ds

        zs_inference(
            ds_name=ds_name, 
            model_name=model_name, 
            task_idx=task_idx,
            root_prefix="", # "../" if data not in pod, else ""
            batch_size=8
        )