import os
import pickle
import numpy as np
from tqdm import tqdm
import json

# ===== HELPER function ======================================================================

'''
Unified Format:

{
    "uid": subject ID, 
    "data": [n_channel, L],
    "sampling_rate": 16000,
    "label": [
        {"class": label},
        {"reg": label},
        ...
    ]
}

'''

def noisemic_format(origin_data):
    # # check keys
    # print(origin_data.keys()) # ['uid', 'run_state', 'resp_rate', 'audio']
    # exit()
    
    # new format
    return {
        "uid": origin_data["uid"],
        "data": np.expand_dims(origin_data["data"], axis=0), # [1, L]
        "sampling_rate": 16000,
        "label": [
            {"class": origin_data['run_state']},
            {"reg": origin_data['resp_rate']}
        ]
    }

def kauh_format(origin_data):
    return {
        "uid": origin_data["patient_id"],
        "data": np.stack([origin_data["filter_E"], origin_data["filter_B"], origin_data["filter_D"]]), # [3, L]
        "sampling_rate": 16000,
        "label": [
            {"class": origin_data['label']},
        ]
    }

def respTR_format(origin_data):
    return {
        "uid": origin_data["subject_id"],
        "data": np.stack([origin_data[k] for k in origin_data.keys() if k not in ["subject_id", 'label']]), # [12, L]
        "sampling_rate": 16000,
        "label": [
            {"class": origin_data['label']},
        ]
    }

def fsd50k_format(origin_data):
    return {
        "uid": origin_data["subject_id"],
        "data": np.expand_dims(origin_data["audio_data"], axis=0), # [1, L]
        "sampling_rate": 16000,
        "label": [
            {"class": origin_data['breathing']},
            {"class": origin_data['cough']},
            {"class": origin_data['laughter']},
            {"class": origin_data['sneeze']},
            {"class": origin_data['speech']}
        ]
    }

def coswara_format(origin_data):
    all_keys = origin_data.keys()
    # for edge case
    for k in [
        "breathing-deep",
        "breathing-shallow",
        "cough-heavy",
        "cough-shallow",
        "counting-fast",
        "counting-normal",
        "vowel-a",
        "vowel-e",
        "vowel-o"
    ]:
        if k not in all_keys:
            return None
    
    # regular return
    return {
        "uid": origin_data["subject_id"],
        "data": np.stack([origin_data[k] for k in [
            "breathing-deep",
            "breathing-shallow",
            "cough-heavy",
            "cough-shallow",
            "counting-fast",
            "counting-normal",
            "vowel-a",
            "vowel-e",
            "vowel-o"
        ]]), # [1, L]
        "sampling_rate": 16000,
        "label": [{"class": origin_data[k]} for k in [
            "smoker",
            "cold",
            "ht",
            "diabetes",
            "cough",
            "diarrhoea",
            "fever",
            "loss_of_smell",
            "bd",
            "st",
            "ihd",
            "asthma",
            "cld",
            "pneumonia",
            "ftg",
            "mp",
        ]]
    }

def wearable_format(origin_data, task_type='class'):
    return {
        "uid": "",
        "data": origin_data["tss"], # TODO
        "sampling_rate": 65,
        "label": [{task_type: origin_data["label"]}] # TODO
    }

# def gameemo_format(origin_data, task_type='class'):
#     # print(origin_data.keys())
#     # print(origin_data["label"])
#     # exit()
#     return {
#         "uid": "",
#         "data": origin_data["tss"], # TODO
#         "sampling_rate": 65,
#         "label": [{task_type: origin_data["label"]}] # TODO
#     }

# ==============================================================================================

def clean_data(dataset='audio_downstream/NoseMic', sample_folder="sample", task_type='class'):
    # init
    root = "../data/{}/{}".format(dataset, sample_folder)
    save_to = "../data/{}/sample_for_downstream".format(dataset)
    os.makedirs(save_to, exist_ok=True)

    # iterate over files
    for fn in tqdm(sorted(os.listdir(root))):
        if fn[0] == ".":
            continue

        # # if exist, pass
        # if os.path.isfile(os.path.join(save_to, fn)):
        #     continue

        with open(os.path.join(root, fn), 'rb') as f:
            origin_data = pickle.load(f)
        
        # new format
        # new_data = noisemic_format(origin_data)
        # new_data = kauh_format(origin_data)
        # new_data = respTR_format(origin_data)
        # new_data = fsd50k_format(origin_data)
        # new_data = coswara_format(origin_data)
        new_data = wearable_format(origin_data, task_type=task_type)

        # new_data = gameemo_format(origin_data, task_type=task_type)

        if new_data is None:
            continue
        
        # # check
        # print(new_data['data'].shape)
        # print(new_data['label'])
        # exit()

        # save
        with open(os.path.join(save_to, fn), 'wb') as f:
            pickle.dump(new_data, f)

def clean_train_test_split_format():
    read_old = lambda s: pickle.load(open("{}/splits".format(s), 'rb'))
    clean_old = lambda s: {"train": [fn.replace("\\", "/").split("/")[-1] for fn in s["train_fnames"]], "test": [fn.replace("\\", "/").split("/")[-1] for fn in s["test_fnames"]]}
    write_new = lambda s, name: json.dump(s, open("{}/train_test_split.json".format(name), 'w'))
    clean_one = lambda s: write_new(clean_old(read_old(s)), s)

    # clean_one("PPG_DM")

def check_bound_for_reg(dataset="clinic_data/opioid_misuse"):
    ranges = dict()
    for fn in tqdm(os.listdir(os.path.join("../data", dataset, "sample_for_downstream"))):
        # fn = fn_obj.name

        with open(os.path.join("../data", dataset, "sample_for_downstream", fn), 'rb') as f:
            data = pickle.load(f)
        label = data["label"]
        for l_i in range(len(label)):
            if label[l_i].get("reg") is None:
                continue
            else:
                val = label[l_i]["reg"]
                # print(label)
                # exit()
                if ranges.get(l_i) is None:
                    ranges[l_i] = [val, val]
                ranges[l_i][0] = min(ranges[l_i][0], val)
                ranges[l_i][1] = max(ranges[l_i][1], val)
    
    for i in ranges:
        print(i, ":", ranges[i])

def split_by_subject_id(dataset="clinic_data/studentlife", check_label_dist=False):
    import matplotlib.pyplot as plt

    # collect data for each subject
    sub_to_fns = dict()
    labels = dict()
    for fn_obj in os.scandir(os.path.join("data", dataset, "sample_for_downstream")):
        fn = fn_obj.name
        sub_id = fn.split("_")[0]

        if sub_to_fns.get(sub_id) is None:
            sub_to_fns[sub_id] = list()
            labels[sub_id] = list()
        
        if check_label_dist:
            with open(os.path.join("data", dataset, "sample_for_downstream", fn), 'rb') as f:
                data = pickle.load(f)
            labels[sub_id].append(data["label"][-1]["class"]) # -1, -3, 0
        
        sub_to_fns[sub_id].append(fn)
    
    # split the subjects
    all_subs = [k for k in sub_to_fns.keys()]
    np.random.shuffle(all_subs)

    split_p = int(0.8*len(all_subs))
    train_subs = all_subs[:split_p]
    test_subs = all_subs[split_p:]

    # integrate file names
    train_labels, test_labels = list(), list()
    train_test_split = {"train": list(), "test": list()}
    for sub in train_subs:
        train_test_split["train"] += sub_to_fns[sub]
        if check_label_dist:
            train_labels += labels[sub]
    for sub in test_subs:
        train_test_split["test"] += sub_to_fns[sub]
        if check_label_dist:
            test_labels += labels[sub]
    # print(train_labels)
    # print(test_labels)
    
    if check_label_dist:
        fig = plt.figure()

        plt.subplot(2, 1, 1)
        plt.hist(train_labels)
        plt.title("Train distribution")

        plt.subplot(2, 1, 2)
        plt.hist(test_labels)
        plt.title("Test distribution")
        
        plt.show()
    
    # # check
    print(len(train_test_split["train"]), len(train_test_split["test"]))
    # exit()
    # for sub in sub_to_fns:
    #     print(sub, ":", len(sub_to_fns[sub]))
    
    # write to file
    with open(os.path.join("data", dataset, 'train_test_split.json'), 'w') as f:
        json.dump(train_test_split, f)

if __name__ == '__main__':
    pass
    # python3 -m src.downstream.data_clean

    # clean_data(dataset='downstream/PPG_HTN', sample_folder="samples", task_type='class')
    # clean_data(dataset='downstream/PPG_DM', sample_folder="samples", task_type='class')
    # clean_data(dataset='downstream/PPG_CVA', sample_folder="samples", task_type='class')
    # clean_data(dataset='downstream/PPG_CVD', sample_folder="samples", task_type='class')
    # clean_data(dataset='downstream/indian-fPCG', sample_folder="samples", task_type='reg')
    # clean_data(dataset='downstream/ppg_hgb', sample_folder="samples", task_type='reg')
    # clean_data(dataset='downstream/non_invasive_bp', sample_folder="samples", task_type='reg')
    # clean_data(dataset='downstream/drive_fatigue', sample_folder="samples", task_type='class')
    # clean_data(dataset='downstream/ecg_heart_cat', sample_folder="samples", task_type='class')
    # clean_data(dataset='downstream/gameemo', sample_folder="samples", task_type='class')
    # clean_data(dataset='downstream/uci_har', sample_folder="samples", task_type='class')
    # clean_data(dataset='downstream/wesad', sample_folder="samples", task_type='class')