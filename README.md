<div align="center">
  
# 🚀 NormWear: A Foundation Model for Multivariate Wearable Sensing of Physiological Signals.

[![preprint](https://img.shields.io/static/v1?label=arXiv&message=TBD&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/TBD)
[![ckpt-release](https://img.shields.io/badge/Github_Release-Model_Checkpoint-black?logo=github&logoColor=white)](https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear/releases/tag/v1.0.0-alpha)
[![downstream](https://img.shields.io/badge/Google_Drive-Downstream_Datasets-blue?logo=googledrive&logoColor=fff)](https://drive.google.com/file/d/1Mojf_iby8FnUydogwUE-b321fB1V6SGK/view?usp=sharing)

</div>

## ✨ Introduction
This is the official implementation of the paper [Toward Foundation Model for Multivariate Wearable Sensing of Physiological Signals](https://TBD).

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1kLJ9xc7c864mlPzjSo-HCiGRpzdjSMEO" width="100%">
  <br />
</p>

Time-series foundation models have the ability to run inference, mainly forecasting, on any type of time series data, thanks to the informative representations comprising waveform features. Wearable sensing data, on the other hand, contain more variability in both patterns and frequency bands of interest and generally emphasize more on the ability to infer healthcare-related outcomes. The main challenge of crafting a foundation model for wearable sensing physiological signals is to learn generalizable representations that support efficient adaptation across heterogeneous sensing configurations and applications. In this work, we propose NormWear, a step toward such a foundation model, aiming to extract generalized and informative wearable sensing representations. NormWear has been pretrained on a large set of physiological signals, including PPG, ECG, EEG, GSR, and IMU, from various public resources. For a holistic assessment, we perform downstream evaluation on 11 public wearable sensing datasets, spanning 18 applications in the areas of mental health, body state inference, vital sign estimations, and disease risk evaluations. We demonstrate that NormWear achieves a better performance improvement over competitive SoTA baselines with different modeling strategies. In addition, leveraging a novel representation-alignment-match-based method, we align physiological signals embeddings with text embeddings. This alignment enables our proposed foundation model to perform zero-shot inference, allowing it to generalize to previously unseen wearable signal-based health applications.

## 📈 Usage

Download: 

```sh
# Clone the repository
git clone git@github.com:Mobile-Sensing-and-UbiComp-Laboratory/NormWear.git
```
The python libraries dependencies are specified in `NormWear/dependencies.txt`, with an example bash script `NormWear/config_env.sh`. 

The pretrained model checkpoint can be found in [Release](https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear/releases/tag/v1.0.0-alpha).

### Extracting Encoder Embeddings

An example showing how to get signal embedding using NormWear:

```python
import torch
from NormWear.main_model import NormWearModel

# config
device = torch.device('cpu')

### init model ##################################################################
weight_path = "path to checkpoint here"
model = NormWearModel(weight_path=weight_path, optimized_cwt=True).to(device)

# generate data
# test example: 2 samples, 3 sensor, sequence length of 2 seconds
# data shape notation: bn for batch size, nvar for number of sensor channels,
# P for number of patches, E for embedding dimension (768)
sampling_rate = 64
x = torch.rand(2, 3, sampling_rate*2).to(device)

# encoding
out = model.get_embedding(x, sampling_rate=sampling_rate, device=device) # bn, nvar, P, E

# log
print("Input shape:", x.shape) # [2, 3, 128]
print("Output shape:", out.shape) # [2, 3, P, 768]

### Example of aggregation across patches ########################################
# Take [CLS]
# embedding = out[:, :, 0, :] # [2, 3, 768]

# Mean pooling
# embedding = out.mean(dim=2) # [2, 3, 768]

### Example of aggregation across channels #######################################
# Concat
# final_embedding = embedding.flatten(start_dim=1) # [2, 3*768]

# Mean
# final_embedding = embedding.mean(dim=1) # [2, 768]
```

### Zero shot inference

```python
import torch
from NormWear.zero_shot.msitf_fusion import NormWearZeroShot

# config
device = torch.device('cpu')

### init model ##################################################################
# weight_path = "weight path here"
# msitf_ckpt = "fusion weight path here"

model = NormWearZeroShot(
    weight_path=weight_path, 
    msitf_ckpt=msitf_ckpt
).to(device)

### generate data ###############################################################
sampling_rate = 64
x = torch.rand(2, 3, sampling_rate*2).to(device) # test example: 2 samples, 3 sensor, sequence length of 2 seconds

# setup question and options
task_of_interest = [
    "Is the person doing good?"
]

options = [
    "The person is doing so so",
    "The person is under stress.",
    "The person is having fun.",
]

### Inference ###################################################################
# text encoding
txt_embed = model.txt_encode(task_of_interest+options) # [4, 2048]

query_embed, option_embed = txt_embed[:1, :], txt_embed[1:, :]

# signal encoding
signal_embed = model.signal_encode(x, query_embed, sampling_rate=sampling_rate) # bn, nvar, P, E

# calculate distance
pred_prob = model.inference(signal_embed, option_embed) # bn, num_options
pred = torch.argmax(pred_prob, dim=1).numpy()

### check output ################################################################
for i in range(len(x)):
    print("Sample {} inference: {}".format(i, options[pred[i]]))

# log
print("Input shape:", x.shape) # [2, 3, 128]
print("Text embed shape:", txt_embed.shape) # [4, 2048]
print("Signal embed shape:", signal_embed.shape) # [2, 3, P, 768]
print("Inference probability shape:", pred_prob.shape) # # [2, num_options]
```

> [!TIP]  
> For the best performance, conduct prompt engineering on the query could be a good practice. 

## 🔥 Pre-training
To run the evaluation on the downstream datasets, run the following command:
```sh
python3 -m NormWear.pretrain_main
```


| Required Parameter         | Type      | Default   | Description                                                     |
|---------------------|-----------|-----------|-----------------------------------------------------------------------|
| `--batch_size`      | `<int>`  | `4` | Number of input samples for each round of forward and backpropagation.       |
| `--epochs`          | `<int>`  | `400` | Number of epochs.                        |
| `--blr`             | `<float>`  | `1e-3` | base learning rate: absolute_lr = base_lr * total_batch_size / 256 |
| `--warmup_epochs`   | `<int>`  | `10` | Number of epochs to increase learning rate from a small number to `blr`.  |
| `--target_len`      | `<int>`  | `388` | Sequence length of the reconstructed time series |
| `--weight_decay`   | `<float>`  | `5e-2` | Regularization coefficient of L2 penalty on model parameters. |
| `--mask_scheme`   | `<str>`  | `random` | Masking scheme for pretext task (choose from 'random' and 'struc')  |
| `--mask__prob`   | `<float>`  | `0.8` | Masking ratio for random masking  |
| `--mask_t_prob`   | `<float>`  | `0.6` | Masking ratio along time axis of the scalogram. |
| `--mask_f_prob`   | `<float>`  | `0.5` | Masking ratio along scale axis of the scalogram.  |
| `--remark`   | `<string>`  | `""` | Name to mark the current experimental trail. |

An example bash command would be:
```sh
torchrun --nproc_per_node=4 --master_port=12345 -m NormWear.pretrain_main --batch_size 256 --epochs 100 --warmup_epochs 4 --remark test_run
```


## ❄️ Downstream Evaluation

To run the evaluation on the downstream datasets, run the following command:
```sh
python3 -m NormWear.downstream_main
```


| Required Parameter         | Type      | Default   | Description                                                                 |
|--------------------------|-----------|-----------|-----------------------------------------------------------------------------|
| `--model_name`           | `<string>`  | `normwear` | Supported models are [stats, chronos, clap, tfc, normwear]                                 |
| `--model_weight_dir`     | `<string>`  | `""` | Path to the model checkpoint, only `normwear` need this parameter.                           |
| `--group`                 | `<int>`  | `0` | Run a group of downstream tasks. The group can be customized in `NormWear/downstream_main.py` |
| `--data_path`   | `<string>`  | `../data` | Root path for where the downstream data is placed. |
| `--num_runs`   | `<int>`  | `1` | Number of repetition for running the evaluation of each task. |
| `--prepare_embed`   | `<int>`  | `1` | Run the inference and save the embeddings before sending them to evaluation. Set to `1` to overwrite previous saved embedding if there are any. |
| `--remark`   | `<string>`  | `""` | Name to mark the current experimental trail. By default, all the embeddings are stored under a subfolder named as `--model_name` under the folder of each downstream data. |

An example bash command would be:
```sh
CUDA_VISIBLE_DEVICES=0 python3 -m NormWear.downstream_main --model_name normwear --model_weight_dir data/results/normwear_last_checkpoint-15470-correct.pth --group 0 --data_path ../data --num_runs 1 --prepare_embed 1 --remark test_run
```

#### :floppy_disk: The processed clean downstream datasets can be downloaded from [here](https://drive.google.com/file/d/1Mojf_iby8FnUydogwUE-b321fB1V6SGK/view?usp=sharing). 

### ✏️ For adding a downstream dataset, please following the format:
```bash
name_of_new_dataset/
├── sample_for_downstream/
│   ├── name_of_sample_0.pkl
│   ├── name_of_sample_1.pkl
│   ├── name_of_sample_2.pkl
│   └── ...
└── train_test_split.json
```
where the content in each file requires content in the following format:
```python
# data content in name_of_sample_i.pkl
{
    "uid": "", # subject ID identifier
    "data": np.array(sensor_data_here).astype(np.float16), # float numpy array with shape of [num_channels, sequence_length]
    "sampling_rate": 64, # put the correct sampling rate of the sensor signal here. 
    "label": [ # label for each task on this dataset. 'class' for classification and 'reg' for regression.
        {"class": label}, 
        {"reg": label},
        ...
    ]
}

# data content in train_test_split.json
{
  'train': ["name_of_sample_i.pkl", "name_of_sample_j.pkl", "name_of_sample_k.pkl", ...],
  'test': ["name_of_sample_l.pkl", "name_of_sample_m.pkl", "name_of_sample_n.pkl", ...]
}
```
### 🔥 Finetuning model for downstream datasets
To run the evaluation on the downstream datasets, run the following command:
```sh
python -m main_finetune --ds_name <dataset_name> --checkpoint <weight_path>
```
For multiple gpus, run:
```sh
torchrun --nproc_per_node=<number of device> main_finetune.py --world_size <number of device> --ds_name <dataset_name> --checkpoint <weight_path>
```
## 📝 Citation

If you find NormWear model useful for your research, please consider citing the associated [paper]([https://arxiv.org/abs/2403.07815](https://TBD)):

```
@misc{tbd,
      title={Toward Foundation Model for Multivariate Wearable Sensing of Physiological Signals}, 
      author={Yunfei Luo, Yuliang Chen, Asif Salekin, Tauhidur Rahman},
      year={2024},
      eprint={tbd},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/tbd}, 
}
```
