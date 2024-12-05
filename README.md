<div align="center">
  
# 🚀 NormWear: A Foundation Model for Multivariate Wearable Sensing of Physiological Signals.

[![preprint](https://img.shields.io/static/v1?label=arXiv&message=TBD&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/TBD)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-FFD21E)](https://huggingface.co/datasets/TBD)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)](https://huggingface.co/collections/TBD)
[![License: MIT](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

## ✨ Introduction
This is the official implementation of the paper [Toward Foundation Model for Multivariate Wearable Sensing of Physiological Signals](https://TBD).

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1kLJ9xc7c864mlPzjSo-HCiGRpzdjSMEO" width="100%">
  <br />
</p>

Time-series foundation models have the ability to run inference, mainly forecasting, on any type of time series data, thanks to the informative representations comprising waveform features. Wearable sensing data, on the other hand, contain more variability in both patterns and frequency bands of interest and generally emphasize more on the ability to infer healthcare-related outcomes. The main challenge of crafting a foundation model for wearable sensing physiological signals is to learn generalizable representations that support efficient adaptation across heterogeneous sensing configurations and applications. In this work, we propose NormWear, a step toward such a foundation model, aiming to extract generalized and informative wearable sensing representations. NormWear has been pretrained on a large set of physiological signals, including PPG, ECG, EEG, GSR, and IMU, from various public resources. For a holistic assessment, we perform downstream evaluation on 11 public wearable sensing datasets, spanning 18 applications in the areas of mental health, body state inference, biomarker estimations, and disease risk evaluations. We demonstrate that NormWear achieves a better performance improvement over competitive baselines in general time series foundation modeling. In addition, leveraging a novel representation-alignment-match-based method, we align physiological signals embeddings with text embeddings. This alignment enables our proposed foundation model to perform zero-shot inference, allowing it to generalize to previously unseen wearable signal-based health applications.

## 📈 Usage

Download: 

```sh
# Clone the repository
git clone git@github.com:Mobile-Sensing-and-UbiComp-Laboratory/NormWear.git

# Install in editable mode with extra training-related dependencies
cd NormWear && pip install --editable ".[TBD]"
```

The pretrained model checkpoint can be found in Release [TBD].

> [!TIP]  
> This repository is intended for research purposes ...

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
```

## 🔧 Fine-tune
TODO

## 🔥 Pre-training
TODO

## ❄️ Downstream Evaluation

```python
import torch
```

## 📝 Citation

If you find Chronos models useful for your research, please consider citing the associated [paper]([https://arxiv.org/abs/2403.07815](https://TBD)):

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

## marker note

🛡️ 📃 :floppy_disk: 
