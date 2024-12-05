import os
import pickle
import torch
import numpy as np
import torch.nn as nn
from scipy import signal
from scipy.ndimage import gaussian_filter

import torchaudio.transforms as T

from transformers import ClapAudioModelWithProjection, ClapProcessor, AutoTokenizer, ClapTextModelWithProjection

import time

from scipy.stats import skew, kurtosis

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True
    return model

def extract_stat_features(ts, sampling_rate):
    # time domain
    feats = [ts.mean(), ts.std(), ts.max(), ts.min(), skew(ts), kurtosis(ts)]
    feats += [np.quantile(ts, 0.25), np.quantile(ts, 0.5), np.quantile(ts, 0.75)]

    # freq domain
    ts = torch.from_numpy(ts).float()
    # Perform the Fourier Transform
    spectrum = torch.fft.fft(ts)
    magnitude_spectrum = torch.abs(spectrum)
    
    # Only keep the positive frequencies (up to Nyquist frequency)
    magnitude_spectrum = magnitude_spectrum[:len(magnitude_spectrum) // 2]
    
    # Frequency bins
    freqs = torch.fft.fftfreq(len(ts), d=1/sampling_rate)[:len(magnitude_spectrum)]
    
    # Normalized magnitude spectrum for weighted calculations
    normalized_magnitude = magnitude_spectrum / magnitude_spectrum.sum()
    
    # Centroid (weighted mean frequency)
    centroid = torch.sum(freqs * normalized_magnitude)

    # Mean Frequency (unweighted mean)
    mean_freq = torch.mean(freqs)

    # Peak Frequency (frequency with the maximum magnitude)
    peak_freq = freqs[torch.argmax(magnitude_spectrum)]

    # Spread (standard deviation around the centroid)
    spread = torch.sqrt(torch.sum(((freqs - centroid) ** 2) * normalized_magnitude))

    # Quantiles (25%, median (50%), 75%)
    cumulative_magnitude = torch.cumsum(normalized_magnitude, dim=0)
    quantile_25 = freqs[min(torch.searchsorted(cumulative_magnitude, 0.25), len(freqs)-1)]
    median_freq = freqs[min(torch.searchsorted(cumulative_magnitude, 0.5), len(freqs)-1)]
    quantile_75 = freqs[min(torch.searchsorted(cumulative_magnitude, 0.75), len(freqs)-1)]

    freq_feats = [centroid, spread, mean_freq, peak_freq, quantile_25, median_freq, quantile_75]

    feats += [feat.item() for feat in freq_feats]

    return feats


#### More Baseline Models ################
class Demogr_API(nn.Module):
    def __init__(self):
        super().__init__()
        self.demo_size = -1
                
    def get_embedding(self, sample_data, sampling_rate=16000, device=torch.device('cpu'), sub_info=None):
        ds_name, fn, root_prefix = sub_info
        sub_id = fn.split("_")[0] 

        # edge filename case for indian-fPCG
        if sub_id == 'subject':
            sub_id = fn
        # sub_id = fn.split(".")[0] # if it is indian-fPCG

        # load demographic info
        with open("{}data/{}/demo.pkl".format(root_prefix, ds_name), 'rb') as f:
            demo_vec = pickle.load(f)[sub_id] # D
        return torch.from_numpy(demo_vec)
    
# from tsfresh import extract_features
# import pandas as pd
class STAT_API(nn.Module):
    def __init__(self):
        super().__init__()
                
    def get_embedding(self, sample_data, sampling_rate=16000, device=torch.device('cpu')):
        if torch.is_tensor(sample_data):
            sample_data = sample_data.numpy()
        
        # tokenization
        embeds = list()
        for i in range(len(sample_data)):
            features = extract_stat_features(sample_data[i], sampling_rate)

            # df = pd.DataFrame({"value": sample_data[i].astype(np.float32), "time": np.arange(len(sample_data[i])), "id": 1})

            # # Extract features
            # features = extract_features(df, column_id="id", column_sort="time", disable_progressbar=True).to_numpy().squeeze()

            # take log for numerical stability
            features = np.sign(features) * np.log1p(np.abs(features))
            embeds.append(np.nan_to_num(features))
        audio_embedding = torch.from_numpy(np.stack(embeds))
        # audio_embedding = torch.mean(audio_embedding, dim=0) # [783]
        audio_embedding = audio_embedding.flatten() # [16*nvar]

        # print(audio_embedding.shape, audio_embedding.mean(), audio_embedding.min(), audio_embedding.max())
        # exit()

        return audio_embedding

import torch.fft as fft
from .baseline_models.tfc.TFC.model import *
class TFC_API(nn.Module):
    def __init__(self, sampling_rate=65):
        super().__init__()
        
        self.sampling_rate = sampling_rate

        state_dict = torch.load("NormWear/modules/model_ckpts/ckp_last.pt")["model_state_dict"]

        # configs.TSlength_aligned = 178
        # for k in state_dict:
        #     # print(k)
        #     print(k, state_dict[k].shape)
        
        self.backbone = TFC(178)
        self.backbone.load_state_dict(state_dict)

        print("Load Success.")
        # exit()

                
    def get_embedding(self, sample_data, sampling_rate=16000, device=torch.device('cpu')):
        # sample_data: [nvar, L]
        # need x_in_t, x_in_f
        if not torch.is_tensor(sample_data):
            sample_data = torch.from_numpy(sample_data)

        x_in_t = sample_data.unsqueeze(1)[:, :1, -178:].to(device).float() # [nvar, 1, L]
        if x_in_t.shape[-1] < 178:
            x_in_t = torch.concat((torch.zeros(x_in_t.shape[0], x_in_t.shape[1], 178-x_in_t.shape[-1]).to(device), x_in_t), dim=-1)
        x_in_f = fft.fft(x_in_t).abs().float()
        out = self.backbone(x_in_t, x_in_f)

        # audio_embedding = torch.mean(out, dim=0) # [256]
        audio_embedding = out.flatten()

        # print(audio_embedding.shape, audio_embedding.mean(), audio_embedding.min(), audio_embedding.max())
        # exit()

        return audio_embedding

# ======= NEED SPECIAL DEPENDENCIES ========================================================
# from msclap import CLAP
# https://github.com/microsoft/CLAP
class CLAP_API(nn.Module):
    def __init__(self, sampling_rate=48000):
        super().__init__()
        self.backbone = freeze_model(ClapAudioModelWithProjection.from_pretrained("laion/clap-htsat-fused"))
        self.processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")

        self.nlp_model = freeze_model(ClapTextModelWithProjection.from_pretrained("laion/clap-htsat-fused"))
        self.tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-fused")

        self.sampling_rate = sampling_rate
    
    def txt_encode(self, sentences):
        inputs = self.tokenizer(text=sentences, return_tensors="pt", padding=True).to(self.nlp_model.device)
        return self.nlp_model(**inputs).text_embeds # [n, 512]
    
    def forward(self, x, task, label=None):
        # x: [updated] [bn, nvar, L]
        # task, label: string for sentence template match
        # label: None if query only
        device = x.device

        # resample
        bn, nvar, L = x.shape
        resampler = T.Resample(65, self.sampling_rate) # currently hardcode to 65hz
        # print("Resampling...")
        x = resampler(x.reshape(bn*nvar, L).cpu()).float()

        # calculate spectrogram
        # print("Calculate spec...")
        inputs = self.processor(audios=x.cpu().numpy(), return_tensors="pt", sampling_rate=self.sampling_rate)
        for k in inputs:
            inputs[k] = inputs[k].to(device)
        # print("Forward...")
        outputs = self.backbone(**inputs)
        audio_embeds = outputs.audio_embeds.reshape(bn, nvar, -1).mean(dim=1) # [bn*nvar, 512]

        return audio_embeds

    def get_embedding(self, sample_data, sampling_rate=16000, device=torch.device('cpu')):
        # data: [nvar, L]
        if sampling_rate != self.sampling_rate:
            if torch.is_tensor(sample_data):
                sample_data = sample_data.numpy()
            resampler = T.Resample(sampling_rate, self.sampling_rate)
            sample_data = resampler(torch.from_numpy(sample_data.astype(np.float32))).numpy()
        
        # tokenization
        inputs = self.processor(audios=sample_data, return_tensors="pt", sampling_rate=self.sampling_rate)
        for k in inputs:
            inputs[k] = inputs[k].to(device)

        outputs = self.backbone(**inputs)
        audio_embeds = outputs.audio_embeds # [nvar, 512]
        # audio_embedding = torch.mean(audio_embeds, dim=0) # E
        audio_embedding = audio_embeds.flatten()

        return audio_embedding

from chronos import ChronosPipeline
class Chronos_API(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-large",
            device_map="cuda",  # use "cpu" for CPU inference and "mps" for Apple Silicon
            torch_dtype=torch.bfloat16,
        )

    
    def get_embedding(self, sample_data, sampling_rate=16000, device=torch.device('cpu')):
        # data: [nvar, L]

        # forward
        embeddings, tokenizer_state = self.backbone.embed(torch.tensor(sample_data).float()) # [nvar, L, 768]
        # embeddings = torch.mean(embeddings, dim=1) # take the mean, [nvar, E]
        embeddings = embeddings[:, -1, :] # taken the representation at last state (because chronos is trained for forecasting, the last state is the key representation for the start of generation)
        # audio_embeddings = torch.mean(embeddings, dim=0) # E (768)
        audio_embeddings = embeddings.flatten() # E (768)
        return audio_embeddings.float()

    def forward(self, x): 
        pass

#==========================================================================================

# ===== BASELINE END ======================================================================

from ..modules.normwear import *

def wt(ts, lf=0.1, hf=65, wl='gaus1', method='fft'):
    # in: L
    # out: FxL
    # cwtmatr, freqs = pywt.cwt(ts, np.arange(lf, hf), wl, method=method)
    cwtmatr = signal.cwt(ts, signal.ricker, np.arange(lf, hf))
    return cwtmatr #[F, L]

def spec_cwt(audio_data): # [nvar, L]
    x1 = audio_data[:, 1:] - audio_data[:, :-1]
    x2 = x1[:, 1:] - x1[:, :-1]

    all_specs = list()
    for c_i in range(audio_data.shape[0]):
        all_specs.append(torch.stack([
            torch.from_numpy(wt(audio_data[c_i, 2:])).permute(1, 0), # [L, n_mels]
            torch.from_numpy(wt(x1[c_i, 1:])).permute(1, 0), 
            torch.from_numpy(wt(x2[c_i])).permute(1, 0)
        ])) # [3, L, n_mels]

    all_specs = torch.stack(all_specs) # [nvar, 3, L, n_mels]

    return all_specs

class NormWear_API(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = NormWear(img_size=(387,65), patch_size=(9,5),mask_scheme='random',mask_prob=0.8,use_cwt=True,nvar=4, comb_freq=False)

        # weight_path = '../data/results/NormWear_Huge_job_resume_checkpoint-1-correct.pth'
        # weight_path = '../data/results/model_mae_checkpoint-140.pth'
        # '../data/results/NormWear_Huge_job_resume_checkpoint-1-correct.pth' # 2.5m
        # ../data/results/NormWear_Huge_job_checkpoint-0.pth # train from scratch on 1.5Tb
        # weight_path = '../data/results/NormWear_Large_checkpoint-10.pth' # 24w (Currently reproducible the best)
        weight_path = 'data/results/job_rand_maskv3_checkpoint-15470.pth' # 1.5Tb (Currently reproducible the best)
        # weight_path = '../data/results/meanfusion_checkpoint-12000.pth'
        # weight_path = '../data/results/freqmask-scratch_checkpoint-13470.pth'
        # weight_path = '../data/results/timemask-scratch_checkpoint-13470.pth'
        # weight_path = '../data/results/job_rand_maskv3_checkpoint-0epoch-6000_correct.pth'
        # '../data/results/model_mae_checkpoint-140.pth' # 37k

        # load pretrained checkpoint
        local_weight_path = weight_path.replace("../", "")
        if os.path.isfile(local_weight_path):
            stat_dict = torch.load(local_weight_path, map_location=torch.device('cpu'))
        else:
            stat_dict = torch.load(weight_path, map_location=torch.device('cpu'))['model']

            # save weight in pod
            root_comp = local_weight_path.split("/")
            os.makedirs("/".join(root_comp[:-1]), exist_ok=True)
            torch.save(stat_dict, "/".join(root_comp))

        # stat_dict = torch.load('../data/results/model_mae_checkpoint-140.pth', map_location=torch.device('cpu'))['model']
        self.backbone.load_state_dict(stat_dict)
        print("Model load successfull.")

        self.sampling_rate = 65
    
    def get_embedding(self, sample_data, sampling_rate=16000, device=torch.device('cpu'), sub_info=None):
        # data: [nvar, L]
        if torch.is_tensor(sample_data):
            sample_data = sample_data.numpy()
        sample_data = torch.from_numpy(sample_data.astype(np.float32))

        # resample
        if sampling_rate != self.sampling_rate:
            if sampling_rate > 256:
                resampler = T.Resample(sampling_rate, self.sampling_rate)
                sample_data = resampler(sample_data) # [nvar, L]

        # calculate spectrogram
        spec = spec_cwt(sample_data.numpy()).float().unsqueeze(0).to(device) # [1, nvar, 3, L, n_mels]
        # ====================================================================

        # forward
        out, hiddens = self.backbone.get_signal_embedding(spec, hidden_out=True, device=device) # 1, nvar, P, E
        # hidden: list([1, nvar, P, E])

        out = torch.mean(out[0, :, :, :], dim=1) # nvar, E
        # out = out[0, :, 0, :] # nvar, E

        # # aggregate
        # out_prev = torch.mean(hiddens[-2][0, :, :, :], dim=1) # nvar, E
        # out = torch.concat((out, out_prev), dim=1) # nvar, E*2

        # audio_embeddings = torch.mean(out, dim=0)
        audio_embeddings = out.flatten()
        # audio_embeddings = torch.mean(out[0, :, 0, :], dim=0) # take the [CLS]: [512]

        # concat demo
        if sub_info is not None:
            ds_name, fn, root_prefix = sub_info
            sub_id = fn.split("_")[0] 

            # edge filename case for indian-fPCG
            if sub_id == 'subject':
                sub_id = fn
            # sub_id = fn.split(".")[0] # if it is indian-fPCG

            with open("{}data/{}/demo.pkl".format(root_prefix, ds_name), 'rb') as f:
                demo_vec = pickle.load(f)[sub_id] # D
            demo_vec = torch.from_numpy(demo_vec).to(device)
            audio_embeddings = torch.concat((demo_vec, audio_embeddings)).flatten()
    
        return audio_embeddings
    
    def encode(self, x, device=torch.device('cpu')):
        # x: [bn, nvar, L]
        # calculate cwt
        bn, nvar, L = x.shape
        cwt_res = cwt_wrap(x.view(bn*nvar, L), 0.1, 64) # bn*nvar, 3, L, n_mels
        _, n_, new_L, n_scale = cwt_res.shape
        cwt_res = cwt_res.view(bn, nvar, n_, new_L, n_scale) # bn, nvar, 3, L, n_mels
        
        # forward
        out = self.backbone.forward_all(cwt_res, hidden_out=False, device=device) # bn, nvar, P, E
        return out

    def forward(self, x): 
        pass

from .baseline_models.crossvit.crossvit import *
class CrossVitAPI(nn.Module):
    def __init__(self):
        super().__init__()

        self.sampling_rate = 65

        weight_path = '../data/results/model_checkpoint_cross_freeze_vit100_99.pth'

        self.encoder = CrossSignalViT(device='cuda')
        # load pretrained checkpoint
        local_weight_path = weight_path.replace("../", "")
        if os.path.isfile(local_weight_path):
            stat_dict = torch.load(local_weight_path, map_location=torch.device('cpu'))
        else:
            stat_dict = torch.load(weight_path, map_location=torch.device('cpu'))['model']

            # save weight in pod
            root_comp = local_weight_path.split("/")
            os.makedirs("/".join(root_comp[:-1]), exist_ok=True)
            torch.save(stat_dict, "/".join(root_comp))

        # load fetched stat_dict
        self.encoder.load_state_dict(stat_dict)
        self.encoder = self.encoder.bfloat16()
    
    def get_embedding(self, sample_data, sampling_rate=16000, device=torch.device('cpu')):
        # data: [nvar, L]
        if torch.is_tensor(sample_data):
            sample_data = sample_data.numpy()
        sample_data = torch.from_numpy(sample_data.astype(np.float32))

        # resample
        if sampling_rate != self.sampling_rate:
            if sampling_rate > 256:
                resampler = T.Resample(sampling_rate, self.sampling_rate)
                sample_data = resampler(sample_data) # [nvar, L]

        # calculate spectrogram
        spec = spec_cwt(sample_data.numpy()).float().unsqueeze(0).to(device) # [1, nvar, 3, L, n_mels]
        # ====================================================================

        # forward
        # x: [1, nvar, 3, L, n_mels]
        # N, C, L, 65, 3
        out = self.encoder.forward_all(spec.bfloat16()) # 1, C, L, 768
        out = torch.mean(out, dim=2).float() # 1, C, 768

        return out.flatten() # 768
        
    def forward(self, x): 
        pass
