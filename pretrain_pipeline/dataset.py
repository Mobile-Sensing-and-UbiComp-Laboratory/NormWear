import pickle
# import h5py
import os
from glob import glob
from torch.utils.data import Dataset
import random
from tqdm import tqdm
import torch
# from torch.utils.data import DataLoader, Subset
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
import torch.nn.functional as F
import json

class LinearProbDataset(Dataset):
    def __init__(self,fnames,task): 
        
        #self.dataset_path = os.path.join(data_dir,dataset_name,'samples')
        self.fnames = fnames
        self.max_L = 387 # fix length
        self.task = task

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        f_path = self.fnames[idx]

        with open(f_path, 'rb') as f:
            data_dict = pickle.load(f)

        x = torch.from_numpy(data_dict['cwt']).permute(0, 3, 1, 2)  # n_var, 3, L, 65
        if torch.isnan(x).any():
            return None  # Ignore the sample with NaN values

        n_var, n_channels, L, H = x.size()
        padded_inputs = torch.zeros((n_var, n_channels, self.max_L, H))
        
        if L < self.max_L:
            padded_inputs[:, :, -L:, :] = x
        else:
            padded_inputs = x[:, :, -self.max_L:, :]
        
        label = data_dict['label']

        if isinstance(label, dict):
            values = [label[key] for key in label.keys()]
            label = values
        
        if self.task == "reg":
            label = torch.tensor(label).float()
        else:
            label = torch.tensor(label).long()

       
        
        return {
            'input':padded_inputs,
            'label':label,
        }

def linprob_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch)

########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################

class PretrainDataset(Dataset):
    def __init__(self, data_dir, dataset_names, is_test=0):
        # record all file_names for getitem to read
        self.fnames = [
            os.path.join(data_dir, dataset_name, fn)
            for dataset_name in dataset_names
            for fn in sorted(os.listdir(os.path.join(data_dir, dataset_name)))
            if fn[0] != "."
        ]
        if is_test == 1:
            self.fnames = self.fnames[:1]
        self.sensors = ["PPG", "ECG", "GSR", "PCG", "EEG_F", "EEG_O", "EEG_L", "EEG_R", "ACC_X", "ACC_Y", "ACC_Z"]
        self.num_class = len(self.sensors)

        
    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        # Load Pickle
        with open(self.fnames[idx], 'rb') as f:
            data_dict = pickle.load(f)
    
        target = torch.from_numpy(data_dict['tss'])
        
        x = torch.from_numpy(data_dict['cwt']).permute(0, 3, 1, 2)  # n_var, 3, L, 65

        if torch.isnan(target).any() or torch.isnan(x).any():
            return None  # Ignore the sample with NaN values
        
        # x = torch.stack([self.input_transforms(img) for img in x])

        return {'target': target, 'input': x}

def collate_fn(batch, pad_nvar=4):
    # Filter out None values from the batch
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return {'target': torch.empty(0), 'input': torch.empty(0)}

    # Find the maximum sequence length L in the batch
    max_L = 387
    
    batch_size = len(batch)
    input_dim_1 = batch[0]['input'].size(1)
    input_dim_4 = batch[0]['input'].size(3)
    
    # Initialize padded tensors for inputs and targets
    padded_inputs = torch.zeros((batch_size, pad_nvar, input_dim_1, max_L, input_dim_4))
    padded_targets = torch.zeros((batch_size, pad_nvar, max_L + 1))

    for i, item in enumerate(batch):
        n_var = item['input'].size(0)
        L = item['input'].size(2)
        
        # If the number of variables is greater than pad_nvar, randomly sample pad_nvar variables
        if n_var > pad_nvar:
            indices = random.sample(range(n_var), pad_nvar)
            item['input'] = item['input'][indices]
            item['target'] = item['target'][indices]
            n_var = pad_nvar


        # Pad the inputs and targets
        padded_inputs[i, :n_var, :, :L, :] = item['input']
        padded_targets[i, :n_var, :L + 1] = item['target']

        # Copy last valid sequence to pad remaining variables up to pad_nvar
        for j in range(n_var, pad_nvar):
            padded_inputs[i, j] = padded_inputs[i, n_var - 1]
            padded_targets[i, j] = padded_targets[i, n_var - 1, :L + 1]
        

    return {'target': padded_targets, 'input': padded_inputs}

##############CWT Helper function########################################################################################################
def ricker_wavelet(points, scale):
    """Generate the Ricker (Mexican hat) wavelet for a given scale."""
    # a = scale
    # A = 2 / (torch.sqrt(3 * a) * torch.pi**0.25)  # Normalization factor
    # wavelet = A * (1 - (t / a)**2) * torch.exp(-0.5 * (t / a)**2)
    # return wavelet

    A = 2 / (torch.sqrt(3 * scale) * torch.pi**0.25)  # Normalization factor
    wsq = scale**2
    vec = torch.arange(0, points, device=scale.device) - (points - 1.0) / 2
    xsq = vec**2
    mod = (1 - xsq / wsq)
    gauss = torch.exp(-xsq / (2 * wsq))
    wavelet = A * mod * gauss
    return wavelet

def cwt_ricker(x, lowest_scale, largest_scale, step=1, wavelet_len=100):
    """
    Compute the CWT using the Ricker wavelet in PyTorch with simplified inputs.
    
    Args:
        x (torch.Tensor): Input time-series of shape (batch_size, sequence_length).
        lowest_scale (float): The lowest scale for the wavelet.
        largest_scale (float): The largest scale for the wavelet.
        step (float): Step size for generating scales.
        wavelet_len (int): Length of the wavelet.
        
    Returns:
        torch.Tensor: CWT scalogram of shape (batch_size, num_scales, sequence_length).
    """
    batch_size, sequence_length = x.shape
    x = x.unsqueeze(1)  # Add channel dimension, now (batch_size, 1, sequence_length)
    
    # Generate scales
    scales = torch.arange(lowest_scale, largest_scale + step, step, device=x.device)
    num_scales = scales.shape[0]
    
    # Prepare the wavelet basis for each scale
    # t = torch.linspace(-wavelet_len // 2, wavelet_len // 2, wavelet_len, device=x.device)
    wavelet_len = min(10*largest_scale, sequence_length)
    wavelets = torch.stack([ricker_wavelet(wavelet_len, scale) for scale in scales])
    # wavelets = torch.stack([ricker_wavelet(min(10*scale, sequence_length), scale) for scale in scales])
    wavelets = wavelets.view(num_scales, 1, -1)  # (num_scales, 1, wavelet_len)
    
    # Perform convolution for each scale
    cwt_output = F.conv1d(x, wavelets, padding=wavelet_len // 2)
    
    return cwt_output

def cwt_wrap(x, lowest_scale=0.1, largest_scale=64, step=1, wavelet_len=100):
    # x: bn, L
    # return: bn, 3, L, n_mels
    d1 = x[:, 1:] - x[:, :-1]  # bn, L-1
    d2 = d1[:, 1:] - d1[:, :-1] # bn, L-2
    x = torch.stack([x[:, 2:], d1[:, 1:], d2]).float().permute(1, 0, 2) # bn, 3, L-1
    bn, n_, new_L = x.shape
    cwt_res = cwt_ricker(x.reshape(bn*n_, new_L), lowest_scale, largest_scale, step=step, wavelet_len=wavelet_len) # bn*3, 65, new_L
    _, n_scale, new_L = cwt_res.shape
    return cwt_res.reshape(bn, n_, n_scale, new_L).permute(0, 1, 3, 2) # bn, 3, L, n_mels

#################################################################################
# Finetune Data loader
class dataset_class(Dataset):
    def __init__(self, data_dir,is_train=True,max_len=390,use_spec=False):
        super(dataset_class, self).__init__()
        
        self.data_dir = data_dir
        with open(os.path.join(self.data_dir,'train_test_split.json'), 'r') as file:
            split = json.load(file) # a dictionary with key 'train', 'test'
        
        if is_train:
            self.data_fn = split['train']
        else:
            self.data_fn = split['test']

        self.max_len = max_len
        self.use_spec = use_spec
    def __getitem__(self, idx):
        curr_fn = self.data_fn[idx]
        with open(os.path.join(self.data_dir,'sample_for_downstream',curr_fn),'rb') as file:
            data = pickle.load(file)

        tss = torch.from_numpy(data['data']).float()
        tss = torch.nan_to_num(tss,0.0)
        label = torch.tensor(data['label'][0]['class'])

        # Pad `tss` to `max_len` by repeating the first column
        if tss.shape[1] < self.max_len:  # Check if `L < max_len`
            pad_size = self.max_len - tss.shape[1]
            # Repeat the first column `pad_size` times
            pad = tss[:, 0:1].repeat(1, pad_size)  # Shape: [nvar, pad_size]
            # Concatenate padding at the beginning
            tss = torch.cat((pad, tss), dim=1)     # Final shape: [nvar, max_len]

        if self.use_spec:
            # calculate spectrogram
            cwt = cwt_wrap(tss) # nvar, 3, L, n_mels
            return {'input':cwt, 'label':label}

        else: return tss, label, 0 # make training script happy

    def __len__(self):
        return len(self.data_fn)
    
# example usage
if __name__ == '__main__':
    dataset_train = PretrainDataset(
        data_dir='../data/pretrain',
        dataset_names=[
            "dalia",
        ],
        is_test=0
    )
    print("Number of Samples:", len(dataset_train))

    # Create a DataLoader for the entire dataset
    dataset_loader = DataLoader(
        dataset_train,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda x: collate_fn(x, pad_nvar=4),
        pin_memory=True
    )

    for batch in tqdm(dataset_loader):
        pass