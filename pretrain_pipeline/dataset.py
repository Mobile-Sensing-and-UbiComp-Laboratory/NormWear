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
# import torch.nn.functional as F

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