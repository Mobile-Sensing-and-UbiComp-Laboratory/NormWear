from scipy import signal
import torchaudio.transforms as T

from .modules.normwear import *

def wt(ts, lf=0.1, hf=65):
    # in: L
    # out: FxL
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

class NormWearModel(nn.Module):
    def __init__(
            self, 
            optimized_cwt=False, 
            use_cwt=True,
            weight_path='modules/normwear_ckpt-15470.pth'
        ):
        super().__init__()

        # # weight path note
        # weight_path = '../data/results/model_mae_checkpoint-140.pth' # 37k
        # weight_path = '../data/results/NormWear_Large_checkpoint-10.pth' # 24w 
        # weight_path = '../data/results/job_rand_maskv3_checkpoint-15470.pth' # 1.5Tb
        
        self.optimized_cwt = optimized_cwt

        # construct encoder
        self.backbone = NormWear(img_size=(387,65), patch_size=(9,5),mask_scheme='random',mask_prob=0.8,use_cwt=use_cwt,nvar=4, comb_freq=False)

        # load pretrained checkpoint
        stat_dict = torch.load(weight_path, map_location=torch.device('cpu'))['model']

        self.backbone.load_state_dict(stat_dict)
        print("Model Checkpoint is successfully loaded!")

        self.sampling_rate = 65
    
    def get_embedding(self, sample_data, sampling_rate=65, device=torch.device('cpu')):
        # data: [bn, nvar, L]
        if torch.is_tensor(sample_data):
            sample_data = sample_data.numpy()
        sample_data = sample_data.astype(np.float32)

        # resample
        if sampling_rate != self.sampling_rate:
            if sampling_rate > 256:
                resampler = T.Resample(sampling_rate, self.sampling_rate)

                bn, nvar, L = sample_data.shape
                sample_data = resampler(sample_data.reshape(bn*nvar, L)) # [nvar*nvar, new_L]
                _, new_L = sample_data.shape
                sample_data = sample_data.reshape(bn, nvar, new_L) # [bn, nvar, new_L]

        if torch.is_tensor(sample_data):
            sample_data = sample_data.numpy()
        
        # calculate spectrogram
        spec = self.calc_cwt(sample_data, device=device).float() # [bn, nvar, 3, L, n_scales]
        # ====================================================================

        # forward
        # out, hiddens = self.backbone.get_signal_embedding(spec.to(device), hidden_out=True, device=device) # bn, nvar, P, E
        # hidden: list([bn, nvar, P, E])

        out = self.backbone.get_signal_embedding(spec.to(device), hidden_out=False, device=device) # bn, nvar, P, E

        # keep all representations
        signal_embeddings = out # bn, nvar, P, E

        # potential aggregation process
        # out = torch.mean(out[:, :, :, :], dim=2) # bn, nvar, E (take mean)
        # out = out[:, :, 0, :] # bn, nvar, E (take CLS)

        # signal_embeddings = torch.mean(out, dim=0) # average over sensor channels
        # signal_embeddings = out.flatten() # concat all sensor channels
    
        return signal_embeddings # raw: bn, nvar, P, E
    
    def calc_cwt(self, x, device=torch.device('cpu')):
        # x: [bn, nvar, L]
        # return: # bn, nvar, 3, L, n_scales
        bn, nvar, L = x.shape

        if self.optimized_cwt: # use the version implemented with pytorch
            if not torch.is_tensor(x):
                x = torch.from_numpy(x).to(device)
            # calculate cwt
            cwt_res = cwt_wrap(x.view(bn*nvar, L), 0.1, 64) # bn*nvar, 3, L, n_scales
            _, n_, new_L, n_scale = cwt_res.shape
            cwt_res = cwt_res.view(bn, nvar, n_, new_L, n_scale) # bn, nvar, 3, L, n_scales
        else: # vanilla CWT
            cwt_res = torch.stack([spec_cwt(sample) for sample in x]) # bn, nvar, 3, L, n_scales
        return cwt_res

    def forward(self, x): 
        pass

if __name__ == '__main__':
    device = torch.device('cpu')

    # init model
    model = NormWearModel(weight_path='NormWear/modules/model_ckpts/NormWear_Large_checkpoint-10.pth', optimized_cwt=True).to(device)

    # test I/O
    x = torch.rand(1, 8, 32).to(device) # 2 samples, 3 sensor, sequence length of 8
    y = model.get_embedding(x, sampling_rate=65, device=device) # bn, nvar, P, E

    # log
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)