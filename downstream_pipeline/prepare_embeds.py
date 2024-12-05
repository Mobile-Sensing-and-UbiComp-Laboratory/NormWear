import os
import sys
import pickle
import pandas as pd

from .model_apis import *

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
print("DEVICE:", DEVICE)

# ============= helper functions ================================================
def load_model(model_name='ast'):
    # all models should follows the function structure of AST_API
    if model_name == 'ast':
        model = AST_API()
    elif model_name == 'clap':
        model = CLAP_API()
    elif model_name == 'opera':
        model = OPERA_API()
    elif model_name == 'normwear':
        model = NormWear_API()
    elif model_name == 'chronos':
        model = Chronos_API()
    elif model_name == 'healthmae':
        model = HealthMAE_API()
    # added baselines
    elif model_name == 'stats':
        model = STAT_API()
    elif model_name == 'tfc':
        model = TFC_API()
    elif model_name == 'demo':
        model = Demogr_API()
    elif model_name == 'crossvit':
        model = CrossVitAPI()
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

def audio_embedding_prepare(data_rootpath="audio_downstream/Coswara", model_name='ast', root_prefix="../", remark=""):
    # construct model
    model = load_model(model_name=model_name)

    save_remark = remark if len(remark) > 0 else model_name

    # initialize folder for save all the embedding
    os.makedirs('{}data/{}/{}_wav_embed'.format(root_prefix, data_rootpath, save_remark), exist_ok=True)

    # get embedding for each sample
    for fn in tqdm(sorted(os.listdir('{}data/{}/sample_for_downstream'.format(root_prefix, data_rootpath)))):
        # edge case
        if fn[0] == '.':
            continue
        
        # load sample
        # read data
        with open(os.path.join('{}data/{}/sample_for_downstream'.format(root_prefix, data_rootpath), fn), 'rb') as f:
            sample = pickle.load(f) # ['uid', 'data', 'label', 'sampling_rate']
        
        # expand 1 dimension if only single dimension
        if len(sample['data'].shape) == 1:
            sample['data'] = np.expand_dims(sample['data'], axis=0)

        # test clap pipeline
        with torch.no_grad():
            # if model_name in["demo", "normwear"]:
            if model_name in["demo"]: # normal demographic test
                embed = model.get_embedding(
                    sample['data'], 
                    sampling_rate=sample['sampling_rate'],
                    device=DEVICE,
                    sub_info=(data_rootpath, fn, root_prefix) # comment out if not demographic
                ) # E
            else:
                embed = model.get_embedding(
                    sample['data'], 
                    sampling_rate=sample['sampling_rate'],
                    device=DEVICE,
                ) # E

        # # check
        # print(embed.shape)
        # print(torch.mean(embed))
        # # print(sample['label'])
        # exit()
        
        # save the embedding
        with open(os.path.join('{}data/{}/{}_wav_embed'.format(root_prefix, data_rootpath, save_remark), fn), 'wb') as f:
            pickle.dump({
                "uid": sample["uid"], 
                "sampling_rate": sample['sampling_rate'], 
                "embed": embed.cpu().numpy().astype(np.float16), # E
                "label": sample['label']
            }, f)

def combine_normwear_ast(data_rootpath="audio_downstream/Coswara", root_prefix="../"):
    # initialize folder for save all the embedding
    os.makedirs('{}data/{}/nacombine_wav_embed'.format(root_prefix, data_rootpath), exist_ok=True)

    # get embedding for each sample
    for fn in tqdm(sorted(os.listdir('{}data/{}/sample_for_downstream'.format(root_prefix, data_rootpath)))):
        # edge case
        if fn[0] == '.':
            continue
        
        # load sample
        # read data
        with open(os.path.join('{}data/{}/sample_for_downstream'.format(root_prefix, data_rootpath), fn), 'rb') as f:
            sample = pickle.load(f) # ['uid', 'data', 'label', 'sampling_rate']
        
        # TODO combine embeds
        with open(os.path.join('{}data/{}/normwear_wav_embed'.format(root_prefix, data_rootpath), fn), 'rb') as f:
            normwear_embed = pickle.load(f)["embed"]
        with open(os.path.join('{}data/{}/ast_wav_embed'.format(root_prefix, data_rootpath), fn), 'rb') as f:
            ast_embed = pickle.load(f)["embed"]
        embed = np.concatenate((normwear_embed, ast_embed), axis=0)

        # # check
        # print(embed.shape)
        # print(np.mean(embed), np.std(embed), np.min(embed), np.max(embed))
        # exit()
        
        # save the embedding
        with open(os.path.join('{}data/{}/nacombine_wav_embed'.format(root_prefix, data_rootpath), fn), 'wb') as f:
            pickle.dump({
                "uid": sample["uid"], 
                "sampling_rate": sample['sampling_rate'], 
                "embed": embed, # E*2
                "label": sample['label']
            }, f)

if __name__ == '__main__':
    # python3 -m src.downstream.prepare_embeds chronos audio_downstream/KAUH
    # python3 -m src.downstream.prepare_embeds ast downstream/PPG_DM

    # input model name
    model_name = sys.argv[1] # ast, clap, opera, normwear
    data_rootpath = sys.argv[2] # e.g. audio_downstream/KAUH

    # process to get all embeds
    audio_embedding_prepare(model_name=model_name, data_rootpath=data_rootpath)

    # # combine embeds
    # # python3 -m src.downstream.prepare_embeds audio_downstream/KAUH
    # data_rootpath = sys.argv[1] # e.g. audio_downstream/KAUH
    # combine_normwear_ast(data_rootpath=data_rootpath)