import os
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..main_model import *
from .sentence_template import *

# Basic Module
def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True
    return model

class CheckShape(nn.Module):
    def __init__(self, remark, key=None):
        super().__init__()
        self.remark = remark
        self.key = key
    def forward(self, x, **kwargs):
        if self.remark is not None:
            print(self.remark, x.shape)
        
        out = x
        if self.key is not None:
            out = self.key(x)
        return out
    
class VAE_Latent(nn.Module):
    def __init__(self, emb_size, out_size):
        super().__init__()

        self.mu = nn.Linear(emb_size, out_size)
        self.var = nn.Sequential(
            nn.Linear(emb_size, out_size),
            nn.Softplus()
        )
        
    def forward(self, x, latent_only=True):
        # generate mean and variance
        mu, var = self.mu(x), self.var(x)

        # reparametrization trick
        if self.training:
            eps = torch.randn_like(var)
            z = mu + var*eps
        else:
            z = mu
        
        # output
        if latent_only:
            return z
        return z, mu, var
    
class FeedForward(nn.Module):
    def __init__(self, emb_size, hidden_size, dropout=0.1, add_norm=True, vae_out=True):
        super().__init__()
        self.add_norm = add_norm

        last_out = nn.Linear if not vae_out else VAE_Latent

        self.fc_liner = nn.Sequential(
            nn.Linear(emb_size, hidden_size),
            nn.GELU(),
            # nn.Dropout(p=dropout),
            # nn.Linear(hidden_size, emb_size),
            last_out(hidden_size, emb_size),
            nn.Dropout(p=dropout),
        )

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-6)

    def forward(self, x):
        out = self.fc_liner(self.LayerNorm(x))
        if self.add_norm:
            return x + out
        return out

# Aggregation Module
class MSiTFAggregation(nn.Module):
    def __init__(self, num_neurons=768, query_size=384, fuse_method='msitf', dropout=0.1, vae_out=True):
        super().__init__()
        self.fuse_method = fuse_method # mean, last, msitf

        if fuse_method == 'msitf':
            # self.v = nn.Linear(num_neurons, query_size)

            # importance, learn-to-drop
            self.cd = nn.Sequential(
                nn.Linear(num_neurons, 2),
                nn.Sigmoid()
            )
            self.t_bound = 5e-1

            # relevance
            self.k = nn.Linear(num_neurons, query_size)
            self.v = nn.Linear(num_neurons, query_size)
            self.q = nn.Linear(query_size, query_size)

            # recency
            self.decay_r = 0.997

            # output fc
            self.out_fc = FeedForward(query_size, query_size*4, dropout=dropout, vae_out=vae_out)

    def forward(self, x, query, mask=None, calc_recency=True, return_scores=False, device=torch.device('cpu'), rel_only=False, use_query=True): 
        # x: (N, nvar, L, E)
        # query: N, nvar, E
        if self.fuse_method == 'mean':
            return torch.mean(x, dim=1)
        elif self.fuse_method == 'last':
            return x[:, :, -1]

        # if not use query for aggregation (mean pool by default)
        if not use_query:
            return self.out_fc(self.v(x).mean(dim=1).mean(dim=1))
        
        # memory stream based fusion
        N, nvar, L, E = x.shape
        # recency
        if calc_recency:
            recency = torch.tensor([self.decay_r ** p for p in range(L)], requires_grad=False).float().to(device) # .bfloat16() or .float()
            recency = torch.stack([recency for _ in range(nvar)]).flatten()
            recency = torch.flip(recency, [0]).view(1, -1) # 1, L
        x = x.reshape(N, nvar*L, E)

        # relevance
        # query: N, E_q
        # query = query.unsqueeze(1) # N, 1, E_q
        q = self.q(query)
        k = self.k(x)
        v = self.v(x)
        # print(q.shape, k.shape)
        den = torch.norm(q, dim=-1)*torch.norm(k, dim=-1) # N, L
        attn = torch.sum(k*q, dim=-1) / den # N, L
        relevance = torch.softmax(attn, dim=-1)# N, L

        # importance
        if not rel_only:
            prob = self.cd(x) # N, L, 2
            log_prob = torch.log(prob)
            if self.training:
                eps = -torch.log(-torch.log(torch.rand(prob.shape))).float().to(device) # gumbel distribution
            else:
                eps = torch.tensor(0.577).float().to(device) # empirical mean of gumbel distribution
            log_prob = (log_prob + eps) / self.t_bound # N, L, 2
            prob_matrix = torch.exp(log_prob) / torch.exp(log_prob).sum(dim=-1, keepdim=True) # N, L, 2
            # mask = 0.5 < prob_matrix
            # prob_matrix = prob_matrix * mask # N, L, 2
            importance = prob_matrix[:, :, 1] # N, L

            # integrate scores, normalize, and weighted sum
            retrieval = relevance+(0.5*importance) # N, L
            if calc_recency:
                retrieval = retrieval + (0.2*recency)
        else:
            retrieval = relevance

        # softmax
        retrieval = torch.softmax(retrieval, dim=-1) # N, L
        # retrieval = retrieval / torch.sum(retrieval, dim=1, keepdim=True) # N, L
        final_out = torch.sum(v*retrieval.unsqueeze(-1), dim=1) # N, E_q

        final_out = self.out_fc(final_out)

        # return
        if return_scores:
            return v, recency, relevance, importance
        return final_out

# main module
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.hidden_states[-1] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def txt_encode(task=None, label=None, model=None, task_type=None):
    sentences = list()
    if len(task) == 1 and label is not None: # label sentence only (for gt encode only)
        # answers
        for l in label:
            if task_type == 'reg':
                sentences.append(np.random.choice(sentence_template[task[0]]['answer_template'], 1)[0].format(l))
            else:
                sentences.append(sentence_template[task[0]]['answer_template'][l])
    elif label is None and len(task) == 1: # query sentence only (for inference forward only)
        # questions
        for key in task:
            sentences.append(np.random.choice(sentence_template[key]['question_template'], 1)[0])
    
    # if above 2 case meet (for pretrain)
    if len(sentences) > 0: 
        # text encoding
        return model.txt_encode(sentences)
    
    # else, both task and label have things
    questions, answers = list(), list()
    for t_i in range(len(task)):
        questions.append(np.random.choice(sentence_template[task[t_i]]['question_template'], 1)[0])
        answers.append(np.random.choice(sentence_template[task[t_i]]['answer_template'], 1)[0].format(label[t_i]))
    
    # text encoding
    embeds = model.txt_encode(questions+answers) # N, 2048

    return embeds[:len(questions), :], embeds[len(questions):, :] # question, answer

class NormWearZeroShot(nn.Module):
    def __init__(
            self, 
            msitf_ckpt="",
            weight_path="",
            use_query=True, # True
            rel_only=False # False
        ):
        super().__init__()

        self.use_query = use_query
        self.rel_only = rel_only

        # text encoder
        self.tokenizer = AutoTokenizer.from_pretrained("muzammil-eds/tinyllama-2.5T-Clinical-v2")
        self.nlp_model = freeze_model(AutoModelForCausalLM.from_pretrained("muzammil-eds/tinyllama-2.5T-Clinical-v2"))
        self.query_size = 2048

        # sensor encoder
        self.sensor_model = freeze_model(NormWearModel(weight_path=weight_path))
        self.aggregator = MSiTFAggregation(num_neurons=768, query_size=self.query_size, vae_out=True)

        # load pretrained weight
        if len(msitf_ckpt) > 0:
            try:
                stat_dict = torch.load(msitf_ckpt, map_location=torch.device('cpu'))
            
                # # comment this out if the error on model save is fixed
                # stat_dict = torch.load(msitf_ckpt, map_location=torch.device('cpu'))['model']
                # current_state_dict_keys = self.aggregator.state_dict().keys()
                # filtered_state_dict = {k.replace("aggregator.", ""): v for k, v in stat_dict.items() if k.replace("aggregator.", "") in current_state_dict_keys}
                filtered_state_dict = stat_dict

                # load
                self.aggregator.load_state_dict(filtered_state_dict)
                print("MSiTF Checkpoint is successfully loaded!")
            except:
                print("Error occur during loading checkpoint, please check.")

        # loss
        loss_l1 = nn.L1Loss()
        # loss_l1 = nn.MSELoss()
        loss_cos = nn.CosineEmbeddingLoss()
        self.lambda_temp = nn.Parameter(torch.ones(1)*42, requires_grad=True)
        def ctr_loss(x, y):
            # x, y: bn, E
            dot_prod = torch.exp(torch.matmul(x, y.T)) ** (1/self.lambda_temp) # bn, bn
            loss = F.softmax(dot_prod, dim=1) # bn, bn
            loss = torch.log(torch.diagonal(loss) / loss.sum(dim=1)) # bn
            return -loss.mean()

        # aggregate loss
        self.loss_f = lambda x, y: torch.sum(torch.nan_to_num(torch.stack([
            2*loss_l1(x, y),
            loss_cos(x, y, torch.ones(len(y)).to(x.device)),
            # ctr_loss(x, y)
        ])))

    def txt_encode(self, sentences):
        inputs = self.tokenizer(sentences, padding=True, return_tensors="pt").to(self.nlp_model.device)
        out = self.nlp_model(**inputs, output_hidden_states=True)
        return mean_pooling(out, inputs.attention_mask) # N, 2048
    
    def signal_encode(self, x, query, sampling_rate=65):
        # x: [bn, nvar, L]
        device = x.device

        # sensor encoding
        sensor_out = self.sensor_model.get_embedding(x, sampling_rate=sampling_rate, device=device) # bn, nvar, P, E

        if query.shape[0] == 1: # if single question for all samples in input batch
            query = query.expand(sensor_out.shape[0], query.shape[1]) # (bn, 2048)
        
        # aggregate
        bn, nvar, P, E  = sensor_out.shape

        query = query.unsqueeze(1).expand(bn, nvar*P, query.shape[1]) # (bn, nvar, 2048)

        # per channel aggregate
        ch_aggregate_out = self.aggregator(
            sensor_out, 
            query,
            device=device,
            rel_only=self.rel_only,
            use_query=self.use_query
        ) # bn*nvar, E

        return ch_aggregate_out

    def inference(self, signal_embed, option_embed):
        # Manhattan distance
        distances = torch.abs(signal_embed[:, None, :] - option_embed[None, :, :]).sum(dim=-1) # bn, num_choice

        # # dot product
        # distances = 1 / torch.matmul(signal_embed, option_embed.T)  # bn, num_choice

        sims = distances
        sims = 1 - (sims / torch.sum(sims, dim=1, keepdim=True))
        sims = torch.nan_to_num(sims) + 1e-8 # bn, num_choice

        y_preds = nn.functional.softmax(sims.float(), dim=1) # bn, num_choice
        return y_preds

    def forward(self, x, task, label=None, sampling_rate=65):
        # x: [bn, nvar, L]
        # task, label: string for sentence template match
        # label: None if query only
        # x = torch.nan_to_num(x) # numerical stability
        
        # text forward
        txt_embed_res = txt_encode(task=task, label=label, model=self) # (bn, E), (bn, E)

        # unpack
        if type(txt_embed_res) == tuple:
            query, gt = txt_embed_res
        else:
            query = txt_embed_res

        sensor_embed = self.signal_encode(x, query, sampling_rate=sampling_rate)

        
        if label is None:
            return sensor_embed

        # calculate loss
        loss = self.loss_f(sensor_embed, gt)

        return loss
