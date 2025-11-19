import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast
from einops import rearrange, repeat

from functools import partial
from contextlib import contextmanager


def exists(val):
    return val is not None

class AutoDiscretizationEmbedding2(nn.Module):
    def __init__(self, dim, max_seq_len, bin_num, bin_alpha, mask_token_id = None, pad_token_id = None):
        super().__init__()
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.bin_num = bin_num
        self.bin_alpha = bin_alpha
        
        self.mlp = nn.Linear(1, self.bin_num)
        self.mlp2 = nn.Linear(self.bin_num, self.bin_num)
        self.LeakyReLU = nn.LeakyReLU(0.1)
        self.Softmax = nn.Softmax(dim=-1)
        self.emb = nn.Embedding(self.bin_num, self.dim)
        
        self.emb_mask = nn.Embedding(1, self.dim)
        self.emb_pad = nn.Embedding(1, self.dim)
        
        self.bin_num_idx = torch.tensor(range(self.bin_num))
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id

        self.tensor0 = torch.tensor(0, dtype=torch.long)

    def forward(self, x, output_weight=0):
        x_mask_idx = (x==self.mask_token_id).nonzero()
        x_pad_idx = (x==self.pad_token_id).nonzero()
        
        x = self.mlp(x) # [B,N,1] -> [B,N,H]
        x = self.LeakyReLU(x) # [B,N,H]
        x_crosslayer = self.mlp2(x) # [B,N,H]
        x = self.bin_alpha * x + x_crosslayer # [B,N,H]
        weight = self.Softmax(x) # [B, N, H]
        
        bin_num_idx = self.bin_num_idx.to(x.device) # [H,]
        
        token_emb = self.emb(bin_num_idx) # [H, D]
        x = torch.matmul(weight, token_emb) #[B, N, D]

        tensor0 = torch.tensor(0, dtype=torch.long, device=x.device)

        mask_token_emb = self.emb_mask(tensor0).to(x.device).type(x.dtype)
        x[x_mask_idx[:,0],x_mask_idx[:,1],:] = mask_token_emb.repeat(x_mask_idx.shape[0],1)

        pad_token_emb = self.emb_pad(tensor0).to(x.device).type(x.dtype)
        x[x_pad_idx[:,0],x_pad_idx[:,1],:] = pad_token_emb.repeat(x_pad_idx.shape[0],1)
    
        if output_weight:
            return x,weight
        return x

class RandomPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        """
        data_labels True 表面使用了当前数据，False 表明未使用
        """
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)


class MaeAutobin(nn.Module):
    def __init__(
            self,
            *,
            num_tokens,  # num of tokens
            max_seq_len,  # max length of sequence
            embed_dim,  # encoder dim of tokens
            decoder_embed_dim,
            tie_embed=False,
            # False: output is num of tokens, True: output is dim of tokens  //multiply final embeddings with token weights for logits, like gpt decoder//
            bin_alpha = 1.0,
            bin_num = 10,
            pad_token_id = None,
            mask_token_id = None,
    ):
        super(MaeAutobin, self).__init__()

        self.max_seq_len = max_seq_len
        self.num_tokens = num_tokens
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id

        # encoder
        self.token_emb = AutoDiscretizationEmbedding2(embed_dim, max_seq_len, bin_num=bin_num, bin_alpha=bin_alpha, pad_token_id=self.pad_token_id, mask_token_id=self.mask_token_id)
        self.pos_emb = nn.Embedding(max_seq_len+1, embed_dim)  #RandomPositionalEmbedding(embed_dim, max_seq_len)

        # ## DEBUG
        self.encoder = None

        ##### decoder
        self.decoder = None
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.norm = nn.LayerNorm(decoder_embed_dim)
        self.to_final = nn.Linear(decoder_embed_dim, 1)

    def forward(self, x, padding_label, encoder_position_gene_ids, encoder_labels, decoder_data,
                mask_gene_name, mask_labels, decoder_position_gene_ids, decoder_data_padding_labels=None,
                output_attentions=False, encoder_perturb_label=None, decoder_perturb_label=None, perturb_emb=None):
        b, n, device = *x.shape, x.device
        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'
        #         print('00 input b,n,device',b,n,device,x.shape)

        # token and positional embedding
        x = self.token_emb(torch.unsqueeze(x, 2), output_weight = 0)
        if output_attentions:
            x.requires_grad_()  # used for attn_map output

        position_emb = self.pos_emb(encoder_position_gene_ids)

        x += position_emb

        if encoder_perturb_label is not None:
            x[encoder_perturb_label] += perturb_emb

        x = self.encoder(x, padding_mask=padding_label)

        decoder_data = self.token_emb(torch.unsqueeze(decoder_data, 2))
        position_emb = self.pos_emb(decoder_position_gene_ids)
        if mask_gene_name:
            print('mask_gene_name not done')
            exit(0)

        if decoder_perturb_label is not None:
            decoder_data[decoder_perturb_label] += perturb_emb

        batch_idx, gen_idx = (encoder_labels == True).nonzero(as_tuple=True)
        decoder_data[batch_idx, gen_idx] = x[~padding_label].to(decoder_data.dtype)

        decoder_data += position_emb

        decoder_data = self.decoder_embed(decoder_data)
        x = self.decoder(decoder_data, padding_mask=decoder_data_padding_labels)

        x = self.norm(x)
        if exists(self.to_final):
            x = self.to_final(x)
            return x.squeeze(2)  
        else:
            return x
        return x


############
# Below by Ding
############

from torch.nn import MultiheadAttention
from torch.nn import Linear


class PertAdapterCrossAttn(nn.Module):
    #########
    # v0 and v1: value as pert_encodings or exp_encodings
    #########
    def __init__(self, d_model, nhead):
        super(PertAdapterCrossAttn, self).__init__()
        self.mha = MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.linear1 = Linear(d_model, d_model//2)
        self.linear2 = Linear(d_model//2, d_model)
        self.activation= F.relu
    
    def forward(self, exp_encodings, pert_encodings):
        x = self.mha(query = exp_encodings, 
                     key = pert_encodings, 
                     #value = pert_encodings,
                     value = exp_encodings,
                    need_weights=False)[0]
        x = self.norm1(x + exp_encodings)
        x = self.norm2(x + self.linear2(self.activation(self.linear1(x))))
        return x


class PertAdapterConcat(nn.Module):
    #########
    # v2: input concat in dim 2.
    #########
    def __init__(self, d_model, nhead):
        super(PertAdapterConcat, self).__init__()
        self.mha = MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm0 = torch.nn.LayerNorm(d_model)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.linear0 = Linear(2 * d_model, d_model)
        self.linear1 = Linear(d_model, d_model//2)
        self.linear2 = Linear(d_model//2, d_model)
        self.activation= F.relu
    
    def forward(self, exp_encodings, pert_encodings):
        x_in = torch.cat((exp_encodings, pert_encodings), dim = 2)  #B, N, 2*D
        x_in = self.norm0(self.linear0(x_in)) #B, N, D
        x = self.mha(query = x_in, 
                     key = x_in, 
                     value = x_in,
                    need_weights=False)[0]
        x = self.norm1(x + x_in)
        x = self.norm2(x + self.linear2(self.activation(self.linear1(x))))
        return x



class PertAdapterAddition(nn.Module):
    #########
    # v3.0: input directly addition, and then self-attn.
    #########
    def __init__(self, d_model, nhead):
        super(PertAdapterAddition, self).__init__()
        self.mha = MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.linear1 = Linear(d_model, d_model//2)
        self.linear2 = Linear(d_model//2, d_model)
        self.activation= F.relu
    
    def forward(self, exp_encodings, pert_encodings):
        x_in = exp_encodings + pert_encodings
        x = self.mha(query = x_in, 
                     key = x_in, 
                     value = x_in,
                    need_weights=False)[0]
        x = self.norm1(x + x_in)
        x = self.norm2(x + self.linear2(self.activation(self.linear1(x))))
        return x


class PertAdapterAddition_v1(nn.Module):
    #########
    # v3.1: input directly addition, also addition with the sum-pooled pert-encodings, and then self-attn.
    #########
    def __init__(self, d_model, nhead):
        super(PertAdapterAddition_v1, self).__init__()
        self.mha = MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm0 = torch.nn.LayerNorm(d_model)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.linear1 = Linear(d_model, d_model//2)
        self.linear2 = Linear(d_model//2, d_model)
        self.activation= F.relu
    
    def forward(self, exp_encodings, pert_encodings):
        x_in = exp_encodings + pert_encodings + pert_encodings.sum(dim = 1, keepdim=True) #B, N, D + #B, N, D + #B, 1, D 
        x_in = self.norm0(x_in)
        x = self.mha(query = x_in, 
                     key = x_in, 
                     value = x_in,
                    need_weights=False)[0]
        x = self.norm1(x + x_in)
        x = self.norm2(x + self.linear2(self.activation(self.linear1(x))))
        return x

class PertAdapterAddition_v2(nn.Module):
    #########
    # v3.2: input directly addition, only addition with the sum-pooled pert-encodings, and then self-attn.
    #########
    def __init__(self, d_model, nhead):
        super(PertAdapterAddition_v2, self).__init__()
        self.mha = MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm0 = torch.nn.LayerNorm(d_model)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.linear1 = Linear(d_model, d_model//2)
        self.linear2 = Linear(d_model//2, d_model)
        self.activation= F.relu
    
    def forward(self, exp_encodings, pert_encodings):
        x_in = exp_encodings + pert_encodings.sum(dim = 1, keepdim=True) #B, N, D + #B, 1, D 
        x_in = self.norm0(x_in)
        x = self.mha(query = x_in, 
                     key = x_in, 
                     value = x_in,
                    need_weights=False)[0]
        x = self.norm1(x + x_in)
        x = self.norm2(x + self.linear2(self.activation(self.linear1(x))))
        return x


class PertAdapterAddition_v3(nn.Module):
    #########
    # v3.3: input directly addition, only addition with the sum-pooled pert-encodings, and then weighted self-attn.
    #########
    def __init__(self, d_model, nhead, attn_weight):
        super(PertAdapterAddition_v3, self).__init__()
        self.mha = MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm0 = torch.nn.LayerNorm(d_model)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.linear1 = Linear(d_model, d_model//2)
        self.linear2 = Linear(d_model//2, d_model)
        self.attn_weight = attn_weight
        self.activation= F.relu
    
    def forward(self, exp_encodings, pert_encodings):
        x_in = exp_encodings + pert_encodings #B, N, D + #B, 1, D 
        x_in = self.norm0(x_in)
        x = self.mha(query = x_in, 
                     key = x_in, 
                     value = x_in,
                    need_weights=False)[0]
        x = self.norm1(self.attn_weight * x + x_in)
        x = self.norm2(x + self.linear2(self.activation(self.linear1(x))))
        return x


class MaeAutobinPertAdapter(nn.Module):
    def __init__(
            self,
            *,
            num_tokens,  # num of tokens
            max_seq_len,  # max length of sequence
            embed_dim,  # encoder dim of tokens
            decoder_embed_dim,
            tie_embed=False,
            # False: output is num of tokens, True: output is dim of tokens  //multiply final embeddings with token weights for logits, like gpt decoder//
            bin_alpha = 1.0,
            bin_num = 10,
            pad_token_id = None,
            mask_token_id = None,
    ):
        super(MaeAutobinPertAdapter, self).__init__()

        self.max_seq_len = max_seq_len
        self.num_tokens = num_tokens
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id

        # encoder
        self.token_emb = AutoDiscretizationEmbedding2(embed_dim, max_seq_len, bin_num=bin_num, bin_alpha=bin_alpha, pad_token_id=self.pad_token_id, mask_token_id=self.mask_token_id)
        self.pos_emb = nn.Embedding(max_seq_len+1, embed_dim)  #RandomPositionalEmbedding(embed_dim, max_seq_len)

        # ## DEBUG
        self.encoder = None

        ##### decoder
        self.decoder = None
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        ##### PertAdapter
        self.pert_adapter = None

        self.norm = nn.LayerNorm(decoder_embed_dim)
        self.to_final = nn.Linear(decoder_embed_dim, 1)

    def forward(self, x, padding_label, encoder_position_gene_ids, encoder_labels, decoder_data,
                decoder_position_gene_ids, decoder_data_padding_labels=None,
                output_attentions=False, 
                perturbation_encodings=None):
        b, n, device = *x.shape, x.device
        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'
        #         print('00 input b,n,device',b,n,device,x.shape)

        # token and positional embedding
        x = self.token_emb(torch.unsqueeze(x, 2), output_weight = 0)
        if output_attentions:
            x.requires_grad_()  # used for attn_map output

        position_emb = self.pos_emb(encoder_position_gene_ids)
        x += position_emb
        x = self.encoder(x, padding_mask=padding_label)

        '''
        decoder_data = self.token_emb(torch.unsqueeze(decoder_data, 2))
        position_emb = self.pos_emb(decoder_position_gene_ids)

        batch_idx, gen_idx = (encoder_labels == True).nonzero(as_tuple=True)
        decoder_data[batch_idx, gen_idx] = x[~padding_label].to(decoder_data.dtype)

        decoder_data += position_emb

        decoder_data = self.decoder_embed(decoder_data)
        x = self.decoder(decoder_data, padding_mask=decoder_data_padding_labels)

        x = self.norm(x)'''

        ####Pertbation Adapter:
        x = self.pert_adapter(x, perturbation_encodings)

        if exists(self.to_final):
            x = self.to_final(x)
            return x.squeeze(2)  
        else:
            return x
        return x



from .transformer import pytorchTransformerModule
from .performer_module import PerformerModule

def select_module(config, sub_config, module_name):
    if module_name == 'performer':
        return PerformerModule(
            max_seq_len=config['seq_len'],
            dim=sub_config['hidden_dim'],
            depth=sub_config['depth'],
            heads=sub_config['heads'],
            dim_head=sub_config['dim_head'],
            ff_dropout=sub_config.get('ff_dropout',0.0),
            attn_dropout=sub_config.get('attn_dropout',0.0)
        )
    elif module_name == 'transformer':
        return pytorchTransformerModule(
            max_seq_len=config['seq_len'],
            dim=sub_config['hidden_dim'],
            depth=sub_config['depth'],
            heads=sub_config['heads']
        )
    else:
        print('module type error')
        exit(0)

def select_model(config, **kwargs):
    if config["model"] == "mae_autobin":
        encoder_config =config['encoder']
        decoder_config = config['decoder']
        encoder = select_module(config, encoder_config, config['encoder']['module_type'])
        decoder = select_module(config, decoder_config, config['decoder']['module_type'])
        pert_adapter = PertAdapterAddition_v3(d_model = config['decoder']['hidden_dim'],
                                            nhead = config['decoder']['heads'],
                                            **kwargs)
        model = MaeAutobinPertAdapter(
            num_tokens=config['n_class'],
            max_seq_len=config['seq_len'],
            embed_dim=config['encoder']['hidden_dim'],
            decoder_embed_dim=config['decoder']['hidden_dim'],
            bin_alpha = config['bin_alpha'],
            bin_num = config['bin_num'],
            pad_token_id = config['pad_token_id'],
            mask_token_id = config['mask_token_id'],
        )
        model.encoder = encoder
        model.decoder = decoder
        model.pert_adapter = pert_adapter
    else:
        raise NotImplementedError("Unknown model type!")
    return model

def get_sub_config(config, target):
    """
    获取 包含 target 的 config
    """
    sub_config = {}
    for k in config.keys():
        if target in k:
            tmp_name = k.replace(target + '_', '')
            sub_config[tmp_name] = config[k]
    return sub_config

def convertconfig(ckpt):
    newconfig = {}
    newconfig['config']={}
    model_type = ckpt['config']['model']
    
    for key, val in ckpt['config']['model_config'][model_type].items():
        newconfig['config'][key]=val
        
    for key, val in ckpt['config']['dataset_config']['rnaseq'].items():
        newconfig['config'][key]=val
        
    if model_type == 'performergau_resolution':
        model_type = 'performer_gau'
    
    import collections
    d = collections.OrderedDict()
    for key, val in ckpt['state_dict'].items():
        d[str(key).split('model.')[1]]=val
        
    newconfig['config']['model_type']=model_type
    newconfig['model_state_dict']=d
    newconfig['config']['pos_embed']=False
    newconfig['config']['device']='cuda'
    return newconfig

def load_model_frommmf_pert_adapter(best_ckpt_path, key='gene', device='cpu', tune_to_final = False, **kwargs):
    model_data = torch.load(best_ckpt_path,map_location=device)
    model_data = model_data[key]
    model_data = convertconfig(model_data)
    if not model_data.__contains__('config'):
        print('***** No config *****')
        config={}
        config['model_type']='flash_all'
    else:
        config=model_data['config']
        print(config)
    if not config.__contains__('qv_dim'):
        if config['model'] != 'mae_autobin':
            if config.__contains__('dim_head'):
                config['qv_dim']=config['dim_head']
            else:
                print('***** No qv_dim ***** set 64')
                config['qv_dim']= 64
    if not config.__contains__('ppi_edge'):
        config['ppi_edge']=None
    model = select_model(config, **kwargs)
    model_state_dict = model_data['model_state_dict']    
    model.load_state_dict(model_state_dict, strict=False)

    #print(f"tune_to_final: {tune_to_final}")
    for pname, param in model.named_parameters():
        if 'adapter' in pname:
            #print(f"finetune: {pname}")
            param.requires_grad = True
        elif tune_to_final and ('to_final' in pname):
            #print(f"finetune: {pname}")
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model.cuda(),config


def gatherData(data, labels, pad_token_id):
    value_nums = labels.sum(1)
    max_num = max(value_nums)


    fake_data = torch.full((data.shape[0], max_num), pad_token_id,
                           device=data.device)
    data = torch.hstack([data, fake_data])

    fake_label = torch.full((labels.shape[0], max_num), 1,
                            device=labels.device)
    none_labels = ~labels
    labels = labels.float()
    labels[none_labels] = torch.tensor(-float('Inf'), device=labels.device)

    tmp_data = torch.tensor([(i + 1) * 20000 for i in range(labels.shape[1], 0, -1)], device=labels.device)
    labels += tmp_data

    labels = torch.hstack([labels, fake_label])

    fake_label_gene_idx = labels.topk(max_num).indices

    new_data = torch.gather(data, 1, fake_label_gene_idx)

    padding_labels = (new_data == pad_token_id)

    return new_data, padding_labels



def getEncoerDecoderData(data, data_raw, config):
    decoder_data = data.clone().detach()
    decoder_data_padding = torch.full_like(data, False, dtype=torch.bool).to(data.device)

    encoder_data_labels = data_raw > 0
    encoder_data, encoder_data_padding = gatherData(decoder_data, encoder_data_labels,
                                                    config['pad_token_id'])
    new_data_raw = data_raw
    data_gene_ids = torch.arange(data.shape[1], device=data.device).repeat(data.shape[0], 1)
    encoder_position_gene_ids, _ = gatherData(data_gene_ids, encoder_data_labels,
                                                config['pad_token_id'])
    decoder_position_gene_ids = data_gene_ids
    data_mask_labels = None

    encoder_position_gene_ids[encoder_data_padding] = config["seq_len"]
    decoder_position_gene_ids[decoder_data_padding] = config["seq_len"]

    return encoder_data, encoder_position_gene_ids, encoder_data_padding, encoder_data_labels, decoder_data, decoder_data_padding, new_data_raw, data_mask_labels, decoder_position_gene_ids
