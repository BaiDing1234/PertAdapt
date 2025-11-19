import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
import numpy as np
from torch_geometric.nn import SGConv
from modelgenerator.tasks import Embed

from .utils import print_sys
from copy import deepcopy
import os
from torch.nn import MultiheadAttention
from scipy import sparse
import pickle


class MLP(torch.nn.Module):

    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers = layers + [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 1 else None,
                torch.nn.ReLU()
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        self.network = torch.nn.Sequential(*layers)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        return self.network(x)
    



class GEARS_Model_Pert_Adapter_New_aido(torch.nn.Module):
    """
    GEARS
    """

    def __init__(self, args):
        super(GEARS_Model_Pert_Adapter_New_aido, self).__init__()
        self.args = args

        if args.get('record_pred', False):
            self.pred_dir = f"/l/users/ding.bai/scFoundation/aido_pert/results_record/preds/{self.args['exp_name']}"
            if not os.path.exists(self.pred_dir):
                os.mkdir(self.pred_dir)
            self.pred_batch_idx = 0

        self.num_genes = args['num_genes']
        self.num_perts = args['num_perts']
        hidden_size = args['hidden_size']
        self.hidden_size = hidden_size
        self.uncertainty = args['uncertainty']
        self.num_layers = args['num_go_gnn_layers']
        self.indv_out_hidden_size = args['decoder_hidden_size']
        self.num_layers_gene_pos = args['num_gene_gnn_layers']
        self.no_perturb = args['no_perturb']
        self.cell_fitness_pred = args['cell_fitness_pred']
        self.pert_emb_lambda = 0.2
        
        # perturbation positional embedding added only to the perturbed genes
        self.pert_w = nn.Linear(1, hidden_size)
           
        # gene/globel perturbation embedding dictionary lookup            
        self.gene_emb = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.pert_emb = nn.Embedding(self.num_perts, hidden_size, max_norm=True)

        # transformation layer
        self.emb_trans = nn.ReLU()
        self.pert_base_trans = nn.ReLU()
        self.transform = nn.ReLU()
        self.emb_trans_v2 = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        self.pert_fuse = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        
        ### perturbation gene ontology GNN
        self.G_sim = args['G_go'].to(args['device'])
        self.G_sim_weight = args['G_go_weight'].to(args['device'])

        self.sim_layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            self.sim_layers.append(SGConv(hidden_size, hidden_size, 1))
        
        # decoder shared MLP
        self.recovery_w = MLP([hidden_size, hidden_size*2, hidden_size], last_layer_act='linear')
        
        # gene specific decoder
        self.indv_w1 = nn.Parameter(torch.rand(self.num_genes,
                                               hidden_size, 1))
        self.indv_b1 = nn.Parameter(torch.rand(self.num_genes, 1))
        self.act = nn.ReLU()
        nn.init.xavier_normal_(self.indv_w1)
        nn.init.xavier_normal_(self.indv_b1)
        
        # Cross gene MLP
        self.cross_gene_state = MLP([self.num_genes, hidden_size,
                                     hidden_size])
        # final gene specific decoder
        self.indv_w2 = nn.Parameter(torch.rand(1, self.num_genes,
                                           hidden_size+1))
        self.indv_b2 = nn.Parameter(torch.rand(1, self.num_genes))
        nn.init.xavier_normal_(self.indv_w2)
        nn.init.xavier_normal_(self.indv_b2)
        
        # batchnorms
        self.bn_emb = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base_trans = nn.BatchNorm1d(hidden_size)
        
        # uncertainty mode
        if self.uncertainty:
            self.uncertainty_w = MLP([hidden_size, hidden_size*2, hidden_size, 1], last_layer_act='linear')
        
        #if self.cell_fitness_pred:
        self.cell_fitness_mlp = MLP([self.num_genes, hidden_size*2, hidden_size, 1], last_layer_act='linear')
        
        if args['model_type'] == 'aido':
            class bf_model(torch.nn.Module):
                def __init__(self, model, device, in_features, out_features):
                    super().__init__()
                    aido_cell_config = args.get('aido_cell_config', 'aido_cell_100m')
                    
                    self.model = model.to(device).to(torch.bfloat16).eval()
                    self.to_out = nn.Linear(in_features, out_features, device=device).to(torch.bfloat16)
                    self.to_out.requires_grad_(True)

                def forward(self, x):
                    orig_dtype = x.dtype
                    x = x.to(torch.bfloat16)
                    x = x[:, :-1]
                    x_transformed = model.transform({'sequences': x})
                    out = self.to_out(self.model(x_transformed))
                    return out.to(orig_dtype)
            
            #args['aido_cell_config'] = 'aido_cell_3m', 'aido_cell_10m', 'aido_cell_100m'
            aido_cell_config = args.get('aido_cell_config', 'aido_cell_100m')
            model = Embed.from_config({"model.backbone": aido_cell_config, "model.batch_size": args['train_bs']}).eval()
            in_features = {'aido_cell_3m': 128,
                                   'aido_cell_10m': 256,
                                   'aido_cell_100m': 640}[aido_cell_config]
            self.singlecell_model = bf_model(model, device = args['device'], 
                                             in_features = in_features, out_features=hidden_size)
            self.pretrained = True
            print(f"Single cell model load success! model type: {aido_cell_config}")
        elif args['model_type'] == 'API':
            def API(x):
                return torch.rand(x.shape[0], x.shape[1]-1,hidden_size).to(x.device)
            self.singlecell_model = API
            self.pretrained = True
        else:
            self.pretrained = False
            print('No Single cell model load!')

        self.pert_adapter = PertAdapterNew(d_model = hidden_size, nhead = 8)
        # nhead = 8: scfoundation decoder number of heads. 
        # nhead = 12: scfoundation encoder number of heads. But 512 not dividable by 12.

    def forward(self, data):
        x, pert_idx = data.x, data.pert_idx
        if self.no_perturb:
            out = x.reshape(-1,1)
            out = torch.split(torch.flatten(out), self.num_genes)           
            return torch.stack(out)
        else:
            num_graphs = len(data.batch.unique())

            ## get base gene embeddings
            if self.pretrained:
                pre_in = x.clone().reshape(num_graphs, self.num_genes+1)
                x = x.reshape(num_graphs, self.num_genes+1)[:,:-1]
                emb = self.singlecell_model(pre_in)
                emb = emb.view(-1, self.hidden_size)
            else:
                x = x.reshape(num_graphs, self.num_genes+1)[:,:-1]
                emb = self.gene_emb(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))
                            
            emb = self.bn_emb(emb)

            ## get perturbation index and embeddings

            pert_index = []
            for idx, i in enumerate(pert_idx):
                for j in i:
                    if j != -1:
                        pert_index.append([idx, j])
            pert_index = torch.tensor(pert_index).T

            pert_global_emb = self.pert_emb(torch.LongTensor(list(range(self.num_perts))).to(self.args['device']))        

            ## augment global perturbation embedding with GNN
            for idx, layer in enumerate(self.sim_layers):
                pert_global_emb = layer(pert_global_emb, self.G_sim, self.G_sim_weight)
                if idx < self.num_layers - 1:
                    pert_global_emb = pert_global_emb.relu()

            ## add global perturbation embedding to each gene in each cell in the batch
            base_emb = torch.zeros((num_graphs, self.hidden_size)).to(self.args['device'])  

            if pert_index.shape[0] != 0:
                ### in case all samples in the batch are controls, then there is no indexing for pert_index.
                pert_track = {}
                for i, j in enumerate(pert_index[0]):
                    if j.item() in pert_track:
                        pert_track[j.item()] = pert_track[j.item()] + pert_global_emb[pert_index[1][i]]
                    else:
                        pert_track[j.item()] = pert_global_emb[pert_index[1][i]]

                if len(list(pert_track.values())) > 0:
                    if len(list(pert_track.values())) == 1:
                        # circumvent when batch size = 1 with single perturbation and cannot feed into MLP
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values()) * 2))
                    else:
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values())))

                    for idx, j in enumerate(pert_track.keys()):
                        base_emb[j] = base_emb[j] + emb_total[idx]

            base_emb = self.pert_adapter(emb.reshape((num_graphs, self.num_genes, -1)), 
                                         base_emb.reshape((num_graphs, 1, -1))) #B, N, D; B, 1, D -> B, N, D

            base_emb = base_emb.reshape(num_graphs * self.num_genes, -1)
            base_emb = self.bn_pert_base(base_emb)

            ## apply the first MLP (gene specific decoder)
            out = self.transform(base_emb)   

            out = self.recovery_w(out)

            out = out.reshape(num_graphs, self.num_genes, -1)
            out = out.unsqueeze(-1) * self.indv_w1
            w = torch.sum(out, axis = 2)
            out = w + self.indv_b1      

            out = out.reshape(num_graphs * self.num_genes, -1) + x.reshape(-1,1)

            ####################
            # Record test predictions #
            ####################

            if self.args.get('start_record', False):
                y_flat = data.y.reshape(-1, 1) #B * N, 1
                
                pred = torch.cat((out, y_flat), dim = 1) #B * N, 3
                pred = pred.cpu().to_sparse()

                # Convert the sparse tensor to a SciPy sparse matrix (COO format)
                values = pred.values().numpy()
                indices = pred.indices().numpy()
                sparse_matrix = sparse.coo_matrix((values, indices), shape=pred.shape)

                # Save the sparse matrix in NPZ format
                with open(f'{self.pred_dir}/b{self.pred_batch_idx}_{num_graphs}_pert.pkl', 'wb') as f:
                    pickle.dump(pert_idx, f)
                sparse.save_npz(f'{self.pred_dir}/b{self.pred_batch_idx}_{num_graphs}.npz', sparse_matrix)
                self.pred_batch_idx += 1

            out = torch.split(torch.flatten(out), self.num_genes)



            ## uncertainty head
            if self.uncertainty:
                out_logvar = self.uncertainty_w(base_emb)
                out_logvar = torch.split(torch.flatten(out_logvar), self.num_genes)
                return torch.stack(out), torch.stack(out_logvar)
            
            if self.cell_fitness_pred:
                return torch.stack(out), self.cell_fitness_mlp(torch.stack(out))
            
            return torch.stack(out)


class PertAdapterNew(nn.Module):
    #########
    # v3.3: input directly addition, only addition with the sum-pooled pert-encodings, and then weighted self-attn.
    #########
    def __init__(self, d_model, nhead, 
                 graph_npz_file = '/l/users/ding.bai/scFoundation/pert/data/go_mask_19264.npz'):
        super(PertAdapterNew, self).__init__()
        self.mha = MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm0 = torch.nn.LayerNorm(d_model)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.linear1 = Linear(d_model, d_model//2)
        self.linear2 = Linear(d_model//2, d_model)
        self.activation= F.relu

        self.Adjacency = sparse.load_npz(graph_npz_file)
        
        #min_weight = self.Adjacency.data.min()
        self.Adjacency = torch.from_numpy(self.Adjacency.toarray()).to(torch.float32).to('cuda') #N, N
        for i in range(self.Adjacency.shape[0]):
            if self.Adjacency[i, i] < 1:
                self.Adjacency[i, i] = 1
        self.Adjacency = torch.where(self.Adjacency > 0, torch.tensor(0.0), torch.tensor(float('-inf')))
        self.Adjacency.requires_grad = False
    
    def forward(self, exp_encodings, pert_encodings):
        x_in = exp_encodings + pert_encodings #B, N, D + #B, 1, D 
        x_in = self.norm0(x_in)
        x = self.mha(query = x_in, 
                     key = x_in, 
                     value = x_in,
                     attn_mask = self.Adjacency,
                    need_weights=False)[0]
        x = self.norm1(x + x_in)
        x = self.norm2(x + self.linear2(self.activation(self.linear1(x))))
        return x


