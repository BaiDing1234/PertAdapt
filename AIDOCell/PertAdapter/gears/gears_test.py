from copy import deepcopy
import argparse
from time import time
import sys, os
import pickle

import scanpy as sc
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import DataLoader
from .scfm_utils import *

from .model import *
from .model_new import GEARS_Model_Pert_Adapter_New_aido
from .inference import evaluate, compute_metrics, deeper_analysis, \
                  non_dropout_analysis, compute_synergy_loss
from .utils import loss_fct, loss_adapt, uncertainty_loss_fct, parse_any_pert, \
                  get_similarity_network, print_sys, GeneSimNetwork, \
                  create_cell_graph_dataset_for_prediction, get_mean_control, \
                  get_GI_genes_idx, get_GI_params

torch.manual_seed(0)

import warnings
warnings.filterwarnings("ignore")

class GEARSAdapt_test:
    def __init__(self, pert_data, 
                 train_bs = 48,
                 test_bs = 48,
                 device = 'cuda',
                 weight_bias_track = False, 
                 proj_name = 'GEARS', 
                 exp_name = 'GEARS',
                 pred_scalar = False,
                 gi_predict = False,
                 loss = 'loss_fct',
                 api_key = 'None'):
        
        self.weight_bias_track = weight_bias_track

        if self.weight_bias_track:
            import wandb
            if api_key != "None":
                wandb.login(key=api_key)
            wandb.init(project=proj_name, name=exp_name, entity = "biofm")
            self.wandb = wandb
        else:
            self.wandb = None
        
        self.device = device
        torch.cuda.set_device(self.device)
        self.config = None
        
        self.dataloader = pert_data.dataloader

        self.adata = pert_data.adata
        self.node_map = pert_data.node_map
        self.node_map_pert = pert_data.node_map_pert

        self.loss = loss


        self.data_path = pert_data.data_path
        self.dataset_name = pert_data.dataset_name
        self.split = pert_data.split
        self.seed = pert_data.seed
        self.train_gene_set_size = pert_data.train_gene_set_size
        self.set2conditions = pert_data.set2conditions
        self.subgroup = pert_data.subgroup
        self.gi_go = pert_data.gi_go
        self.gi_predict = gi_predict
        self.gene_list = pert_data.gene_names.values.tolist()
        self.pert_list = pert_data.pert_names.tolist()
        self.num_genes = len(self.gene_list)
        self.num_perts = len(self.pert_list)
        self.saved_pred = {}
        self.saved_logvar_sum = {}

        self.train_bs = train_bs
        
        self.ctrl_expression = torch.tensor(np.mean(self.adata.X[self.adata.obs.condition == 'ctrl'], axis = 0)).reshape(-1,).to(self.device)
        self.ctrl_expression_std = torch.tensor(np.std(self.adata.X[self.adata.obs.condition == 'ctrl'].toarray(), axis = 0)).reshape(-1,).to(self.device)
        pert_full_id2pert = dict(self.adata.obs[['condition_name', 'condition']].values)

        self.geneid2idx = dict(zip(self.adata.var.index.values, range(len(self.adata.var.index.values))))
        self.pert2pert_full_id = dict(self.adata.obs[['condition', 'condition_name']].values)

        if gi_predict:
            self.dict_filter = None
        else:
            self.dict_filter = {pert_full_id2pert[i]: j for i,j in 
                            self.adata.uns['non_zeros_gene_idx'].items() 
                            if i in pert_full_id2pert}
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
        
        gene_dict = {g:i for i,g in enumerate(self.gene_list)}
        self.pert2gene = {p:gene_dict[pert] for p, pert in enumerate(self.pert_list) if pert in self.gene_list} #map pert_idx to gene_idx
               
    def tunable_parameters(self):
        return {'hidden_size': 'hidden dimension, default 64',
                'num_go_gnn_layers': 'number of GNN layers for GO graph, default 1',
                'num_gene_gnn_layers': 'number of GNN layers for co-expression gene graph, default 1',
                'decoder_hidden_size': 'hidden dimension for gene-specific decoder, default 16',
                'num_similar_genes_go_graph': 'number of maximum similar K genes in the GO graph, default 20',
                'num_similar_genes_co_express_graph': 'number of maximum similar K genes in the co expression graph, default 20',
                'coexpress_threshold': 'pearson correlation threshold when constructing coexpression graph, default 0.4',
                'uncertainty': 'whether or not to turn on uncertainty mode, default False',
                'uncertainty_reg': 'regularization term to balance uncertainty loss and prediction loss, default 1',
                'direction_lambda': 'regularization term to balance direction loss and prediction loss, default 1'
               }
    
    def model_initialize(self, hidden_size = 64,
                         num_go_gnn_layers = 1, 
                         num_gene_gnn_layers = 1,
                         decoder_hidden_size = 16,
                         num_similar_genes_go_graph = 20,
                         num_similar_genes_co_express_graph = 20,                    
                         coexpress_threshold = 0.4,
                         uncertainty = False, 
                         uncertainty_reg = 1,
                         direction_lambda = 1e-1,
                         G_go = None,
                         G_go_weight = None,
                         G_coexpress = None,
                         G_coexpress_weight = None,
                         no_perturb = False, 
                         cell_fitness_pred = False,
                         go_path = None,
                         model_type=None,
                         bin_set=None,
                         load_path=None,
                         finetune_method=None,
                         accumulation_steps=1,
                         mode='v1',
                         highres=0,
                        **kwargs
                        ):
        
        self.config = {'hidden_size': hidden_size,
                       'num_go_gnn_layers' : num_go_gnn_layers, 
                       'num_gene_gnn_layers' : num_gene_gnn_layers,
                       'decoder_hidden_size' : decoder_hidden_size,
                       'num_similar_genes_go_graph' : num_similar_genes_go_graph,
                       'num_similar_genes_co_express_graph' : num_similar_genes_co_express_graph,
                       'coexpress_threshold': coexpress_threshold,
                       'uncertainty' : uncertainty, 
                       'uncertainty_reg' : uncertainty_reg,
                       'direction_lambda' : direction_lambda,
                       'G_go': G_go,
                       'G_go_weight': G_go_weight,
                       'G_coexpress': G_coexpress,
                       'G_coexpress_weight': G_coexpress_weight,
                       'device': self.device,
                       'num_genes': self.num_genes,
                       'num_perts': self.num_perts,
                       'no_perturb': no_perturb,
                       'cell_fitness_pred': cell_fitness_pred,
                       'pert2gene': self.pert2gene,
                        'model_type': model_type,
                        'bin_set': bin_set,
                        'load_path': load_path,
                        'finetune_method': finetune_method,
                        'accumulation_steps': accumulation_steps,
                        'mode':mode,
                        'highres':highres,
                      }
        print('Use accumulation steps:',accumulation_steps)
        print('Use mode:',mode)
        print('Use higres:',highres)

        self.config.update(kwargs)

        if self.wandb:
            self.wandb.config.update(self.config)
        
        if self.config['G_coexpress'] is None:
            ## calculating co expression similarity graph
            edge_list = get_similarity_network(network_type = 'co-express', adata = self.adata, threshold = coexpress_threshold, k = num_similar_genes_co_express_graph, gene_list = self.gene_list, data_path = self.data_path, data_name = self.dataset_name, split = self.split, seed = self.seed, train_gene_set_size = self.train_gene_set_size, set2conditions = self.set2conditions)
            sim_network = GeneSimNetwork(edge_list, self.gene_list, node_map = self.node_map)
            self.config['G_coexpress'] = sim_network.edge_index
            self.config['G_coexpress_weight'] = sim_network.edge_weight
        
        if self.config['G_go'] is None:
            print('No G_go')
            ## calculating gene ontology similarity graph
            edge_list = get_similarity_network(network_type = 'go', adata = self.adata, threshold = coexpress_threshold, k = num_similar_genes_go_graph, gene_list = self.pert_list, data_path = self.data_path, data_name = self.dataset_name, split = self.split, seed = self.seed, train_gene_set_size = self.train_gene_set_size, set2conditions = self.set2conditions, gi_go = self.gi_go, dataset = go_path)
            sim_network = GeneSimNetwork(edge_list, self.pert_list, node_map = self.node_map_pert)
            self.config['G_go'] = sim_network.edge_index
            self.config['G_go_weight'] = sim_network.edge_weight

        model_class = self.config.get('model_class', 'GEARS_Model_Pert_Adapter')
        
        self.config['train_bs'] = self.train_bs
        if model_class == 'GEARS_Model_aido':
            self.model = GEARS_Model_aido(self.config).to(self.device)
        elif model_class == 'GEARS_Model_Pert_Adapter_New_aido':
            self.model = GEARS_Model_Pert_Adapter_New_aido(self.config).to(self.device)

    def test(self, result_dir='./results/aido_pa_adj_weighted_de_loss-2025-05-17_17-40-07', 
             #below all dummy args
             epochs = 20, 
              lr = 1e-3,
              weight_decay = 5e-4,
              valid_every = 1,
              begin_valid_epoch = 1,
              print_every = 50,
              ddp_loss_weight = 1.
             ):
        

        if not self.config['no_perturb']:
            checkpoint = torch.load(os.path.join(result_dir, 'model.pt'), map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        
        n_parameters = sum(p.numel() for p in self.model.parameters())
        print('number of all params (M): %.3f' % (n_parameters / 1.e6))

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of trainable params (M): %.3f' % (n_parameters / 1.e6))


        if self.config.get('record_pred', False):
            self.model.args['start_record'] = True
            print_sys('Start recoding test results.')
            
        # Model testing
        test_loader = self.dataloader['test_loader']
        print_sys("Start Testing...")
        with torch.no_grad():
            test_res = evaluate(test_loader, self.model, self.config['uncertainty'], self.device)
        test_metrics, test_pert_res = compute_metrics(test_res)    
        print_sys(f"Best performing model: {test_metrics}")
        '''
        log = "Best performing model: Test Top 20 DE MSE: {:.4f}"
        print_sys(log.format(test_metrics['mse_de']))''' 
        
        if self.wandb:
            metrics = ['mse', 'pearson']
            for m in metrics:
                self.wandb.log({'test_' + m: test_metrics[m],
                        'test_de_'+m: test_metrics[m + '_de']                     
                        })
                
        out = deeper_analysis(self.adata, test_res)
        out_non_dropout = non_dropout_analysis(self.adata, test_res)
        
        metrics = ['pearson_delta']
        metrics_non_dropout = ['frac_opposite_direction_top20_non_dropout', 'frac_sigma_below_1_non_dropout', 'mse_top20_de_non_dropout']
        
        if self.wandb:
            for m in metrics:
                self.wandb.log({'test_' + m: np.mean([j[m] for i,j in out.items() if m in j])})

            for m in metrics_non_dropout:
                self.wandb.log({'test_' + m: np.mean([j[m] for i,j in out_non_dropout.items() if m in j])})        

        if self.split == 'simulation':
            print_sys("Start doing subgroup analysis for simulation split...")
            subgroup = self.subgroup
            subgroup_analysis = {}
            for name in subgroup['test_subgroup'].keys():
                subgroup_analysis[name] = {}
                for m in list(list(test_pert_res.values())[0].keys()):
                    subgroup_analysis[name][m] = []

            for name, pert_list in subgroup['test_subgroup'].items():
                for pert in pert_list:
                    for m, res in test_pert_res[pert].items():
                        subgroup_analysis[name][m].append(res)

            for name, result in subgroup_analysis.items():
                for m in result.keys():
                    subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
                    if self.wandb:
                        self.wandb.log({'test_' + name + '_' + m: subgroup_analysis[name][m]})

                    print_sys('test_' + name + '_' + m + ': ' + str(subgroup_analysis[name][m]))

            ## deeper analysis
            subgroup_analysis = {}
            for name in subgroup['test_subgroup'].keys():
                subgroup_analysis[name] = {}
            
            for name, pert_list in subgroup['test_subgroup'].items():
                for pert in pert_list:
                    for m in out[pert].keys():
                        if m in subgroup_analysis[name].keys():
                            subgroup_analysis[name][m].append(out[pert][m])
                        else:
                            subgroup_analysis[name][m] = [out[pert][m]]

                    for m in out_non_dropout[pert].keys():
                        if m in subgroup_analysis[name].keys():
                            subgroup_analysis[name][m].append(out_non_dropout[pert][m])
                        else:
                            subgroup_analysis[name][m] = [out_non_dropout[pert][m]]

            for name, result in subgroup_analysis.items():
                for m in result.keys():
                    subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
                    if self.wandb:
                        self.wandb.log({'test_' + name + '_' + m: subgroup_analysis[name][m]})

                    print_sys('test_' + name + '_' + m + ': ' + str(subgroup_analysis[name][m]))
        print_sys('Done!')