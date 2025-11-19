import os
import time
import argparse
import pandas as pd
import scanpy as sc
from os.path import join as pjoin
import torch

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from gears import PertData, GEARSAdapt_ddp

def main(parser):
    args = parser.parse_args()

    local_rank = args.local_rank
    rank = int(os.environ["RANK"])
    is_master = rank == 0

    dist.init_process_group(backend='nccl', init_method='env://')
    print('DDP init!')
    local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)
    world_size = torch.distributed.get_world_size()
    
    print("local_rank: ", local_rank)

    valid_every = args.valid_every
    # get data
    pert_data = PertData(args.data_dir)
    # load dataset in paper: norman, adamson, dixit.
    try:
        if args.data_name in ['norman', 'adamson', 'dixit']:
            pert_data.load(data_name = args.data_name)
        else:
            print('load data')
            pert_data.load(data_path = pjoin(args.data_dir, args.data_name))
    except:
        adata = sc.read_h5ad(pjoin(args.data_dir, args.data_name+'.h5ad'))
        adata.uns['log1p'] = {}
        adata.uns['log1p']['base'] = None
        pert_data.new_data_process(dataset_name=args.data_name, adata=adata)
        pert_data.load(data_path = pjoin(args.data_dir, args.data_name))
        
    # specify data split
    pert_data.prepare_split(split = args.split, seed = args.seed, train_gene_set_size=args.train_gene_set_size)
    # get dataloader with batch size
    pert_data.get_dataloader(batch_size = args.batch_size, test_batch_size = args.test_batch_size)

    # set up and train a model
    if is_master:
        print("EXPERIMENT: " + args.exp_name)
    gears_model = GEARSAdapt_ddp(pert_data, local_rank = local_rank,
                        is_master = is_master,
                        world_size=world_size,
                        device = device,
                        weight_bias_track = (args.api_key != 'None'),
                        train_bs=args.batch_size,
                        test_bs=args.test_batch_size,
                        proj_name = args.proj_name, 
                        exp_name = args.exp_name, 
                        api_key = args.api_key,
                        loss = args.loss)
    if is_master:
        print(f'len(gears_model.dict_filter): {len(gears_model.dict_filter)}')
    gears_model.model_initialize(hidden_size = args.hidden_size, 
                                 model_type = args.model_type,
                                 bin_set=args.bin_set,
                                 load_path=args.singlecell_model_path,
                                 finetune_method=args.finetune_method,
                                 accumulation_steps=args.accumulation_steps,
                                 mode=args.mode,
                                 highres=args.highres,
                                 add_ctrl = args.add_ctrl,
                                 model_class = args.model_class,
                                 use_to_final = args.use_to_final,
                                 tune_to_final = args.tune_to_final,
                                 attn_weight = args.attn_weight,
                                 de_loss_weight = args.de_loss_weight,
                                 record_pred=args.record_pred,
                                 exp_name = args.exp_name)
    if args.ckpt_to_continue != 'None':
        print(f'ckpt_to_continue loaded: {args.ckpt_to_continue}')
        checkpoint = torch.load(args.ckpt_to_continue, map_location=device)
        gears_model.model.load_state_dict(checkpoint)
    gears_model.train(epochs = args.epochs, 
                      result_dir=args.result_dir,lr=args.lr, 
                      valid_every = valid_every,
                      begin_valid_epoch=args.begin_valid_epoch,
                      ddp_loss_weight=args.ddp_loss_weight)

    # save model
    #gears_model.save_model(args.result_dir)

    # save params
    if is_master:
        param_pd = pd.DataFrame(vars(args), index=['params']).T
        param_pd.to_csv(f'{args.result_dir}/params.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GEARS')

    parser.add_argument("--local-rank", type=int, default=-1, help='Local process rank.')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--data_name', type=str, default='norman')
    parser.add_argument('--split', type=str, default='simulation')
    parser.add_argument('--result_dir', type=str, default='./results')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--valid_every', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--train_gene_set_size', type=float, default=0.75)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_type', type=str, default=None)
    parser.add_argument('--bin_set', type=str, default=None)
    parser.add_argument('--singlecell_model_path', type=str, default=None)
    parser.add_argument('--finetune_method', type=str, default=None)
    parser.add_argument('--mode', type=str, default='v1')
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--highres', type=int, default=0)

    parser.add_argument('--loss', type=str, default='loss_fct')
    parser.add_argument('-add_ctrl', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model_class', type=str, default='GEARS_Model_Pert_Adapter')
    
    parser.add_argument('--proj_name', type=str, default='GEARS')
    parser.add_argument('--exp_name', type=str, default='None')
    parser.add_argument('--api_key', type=str, default='None')

    parser.add_argument('-use_to_final', action='store_true')
    parser.add_argument('-tune_to_final', action='store_true')
    parser.add_argument('-record_pred', action='store_true')

    parser.add_argument('--attn_weight', type=float, default=0.)

    parser.add_argument('--ddp_loss_weight', type=float, default=1.)
    parser.add_argument('--begin_valid_epoch', type=int, default=1)

    parser.add_argument('--de_loss_weight', type=float, default=0.5)
    parser.add_argument('--ckpt_to_continue', type=str, default='None')

    


    main(parser)