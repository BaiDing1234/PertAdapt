from scRNA_workflow import *
import pandas as pd
from scipy import sparse as sp
from tqdm import tqdm
import gc
from copy import deepcopy
import scanpy as sc
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Preprocess')
parser.add_argument('--input_file', type=str)
parser.add_argument('--output_file', type=str)
args = parser.parse_args()

input_file = args.input_file
output_file = args.output_file

chunk_size = 100


gene_list_df = pd.read_csv('./OS_scRNA_gene_index.19264.tsv', header=0, delimiter='\t')
gene_list = list(gene_list_df['gene_name'])
new_gene_to_idx = {gene: i for i, gene in enumerate(gene_list)}

adata = sc.read_h5ad(input_file)

overlap_len = np.intersect1d(gene_list, adata.var['gene_name'].tolist()).shape[0]
print(f'Overlap genes: {overlap_len}')

old_genes = [] 
old_genes_idxes = []
for i, gene in enumerate(adata.var.gene_name.tolist()):
    if gene in new_gene_to_idx.keys():
        old_genes.append(gene)
        old_genes_idxes.append(i)

adata = adata[:, old_genes_idxes]
X = np.round(adata.X)
X = sp.csr_matrix(X)
gc.collect()

old_idx_to_new_idx = {j: new_gene_to_idx[gene] for j, gene in enumerate(old_genes)}

data = X.data
indices = X.indices
indptr = X.indptr

obs = deepcopy(adata.obs)
uns = deepcopy(adata.uns)
shape=(X.shape[0], len(gene_list))

adata = 0
gc.collect()


# Remap the column indices using old_idx_to_new_idx
new_indices = []
for j in tqdm(indices):
    new_indices.append(old_idx_to_new_idx[j])
new_indices = np.array(new_indices, dtype=np.int32)

# Construct new sparse matrix with the same data but remapped indices
new_X = sp.csr_matrix((data, new_indices, indptr), shape=shape)
new_X.sort_indices()


# Concatenate all processed chunks

adata_uni = sc.AnnData(X=new_X, dtype=np.float32)
adata_uni.var = pd.DataFrame(index=gene_list, data={'gene_name': gene_list})
adata_uni.obs = obs

adata_uni = BasicFilter(adata_uni,qc_min_genes=200,qc_min_cells=0) # filter cell and gene by lower limit
adata_uni = QC_Metrics_info(adata_uni)

total_count = adata_uni.X.sum(axis=1)
total_count = np.round(np.array(total_count, dtype=np.float32)).ravel()
print(total_count.shape)
adata_uni.obs['total_count'] = total_count
print(f"total_count max {np.max(total_count)} min {np.min(total_count)}")

sc.pp.normalize_total(adata_uni)
sc.pp.log1p(adata_uni)

save_adata_h5ad(adata_uni,output_file)
