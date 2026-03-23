#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate perturbation predictions with:
1) GSEA (Reactome)
2) GSVA (Reactome)
3) PROGENy pathway activity inference

Input assumptions
-----------------
1. adata: .h5ad file
   - obs[condition_key] contains:
       "GeneA+GeneB"     for double perturbation
       "ctrl+GeneB"      for single perturbation
       "ctrl"            for unperturbed
   - var_names are gene symbols
   - total genes = 19264 (or any number, as long as predictions match adata.n_vars)

2. predictions: one pickle file containing:
   {
       "model_name_1": {
           (i,): np.ndarray shape (n_genes,),
           (i, j): np.ndarray shape (n_genes,),
           ...
       },
       "model_name_2": {...},
       ...
   }
   where tuple entries are gene indices in adata.var_names order.

Outputs
-------
- metrics_summary.csv
- detailed per-perturbation csv files for GSEA / GSVA / PROGENy
"""

import argparse
import os
import io
import zipfile
import pickle
import warnings
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jaccard

import anndata as ad
import gseapy as gp
import decoupler as dc

#python GSEA_pathway_analysis.py --download_reactome
#python GSEA_pathway_analysis.py --reactome_gmt


CTRL_TOKENS = {"ctrl", "control", "unperturbed", "untreated", ""}


def parse_args():
    parser = argparse.ArgumentParser(description="Pathway-level evaluation for perturbation predictions.")
    parser.add_argument("--adata", type=str, 
                        default = '/root/autodl-tmp/data/gse90063_k562_ko_tf20_37160_19264_withtotalcount/perturb_processed.h5ad',
                        help="Path to input .h5ad")
    parser.add_argument("--pred_pickle", type=str, 
                        default='pkl/dixit_method_pert_pred_dicts_dict.pkl',
                        help="Path to pickle containing {model_name: {pert_tuple: pred_vector}}")
    parser.add_argument("--outdir", type=str, 
                        default='csv',
                        help="Output directory")

    parser.add_argument("--condition_key", type=str, default="condition",
                        help="obs key containing perturbation strings")
    parser.add_argument("--layer", type=str, default=None,
                        help="Use adata.layers[layer] instead of adata.X")
    parser.add_argument("--profile_mode", type=str, default="both",
                        choices=["direct", "differential", "both"],
                        help="Evaluate direct pseudobulk, differential-to-control, or both")

    parser.add_argument("--reactome_gmt", type=str, default='/root/PertAdapt/scFoundation/PertAdapter/result_process/csv/ReactomePathways.gmt',
                        help="Optional local Reactome GMT path. If absent, script can auto-download.")
    parser.add_argument("--download_reactome", action="store_true",
                        help="Automatically download official Reactome GMT if --reactome_gmt is not provided")
    parser.add_argument("--reactome_url", type=str,
                        default="https://reactome.org/download/current/ReactomePathways.gmt.zip",
                        help="Official Reactome GMT zip URL")

    parser.add_argument("--min_geneset_size", type=int, default=10)
    parser.add_argument("--max_geneset_size", type=int, default=5000)

    parser.add_argument("--gsea_permutations", type=int, default=100)
    parser.add_argument("--gsea_topk", type=int, default=20)
    parser.add_argument("--gsea_processes", type=int, default=1)

    parser.add_argument("--progeny_top", type=int, default=500,
                        help="Top responsive genes per pathway for PROGENy")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def dense_mean(X):
    if sparse.issparse(X):
        return np.asarray(X.mean(axis=0)).ravel()
    return np.asarray(X).mean(axis=0).ravel()


def get_matrix(adata, layer=None):
    if layer is None:
        return adata.X
    return adata.layers[layer]


def canonicalize_pred_key(key):
    if isinstance(key, tuple):
        vals = tuple(sorted(int(x) for x in key))
    elif isinstance(key, list):
        vals = tuple(sorted(int(x) for x in key))
    elif isinstance(key, np.ndarray):
        vals = tuple(sorted(int(x) for x in key.tolist()))
    elif isinstance(key, int):
        vals = (int(key),)
    else:
        raise ValueError(f"Unsupported prediction key type: {type(key)}")
    return vals


def parse_condition_to_tuple(cond, gene_to_idx):
    cond = str(cond).strip()
    if cond.lower() in CTRL_TOKENS:
        return tuple()

    tokens = [x.strip() for x in cond.split("+")]
    genes = []
    for tok in tokens:
        if tok.lower() in CTRL_TOKENS:
            continue
        if tok not in gene_to_idx:
            return None
        genes.append(gene_to_idx[tok])

    return tuple(sorted(set(genes)))


def tuple_to_name(pert, idx_to_gene):
    if len(pert) == 0:
        return "ctrl"
    return "+".join(idx_to_gene[i] for i in pert)


def load_predictions(pred_pickle, n_genes):
    with open(pred_pickle, "rb") as f:
        raw = pickle.load(f)

    if not isinstance(raw, dict):
        raise ValueError("Prediction pickle must contain a dict at top level.")

    out = {}
    for model_name, inner in raw.items():
        if not isinstance(inner, dict):
            raise ValueError(f"Predictions for model {model_name} must be a dict.")
        tmp = {}
        for k, v in inner.items():
            pert = canonicalize_pred_key(k)
            arr = np.asarray(v).reshape(-1)
            if arr.shape[0] != n_genes:
                raise ValueError(
                    f"Prediction vector length mismatch for model={model_name}, key={k}: "
                    f"got {arr.shape[0]}, expected {n_genes}"
                )
            tmp[pert] = arr
        out[model_name] = tmp
    return out


def compute_true_pseudobulks(adata, condition_key, layer=None):
    if condition_key not in adata.obs.columns:
        raise KeyError(f"{condition_key} not found in adata.obs")

    gene_to_idx = {g: i for i, g in enumerate(adata.var_names.tolist())}
    idx_to_gene = {i: g for i, g in enumerate(adata.var_names.tolist())}

    pert_tuples = []
    bad_conditions = 0
    for cond in adata.obs[condition_key].tolist():
        pert = parse_condition_to_tuple(cond, gene_to_idx)
        if pert is None:
            bad_conditions += 1
        pert_tuples.append(pert)

    if bad_conditions > 0:
        warnings.warn(
            f"{bad_conditions} cells have conditions containing genes not found in adata.var_names; skipping them."
        )

    X = get_matrix(adata, layer=layer)

    groups = {}
    for i, pert in enumerate(pert_tuples):
        if pert is None:
            continue
        groups.setdefault(pert, []).append(i)

    pseudobulks = {}
    n_cells = {}
    for pert, cell_idx in groups.items():
        prof = dense_mean(X[cell_idx])
        pseudobulks[pert] = prof
        n_cells[pert] = len(cell_idx)

    if tuple() not in pseudobulks:
        raise ValueError("No control cells found. Expected condition='ctrl' or equivalent.")

    return pseudobulks, n_cells, gene_to_idx, idx_to_gene


def maybe_download_reactome_gmt(outdir, reactome_gmt, download_reactome, reactome_url):
    if reactome_gmt is not None:
        reactome_gmt = Path(reactome_gmt)
        if not reactome_gmt.exists():
            raise FileNotFoundError(f"Reactome GMT not found: {reactome_gmt}")
        return reactome_gmt

    if not download_reactome:
        raise ValueError(
            "Provide --reactome_gmt or use --download_reactome to fetch the official Reactome GMT."
        )

    outdir = Path(outdir)
    ensure_dir(outdir)
    zip_path = outdir / "ReactomePathways.gmt.zip"
    gmt_path = outdir / "ReactomePathways.gmt"

    if not zip_path.exists():
        print(f"[INFO] Downloading Reactome GMT zip to {zip_path}")
        urllib.request.urlretrieve(reactome_url, zip_path)

    if not gmt_path.exists():
        print(f"[INFO] Extracting Reactome GMT to {gmt_path}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            gmt_members = [m for m in zf.namelist() if m.endswith(".gmt")]
            if len(gmt_members) == 0:
                raise RuntimeError("No .gmt file found inside downloaded Reactome zip.")
            member = gmt_members[0]
            with zf.open(member) as fin, open(gmt_path, "wb") as fout:
                fout.write(fin.read())

    return gmt_path


def parse_gmt(gmt_path):
    gene_sets = {}
    with open(gmt_path, "r") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            term = parts[0]
            genes = [x for x in parts[2:] if x != ""]
            if len(genes) > 0:
                gene_sets[term] = genes
    return gene_sets


def filter_gene_sets(gene_sets, valid_genes, min_size=10, max_size=5000):
    valid_genes = set(valid_genes)
    filtered = {}
    for term, genes in gene_sets.items():
        keep = [g for g in genes if g in valid_genes]
        if min_size <= len(keep) <= max_size:
            filtered[term] = keep
    return filtered


def gene_sets_to_long_df(gene_sets):
    rows = []
    for source, targets in gene_sets.items():
        for t in targets:
            rows.append((source, t))
    return pd.DataFrame(rows, columns=["source", "target"])


def build_profile_tables(true_pseudobulks, pred_pseudobulks, control_profile, idx_to_gene, mode):
    shared = sorted(set(true_pseudobulks.keys()) & set(pred_pseudobulks.keys()))
    shared = [p for p in shared if len(p) > 0]  # skip ctrl

    if len(shared) == 0:
        raise ValueError("No shared perturbations found between adata and predictions.")

    genes = [idx_to_gene[i] for i in range(len(idx_to_gene))]
    true_rows = []
    pred_rows = []
    row_names = []

    for pert in shared:
        t = np.asarray(true_pseudobulks[pert]).copy()
        p = np.asarray(pred_pseudobulks[pert]).copy()

        if mode == "differential":
            t = t - control_profile
            p = p - control_profile
        elif mode == "direct":
            pass
        else:
            raise ValueError(f"Unknown mode: {mode}")

        true_rows.append(t)
        pred_rows.append(p)
        row_names.append(tuple_to_name(pert, idx_to_gene))

    true_df = pd.DataFrame(np.vstack(true_rows), index=row_names, columns=genes)
    pred_df = pd.DataFrame(np.vstack(pred_rows), index=row_names, columns=genes)
    return true_df, pred_df


def safe_pearson(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < 2:
        return np.nan
    if np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
        return np.nan
    return pearsonr(x, y)[0]


def safe_spearman(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < 2:
        return np.nan
    return spearmanr(x, y).correlation


def rmse(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return float(np.sqrt(np.mean((x - y) ** 2)))


def topk_jaccard_from_scores(a, b, k=20):
    a = pd.Series(a)
    b = pd.Series(b)
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return np.nan
    a = a.loc[common]
    b = b.loc[common]
    k = min(k, len(common))
    a_top = set(a.abs().sort_values(ascending=False).head(k).index.tolist())
    b_top = set(b.abs().sort_values(ascending=False).head(k).index.tolist())
    if len(a_top | b_top) == 0:
        return np.nan
    return len(a_top & b_top) / len(a_top | b_top)


def run_single_prerank(profile, gene_sets, permutation_num, processes, seed,
                       min_size, max_size):
    # GSEApy prerank expects a ranked list
    rnk = pd.Series(profile.values, index=profile.index, name="score").dropna()
    rnk = rnk[~rnk.index.duplicated(keep="first")]
    rnk = rnk.sort_values(ascending=False)

    pre_res = gp.prerank(
        rnk=rnk,
        gene_sets=gene_sets,
        processes=processes,
        permutation_num=permutation_num,
        min_size=min_size,
        max_size=max_size,
        seed=seed,
        outdir=None,
        no_plot=True,
        verbose=False,
    )

    res = pre_res.res2d.copy()
    res.columns = [str(c).strip() for c in res.columns]
    if "Term" in res.columns:
        res = res.set_index("Term")
    elif "Name" in res.columns:
        res = res.set_index("Name")

    nes_col = None
    for c in res.columns:
        if str(c).strip().upper() == "NES":
            nes_col = c
            break
    if nes_col is None:
        raise RuntimeError(f"Could not find NES column in GSEA result. Columns={list(res.columns)}")

    return res[nes_col].astype(float)


def evaluate_gsea(true_df, pred_df, gene_sets, args, model_name, mode, outdir):
    rows = []
    for pert in true_df.index:
        true_nes = run_single_prerank(
            true_df.loc[pert],
            gene_sets=gene_sets,
            permutation_num=args.gsea_permutations,
            processes=args.gsea_processes,
            seed=args.seed,
            min_size=args.min_geneset_size,
            max_size=args.max_geneset_size,
        )
        pred_nes = run_single_prerank(
            pred_df.loc[pert],
            gene_sets=gene_sets,
            permutation_num=args.gsea_permutations,
            processes=args.gsea_processes,
            seed=args.seed,
            min_size=args.min_geneset_size,
            max_size=args.max_geneset_size,
        )
        common = true_nes.index.intersection(pred_nes.index)
        if len(common) == 0:
            continue

        t = true_nes.loc[common]
        p = pred_nes.loc[common]

        rows.append({
            "perturbation": pert,
            "gsea_nes_pearson": safe_pearson(t.values, p.values),
            "gsea_nes_spearman": safe_spearman(t.values, p.values),
            "gsea_topk_jaccard": topk_jaccard_from_scores(t, p, k=args.gsea_topk),
            "n_pathways": len(common),
        })

    detail = pd.DataFrame(rows)
    detail.to_csv(Path(outdir) / f"gsea_detail__{model_name}__{mode}.csv", index=False)

    summary = {
        "model": model_name,
        "profile_mode": mode,
        "method": "GSEA",
        "metric_gsea_nes_pearson": detail["gsea_nes_pearson"].mean(),
        "metric_gsea_nes_spearman": detail["gsea_nes_spearman"].mean(),
        "metric_gsea_topk_jaccard": detail["gsea_topk_jaccard"].mean(),
        "n_perturbations": len(detail),
    }
    return summary, detail


def normalize_decoupler_output(res, sample_names):
    # New API may return DataFrame, tuple(DataFrame, pvals), or other objects.
    if isinstance(res, pd.DataFrame):
        lower_cols = {str(c).lower(): c for c in res.columns}
        # long format
        if {"source", "condition", "score"}.issubset(lower_cols.keys()):
            tmp = res.copy()
            if "statistic" in lower_cols:
                stat_col = lower_cols["statistic"]
                stats = tmp[stat_col].astype(str).unique().tolist()
                preferred = [s for s in stats if "estimate" in s.lower()]
                chosen = preferred[0] if len(preferred) > 0 else stats[0]
                tmp = tmp[tmp[stat_col].astype(str) == chosen].copy()
            tmp = tmp.rename(columns={
                lower_cols["source"]: "source",
                lower_cols["condition"]: "condition",
                lower_cols["score"]: "score",
            })
            wide = tmp.pivot(index="condition", columns="source", values="score")
            wide = wide.loc[sample_names]
            return wide

        # already wide format
        if set(sample_names).issubset(set(res.index)):
            return res.loc[sample_names].copy()

    if isinstance(res, (tuple, list)):
        for x in res:
            try:
                return normalize_decoupler_output(x, sample_names)
            except Exception:
                continue

    if isinstance(res, dict):
        preferred_keys = ["estimate", "estimates", "score", "scores", "activity", "activities"]
        for k in preferred_keys:
            if k in res:
                return normalize_decoupler_output(res[k], sample_names)
        for _, v in res.items():
            try:
                return normalize_decoupler_output(v, sample_names)
            except Exception:
                continue

    # old API with AnnData in-place
    if hasattr(res, "obsm") and hasattr(res, "obs_names"):
        for key in res.obsm.keys():
            if key.endswith("_estimate"):
                mat = res.obsm[key]
                if isinstance(mat, pd.DataFrame):
                    return mat.loc[sample_names].copy()
                return pd.DataFrame(mat, index=res.obs_names).loc[sample_names].copy()

    raise RuntimeError("Could not normalize decoupler output into a sample x pathway DataFrame.")


def run_decoupler_method(data_df, net_df, method_name):
    """
    method_name: 'gsva' or 'mlm'
    """
    # Intersect genes
    common_genes = [g for g in data_df.columns if g in set(net_df["target"])]
    if len(common_genes) == 0:
        raise ValueError(f"No overlapping genes between data and network for method {method_name}.")
    data_df = data_df.loc[:, common_genes].copy()
    net_df = net_df[net_df["target"].isin(common_genes)].copy()

    # Try latest API first
    if hasattr(dc, "mt") and hasattr(dc.mt, method_name):
        fn = getattr(dc.mt, method_name)
        try:
            res = fn(data=data_df, net=net_df, verbose=False)
            return normalize_decoupler_output(res, data_df.index.tolist())
        except TypeError:
            # Some versions may require fewer args
            res = fn(data_df, net_df)
            return normalize_decoupler_output(res, data_df.index.tolist())
        except Exception as e:
            print(f"[WARN] latest decoupler mt.{method_name} failed, trying old API: {e}")

    # Fallback old API
    old_name = f"run_{method_name}"
    if hasattr(dc, old_name):
        fn = getattr(dc, old_name)
        kwargs = dict(mat=data_df, net=net_df, source="source", target="target", verbose=False)
        if "weight" in net_df.columns:
            kwargs["weight"] = "weight"
        try:
            res = fn(**kwargs)
            return normalize_decoupler_output(res, data_df.index.tolist())
        except Exception as e:
            raise RuntimeError(f"Both new and old decoupler APIs failed for {method_name}: {e}")

    raise RuntimeError(f"Could not find decoupler method for {method_name}.")


def pairwise_pathway_metrics(true_scores, pred_scores):
    rows = []
    for pert in true_scores.index.intersection(pred_scores.index):
        common = true_scores.columns.intersection(pred_scores.columns)
        if len(common) == 0:
            continue
        t = true_scores.loc[pert, common].astype(float).values
        p = pred_scores.loc[pert, common].astype(float).values
        rows.append({
            "perturbation": pert,
            "pathway_pearson": safe_pearson(t, p),
            "pathway_spearman": safe_spearman(t, p),
            "pathway_rmse": rmse(t, p),
            "n_pathways": len(common),
        })
    return pd.DataFrame(rows)


def evaluate_gsva(true_df, pred_df, reactome_net_df, model_name, mode, outdir):
    combined = pd.concat(
        [
            true_df.copy().rename(index=lambda x: f"true::{x}"),
            pred_df.copy().rename(index=lambda x: f"pred::{x}"),
        ],
        axis=0,
    )

    score_df = run_decoupler_method(combined, reactome_net_df, method_name="gsva")

    true_scores = score_df.loc[[f"true::{x}" for x in true_df.index]].copy()
    pred_scores = score_df.loc[[f"pred::{x}" for x in pred_df.index]].copy()
    true_scores.index = true_df.index
    pred_scores.index = pred_df.index

    detail = pairwise_pathway_metrics(true_scores, pred_scores)
    detail.to_csv(Path(outdir) / f"gsva_detail__{model_name}__{mode}.csv", index=False)

    summary = {
        "model": model_name,
        "profile_mode": mode,
        "method": "GSVA",
        "metric_gsva_pathway_pearson": detail["pathway_pearson"].mean(),
        "metric_gsva_pathway_spearman": detail["pathway_spearman"].mean(),
        "metric_gsva_pathway_rmse": detail["pathway_rmse"].mean(),
        "n_perturbations": len(detail),
    }
    return summary, detail


def get_progeny_network(top=500):
    # Latest API
    if hasattr(dc, "op") and hasattr(dc.op, "progeny"):
        net = dc.op.progeny(organism="human", top=top, verbose=False)
        if not isinstance(net, pd.DataFrame):
            net = pd.DataFrame(net)
        return net[["source", "target", "weight"]].copy()

    # Fallback older API
    if hasattr(dc, "get_progeny"):
        net = dc.get_progeny(organism="human", top=top)
        if not isinstance(net, pd.DataFrame):
            net = pd.DataFrame(net)
        cols = [c for c in ["source", "target", "weight"] if c in net.columns]
        return net[cols].copy()

    raise RuntimeError("Could not find PROGENy accessor in decoupler.")


def evaluate_progeny(true_df, pred_df, progeny_net_df, model_name, mode, outdir):
    combined = pd.concat(
        [
            true_df.copy().rename(index=lambda x: f"true::{x}"),
            pred_df.copy().rename(index=lambda x: f"pred::{x}"),
        ],
        axis=0,
    )

    # MLM is a standard choice for weighted PROGENy networks
    score_df = run_decoupler_method(combined, progeny_net_df, method_name="mlm")

    true_scores = score_df.loc[[f"true::{x}" for x in true_df.index]].copy()
    pred_scores = score_df.loc[[f"pred::{x}" for x in pred_df.index]].copy()
    true_scores.index = true_df.index
    pred_scores.index = pred_df.index

    detail = pairwise_pathway_metrics(true_scores, pred_scores)
    detail.to_csv(Path(outdir) / f"progeny_detail__{model_name}__{mode}.csv", index=False)

    summary = {
        "model": model_name,
        "profile_mode": mode,
        "method": "PROGENy",
        "metric_progeny_pathway_pearson": detail["pathway_pearson"].mean(),
        "metric_progeny_pathway_spearman": detail["pathway_spearman"].mean(),
        "metric_progeny_pathway_rmse": detail["pathway_rmse"].mean(),
        "n_perturbations": len(detail),
    }
    return summary, detail


def main():
    args = parse_args()
    ensure_dir(args.outdir)
    np.random.seed(args.seed)

    print("[INFO] Loading AnnData...")
    adata = ad.read_h5ad(args.adata)

    print("[INFO] Computing true pseudobulks from adata...")
    true_pseudobulks, n_cells, gene_to_idx, idx_to_gene = compute_true_pseudobulks(
        adata=adata,
        condition_key=args.condition_key,
        layer=args.layer,
    )
    control_profile = np.asarray(true_pseudobulks[tuple()])

    print("[INFO] Loading predictions...")
    preds = load_predictions(args.pred_pickle, n_genes=adata.n_vars)

    print("[INFO] Preparing Reactome pathways...")
    reactome_gmt = maybe_download_reactome_gmt(
        outdir=args.outdir,
        reactome_gmt=args.reactome_gmt,
        download_reactome=args.download_reactome,
        reactome_url=args.reactome_url,
    )
    reactome_sets = parse_gmt(reactome_gmt)
    reactome_sets = filter_gene_sets(
        reactome_sets,
        valid_genes=adata.var_names.tolist(),
        min_size=args.min_geneset_size,
        max_size=args.max_geneset_size,
    )
    reactome_net_df = gene_sets_to_long_df(reactome_sets)

    if len(reactome_sets) == 0:
        raise RuntimeError("No Reactome gene sets remain after filtering/intersection.")

    print(f"[INFO] Reactome pathways kept: {len(reactome_sets)}")

    print("[INFO] Fetching PROGENy network...")
    progeny_net_df = get_progeny_network(top=args.progeny_top)
    progeny_net_df = progeny_net_df[progeny_net_df["target"].isin(adata.var_names)].copy()

    if progeny_net_df.empty:
        raise RuntimeError("No overlap between PROGENy targets and adata.var_names.")

    modes = ["direct", "differential"] if args.profile_mode == "both" else [args.profile_mode]

    all_summaries = []

    for model_name, model_preds in preds.items():
        print(f"\n[INFO] Evaluating model: {model_name}")

        for mode in modes:
            print(f"[INFO]   Profile mode: {mode}")

            true_df, pred_df = build_profile_tables(
                true_pseudobulks=true_pseudobulks,
                pred_pseudobulks=model_preds,
                control_profile=control_profile,
                idx_to_gene=idx_to_gene,
                mode=mode,
            )

            # Save the actual true/pred profile matrices used for evaluation
            true_df.to_csv(Path(args.outdir) / f"true_profiles__{model_name}__{mode}.csv")
            pred_df.to_csv(Path(args.outdir) / f"pred_profiles__{model_name}__{mode}.csv")

            print("[INFO]     Running GSEA...")
            gsea_summary, _ = evaluate_gsea(
                true_df=true_df,
                pred_df=pred_df,
                gene_sets=reactome_sets,
                args=args,
                model_name=model_name,
                mode=mode,
                outdir=args.outdir,
            )
            all_summaries.append(gsea_summary)

            print("[INFO]     Running GSVA...")
            gsva_summary, _ = evaluate_gsva(
                true_df=true_df,
                pred_df=pred_df,
                reactome_net_df=reactome_net_df,
                model_name=model_name,
                mode=mode,
                outdir=args.outdir,
            )
            all_summaries.append(gsva_summary)

            print("[INFO]     Running PROGENy...")
            progeny_summary, _ = evaluate_progeny(
                true_df=true_df,
                pred_df=pred_df,
                progeny_net_df=progeny_net_df,
                model_name=model_name,
                mode=mode,
                outdir=args.outdir,
            )
            all_summaries.append(progeny_summary)

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(Path(args.outdir) / "metrics_summary.csv", index=False)
    print("\n[INFO] Done. Summary saved to:")
    print(Path(args.outdir) / "metrics_summary.csv")


if __name__ == "__main__":
    main()