#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Additional pathway-level evaluation for perturbation predictions:
1) ssGSEA
2) AUCell
3) Z-score scoring

Input:
- adata (.h5ad), obs[condition_key] contains:
    "GeneA+GeneB"   for double perturbation
    "ctrl+GeneB"    for single perturbation
    "ctrl"          for unperturbed
- pred_pickle:
    {model_name: {pert_tuple: pred_vector}}
- Reactome GMT (local or downloaded)

Output:
- metrics_summary.csv
- detailed per-perturbation CSVs for ssGSEA / AUCell / ZScore
"""

import argparse
import os
import pickle
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr, spearmanr

import anndata as ad
import gseapy as gp
import decoupler as dc


CTRL_TOKENS = {"ctrl", "control", "unperturbed", "untreated", ""}


def parse_args():
    parser = argparse.ArgumentParser(description="Pathway-level evaluation for perturbation predictions.")
    parser.add_argument("--adata", type=str,
                        default='/root/autodl-tmp/data/gse90063_k562_ko_tf20_37160_19264_withtotalcount/perturb_processed.h5ad',
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

    parser.add_argument("--min_geneset_size", type=int, default=10,
                        help="Minimum pathway size after intersecting with genes")
    parser.add_argument("--max_geneset_size", type=int, default=5000,
                        help="Maximum pathway size after intersecting with genes")

    parser.add_argument("--ssgsea_processes", type=int, default=1,
                        help="Number of processes for ssGSEA")
    parser.add_argument("--ssgsea_sample_norm_method", type=str, default="rank",
                        choices=["rank", "log", "log_rank", "custom"],
                        help="sample_norm_method passed to gseapy.ssgsea")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

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
    if layer not in adata.layers:
        raise KeyError(f"Layer '{layer}' not found in adata.layers")
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
        print(f"[WARN] {bad_conditions} cells have conditions containing genes not found in adata.var_names; skipping them.")

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


def normalize_decoupler_output(res, sample_names):
    """
    Normalize decoupler outputs to a sample x pathway DataFrame.
    """
    if isinstance(res, pd.DataFrame):
        lower_cols = {str(c).lower(): c for c in res.columns}

        # Long format
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

        # Some versions may use "sample" or "obs" instead of "condition"
        cond_aliases = ["condition", "sample", "obs", "name"]
        if "source" in lower_cols and "score" in lower_cols:
            cond_col = None
            for alias in cond_aliases:
                if alias in lower_cols:
                    cond_col = lower_cols[alias]
                    break
            if cond_col is not None:
                tmp = res.copy().rename(columns={
                    lower_cols["source"]: "source",
                    cond_col: "condition",
                    lower_cols["score"]: "score",
                })
                wide = tmp.pivot(index="condition", columns="source", values="score")
                wide = wide.loc[sample_names]
                return wide

        # Already wide with samples as index
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
    data_df: sample x genes
    net_df: long-format source-target (and optionally weight) DataFrame
    """
    common_genes = [g for g in data_df.columns if g in set(net_df["target"])]
    if len(common_genes) == 0:
        raise ValueError(f"No overlapping genes between data and network for method {method_name}.")

    data_df = data_df.loc[:, common_genes].copy()
    net_df = net_df[net_df["target"].isin(common_genes)].copy()

    # Latest API
    if hasattr(dc, "mt") and hasattr(dc.mt, method_name):
        fn = getattr(dc.mt, method_name)
        try:
            res = fn(data=data_df, net=net_df, verbose=False)
            return normalize_decoupler_output(res, data_df.index.tolist())
        except TypeError:
            res = fn(data_df, net_df)
            return normalize_decoupler_output(res, data_df.index.tolist())
        except Exception as e:
            print(f"[WARN] latest decoupler mt.{method_name} failed, trying old API: {e}")

    # Older API
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


def evaluate_ssgsea(true_df, pred_df, gene_sets, args, model_name, mode, outdir):
    """
    Uses GSEApy ssGSEA on combined matrix.
    Input to GSEApy should be genes x samples.
    """
    combined = pd.concat(
        [
            true_df.copy().rename(index=lambda x: f"true::{x}"),
            pred_df.copy().rename(index=lambda x: f"pred::{x}"),
        ],
        axis=0,
    )

    expr = combined.T  # genes x samples

    ss = gp.ssgsea(
        data=expr,
        gene_sets=gene_sets,
        outdir=None,
        sample_norm_method=args.ssgsea_sample_norm_method,
        min_size=args.min_geneset_size,
        max_size=args.max_geneset_size,
        processes=args.ssgsea_processes,
        no_plot=True,
        seed=args.seed,
        verbose=False,
    )

    res = ss.res2d.copy()
    # GSEApy docs show res2d with columns Name / Term / ES / NES
    if "Name" not in res.columns or "Term" not in res.columns:
        raise RuntimeError(f"Unexpected ssGSEA result columns: {list(res.columns)}")

    score_col = "NES" if "NES" in res.columns else "ES"
    score_df = res.pivot(index="Term", columns="Name", values=score_col).T

    true_scores = score_df.loc[[f"true::{x}" for x in true_df.index]].copy()
    pred_scores = score_df.loc[[f"pred::{x}" for x in pred_df.index]].copy()
    true_scores.index = true_df.index
    pred_scores.index = pred_df.index

    detail = pairwise_pathway_metrics(true_scores, pred_scores)
    detail.to_csv(Path(outdir) / f"ssgsea_detail__{model_name}__{mode}.csv", index=False)

    summary = {
        "model": model_name,
        "profile_mode": mode,
        "method": "ssGSEA",
        "metric_ssgsea_pathway_pearson": detail["pathway_pearson"].mean(),
        "metric_ssgsea_pathway_spearman": detail["pathway_spearman"].mean(),
        "metric_ssgsea_pathway_rmse": detail["pathway_rmse"].mean(),
        "n_perturbations": len(detail),
    }
    return summary, detail


def evaluate_aucell(true_df, pred_df, reactome_net_df, model_name, mode, outdir):
    combined = pd.concat(
        [
            true_df.copy().rename(index=lambda x: f"true::{x}"),
            pred_df.copy().rename(index=lambda x: f"pred::{x}"),
        ],
        axis=0,
    )

    score_df = run_decoupler_method(combined, reactome_net_df, method_name="aucell")

    true_scores = score_df.loc[[f"true::{x}" for x in true_df.index]].copy()
    pred_scores = score_df.loc[[f"pred::{x}" for x in pred_df.index]].copy()
    true_scores.index = true_df.index
    pred_scores.index = pred_df.index

    detail = pairwise_pathway_metrics(true_scores, pred_scores)
    detail.to_csv(Path(outdir) / f"aucell_detail__{model_name}__{mode}.csv", index=False)

    summary = {
        "model": model_name,
        "profile_mode": mode,
        "method": "AUCell",
        "metric_aucell_pathway_pearson": detail["pathway_pearson"].mean(),
        "metric_aucell_pathway_spearman": detail["pathway_spearman"].mean(),
        "metric_aucell_pathway_rmse": detail["pathway_rmse"].mean(),
        "n_perturbations": len(detail),
    }
    return summary, detail


def evaluate_zscore(true_df, pred_df, reactome_net_df, model_name, mode, outdir):
    combined = pd.concat(
        [
            true_df.copy().rename(index=lambda x: f"true::{x}"),
            pred_df.copy().rename(index=lambda x: f"pred::{x}"),
        ],
        axis=0,
    )

    score_df = run_decoupler_method(combined, reactome_net_df, method_name="zscore")

    true_scores = score_df.loc[[f"true::{x}" for x in true_df.index]].copy()
    pred_scores = score_df.loc[[f"pred::{x}" for x in pred_df.index]].copy()
    true_scores.index = true_df.index
    pred_scores.index = pred_df.index

    detail = pairwise_pathway_metrics(true_scores, pred_scores)
    detail.to_csv(Path(outdir) / f"zscore_detail__{model_name}__{mode}.csv", index=False)

    summary = {
        "model": model_name,
        "profile_mode": mode,
        "method": "ZScore",
        "metric_zscore_pathway_pearson": detail["pathway_pearson"].mean(),
        "metric_zscore_pathway_spearman": detail["pathway_spearman"].mean(),
        "metric_zscore_pathway_rmse": detail["pathway_rmse"].mean(),
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

            true_df.to_csv(Path(args.outdir) / f"true_profiles__{model_name}__{mode}.csv")
            pred_df.to_csv(Path(args.outdir) / f"pred_profiles__{model_name}__{mode}.csv")

            print("[INFO]     Running ssGSEA...")
            ssgsea_summary, _ = evaluate_ssgsea(
                true_df=true_df,
                pred_df=pred_df,
                gene_sets=reactome_sets,
                args=args,
                model_name=model_name,
                mode=mode,
                outdir=args.outdir,
            )
            all_summaries.append(ssgsea_summary)

            print("[INFO]     Running AUCell...")
            aucell_summary, _ = evaluate_aucell(
                true_df=true_df,
                pred_df=pred_df,
                reactome_net_df=reactome_net_df,
                model_name=model_name,
                mode=mode,
                outdir=args.outdir,
            )
            all_summaries.append(aucell_summary)

            print("[INFO]     Running Z-score scoring...")
            zscore_summary, _ = evaluate_zscore(
                true_df=true_df,
                pred_df=pred_df,
                reactome_net_df=reactome_net_df,
                model_name=model_name,
                mode=mode,
                outdir=args.outdir,
            )
            all_summaries.append(zscore_summary)

    summary_df = pd.DataFrame(all_summaries)
    summary_path = Path(args.outdir) / "metrics_summary_additional_methods.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n[INFO] Done. Summary saved to:")
    print(summary_path)


if __name__ == "__main__":
    main()