import argparse
import math
import os
import pickle
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr


CTRL_TOKENS = {"ctrl", "control", "unperturbed", "untreated", ""}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot gene-wise mean prediction error vs. GO term count."
    )
    parser.add_argument("--adata", type=str, 
                        default = '/root/autodl-tmp/data/gse133344_k562gi_oe_pert227_84986_19264_withtotalcount/perturb_processed.h5ad',
                        help="Path to input .h5ad")
    parser.add_argument("--pred_pickle", type=str, 
                        default='pkl/method_pert_pred_dicts_dict.pkl',
                        help="Path to pickle containing {model_name: {pert_tuple: pred_vector}}")
    parser.add_argument(
        "--gene2go_pickle",
        type=str,
        default='/root/autodl-tmp/data/gene2go.pkl',
        help="Path to gene2go pickle: {gene_symbol: set/list of GO terms}",
    )
    parser.add_argument(
        "--output_pdf",
        type=str,
        default='fig/numGO_geneMSE_pa_norman.pdf',
        help="Path to output PDF figure",
    )
    parser.add_argument(
        "--condition_key",
        type=str,
        default="condition",
        help="obs key for perturbation condition",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        help="Optional AnnData layer to use instead of adata.X",
    )
    parser.add_argument(
        "--min_go_terms",
        type=int,
        default=0,
        help="Only genes with at least this many GO terms are included",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Scatter alpha",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=10.0,
        help="Scatter point size",
    )
    return parser.parse_args()


def get_matrix(adata, layer=None):
    if layer is None:
        return adata.X
    if layer not in adata.layers:
        raise KeyError(f"Layer '{layer}' not found in adata.layers")
    return adata.layers[layer]


def dense_mean(X):
    if sparse.issparse(X):
        return np.asarray(X.mean(axis=0)).ravel()
    return np.asarray(X).mean(axis=0).ravel()


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


def load_predictions(pred_pickle, n_genes):
    with open(pred_pickle, "rb") as f:
        raw = pickle.load(f)

    if not isinstance(raw, dict):
        raise ValueError("Prediction pickle must contain a dict at top level.")

    out = {}
    for model_name, inner in raw.items():
        if not isinstance(inner, dict):
            raise ValueError(f"Predictions for model '{model_name}' must be a dict.")
        model_dict = {}
        for k, v in inner.items():
            pert = canonicalize_pred_key(k)
            arr = np.asarray(v).reshape(-1)
            if arr.shape[0] != n_genes:
                raise ValueError(
                    f"Prediction length mismatch for model={model_name}, key={k}: "
                    f"got {arr.shape[0]}, expected {n_genes}"
                )
            model_dict[pert] = arr
        out[model_name] = model_dict
    return out


def compute_true_pseudobulks(adata, condition_key, layer=None):
    if condition_key not in adata.obs.columns:
        raise KeyError(f"{condition_key} not found in adata.obs")

    gene_to_idx = {g: i for i, g in enumerate(adata.var_names.tolist())}
    idx_to_gene = {i: g for i, g in enumerate(adata.var_names.tolist())}

    pert_tuples = []
    skipped = 0
    for cond in adata.obs[condition_key].tolist():
        pert = parse_condition_to_tuple(cond, gene_to_idx)
        if pert is None:
            skipped += 1
        pert_tuples.append(pert)

    if skipped > 0:
        print(f"[Warning] Skipped {skipped} cells because condition genes were not found in var_names.")

    X = get_matrix(adata, layer=layer)

    groups = {}
    for i, pert in enumerate(pert_tuples):
        if pert is None:
            continue
        groups.setdefault(pert, []).append(i)

    pseudobulks = {}
    n_cells = {}
    for pert, idxs in groups.items():
        pseudobulks[pert] = dense_mean(X[idxs])
        n_cells[pert] = len(idxs)

    return pseudobulks, n_cells, gene_to_idx, idx_to_gene


def load_gene2go(path):
    with open(path, "rb") as f:
        gene2go = pickle.load(f)

    if not isinstance(gene2go, dict):
        raise ValueError("gene2go pickle must contain a dict.")

    out = {}
    for gene, gos in gene2go.items():
        if gos is None:
            out[str(gene)] = set()
        elif isinstance(gos, set):
            out[str(gene)] = gos
        else:
            out[str(gene)] = set(gos)
    return out


def compute_genewise_mse(true_pseudobulks, pred_pseudobulks, n_genes):
    shared_perts = sorted(set(true_pseudobulks.keys()) & set(pred_pseudobulks.keys()))
    shared_perts = [p for p in shared_perts if len(p) > 0]  # exclude ctrl

    if len(shared_perts) == 0:
        raise ValueError("No shared non-control perturbations found between truth and predictions.")

    sqerr_sum = np.zeros(n_genes, dtype=float)
    count = 0

    for pert in shared_perts:
        true_vec = np.asarray(true_pseudobulks[pert]).reshape(-1)
        pred_vec = np.asarray(pred_pseudobulks[pert]).reshape(-1)
        sqerr_sum += (pred_vec - true_vec) ** 2
        count += 1

    mean_mse = sqerr_sum / max(count, 1)
    return mean_mse, count, shared_perts


def fit_line_and_corr(x, y):
    if len(x) < 2:
        return np.nan, np.nan, np.nan

    a, b = np.polyfit(x, y, deg=1)
    if np.std(x) == 0 or np.std(y) == 0:
        r = np.nan
    else:
        r, _ = pearsonr(x, y)
    return float(a), float(b), float(r)


def make_panel(ax, x, y, model_name, min_go_terms, n_perts, n_genes, alpha, point_size):
    ax.scatter(x, y, marker='x', alpha=alpha, s=point_size)
    a, b, r = fit_line_and_corr(x, y)

    if np.isfinite(a) and np.isfinite(b):
        x_line = np.linspace(float(np.min(x)), float(np.max(x)), 200)
        y_line = a * x_line + b
        ax.plot(x_line, y_line, linewidth=1)

    #ax.set_title(model_name)
    ax.set_xlabel(f"Number of GO terms per gene")
    ax.set_ylabel("Mean prediction MSE of each gene across perturbations")

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    text = (
        f"n genes = {n_genes}\n"
        f"y = {a:.4e}x + {b:.4e}\n"
        f"r = {r:.4f}"
    )
    #f"n perturbations = {n_perts}\n"
    ax.text(
        0.98,
        0.98,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )


def main():

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.sans-serif': ['DejaVu Sans'],
        'font.size': 9,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'axes.unicode_minus': False
    })

    args = parse_args()
    Path(args.output_pdf).parent.mkdir(parents=True, exist_ok=True)

    print("[Info] Loading AnnData...")
    adata = ad.read_h5ad(args.adata)
    n_genes = adata.n_vars

    print("[Info] Computing true pseudobulks from adata...")
    true_pseudobulks, n_cells, gene_to_idx, idx_to_gene = compute_true_pseudobulks(
        adata=adata,
        condition_key=args.condition_key,
        layer=args.layer,
    )

    print("[Info] Loading predictions...")
    preds = load_predictions(args.pred_pickle, n_genes=n_genes)

    print("[Info] Loading gene2go...")
    gene2go = load_gene2go(args.gene2go_pickle)

    gene_symbols = adata.var_names.tolist()
    go_counts = np.array([len(gene2go.get(g, set())) for g in gene_symbols], dtype=int)

    #model_names = list(preds.keys())
    model_names = ['PertAdapt']
    n_models = len(model_names)

    ncols = min(2, n_models)
    nrows = math.ceil(n_models / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    summary_rows = []

    for ax, model_name in zip(axes, model_names):
        print(f"[Info] Processing model: {model_name}")
        gene_mse, n_shared_perts, shared_perts = compute_genewise_mse(
            true_pseudobulks=true_pseudobulks,
            pred_pseudobulks=preds[model_name],
            n_genes=n_genes,
        )

        mask = go_counts >= args.min_go_terms
        x = go_counts[mask].astype(float)
        y = gene_mse[mask].astype(float)

        if len(x) == 0:
            raise ValueError(
                f"No genes passed min_go_terms={args.min_go_terms} for model {model_name}."
            )

        make_panel(
            ax=ax,
            x=x,
            y=y,
            model_name=model_name,
            min_go_terms=args.min_go_terms,
            n_perts=n_shared_perts,
            n_genes=len(x),
            alpha=args.alpha,
            point_size=args.point_size,
        )

        a, b, r = fit_line_and_corr(x, y)
        summary_rows.append(
            {
                "model": model_name,
                "n_shared_perturbations": n_shared_perts,
                "n_genes_plotted": len(x),
                "min_go_terms": args.min_go_terms,
                "slope_a": a,
                "intercept_b": b,
                "pearson_r": r,
            }
        )

    for i in range(len(model_names), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(args.output_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.splitext(args.output_pdf)[0] + ".summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"[Info] Figure saved to: {args.output_pdf}")
    print(f"[Info] Summary saved to: {summary_csv}")


if __name__ == "__main__":
    main()