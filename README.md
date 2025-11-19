# PertAdapt: Unlocking Single-Cell Foundation Models for Genetic Perturbation Prediction via Condition-Sensitive Adaptation

<p align="center"><img src="https://github.com/BaiDing1234/PertAdapt/blob/main/img/model.png" alt="attnpert" width="900px" /></p>

---

## 🔍 Overview

**PertAdapt** is a plug-in perturbation adapter that unlocks the predictive power of large **single-cell foundation models (FMs)** for **genetic perturbation response prediction**.

Built on top of **scFoundation** and **AIDO.Cell**, PertAdapt introduces:

- A **condition-sensitive perturbation adapter** with a **gene-similarity–masked attention mechanism**
- An **adaptive loss** to handle the imbalance between perturbation-sensitive and perturbation-insensitive genes

PertAdapt is evaluated on multiple **single-gene** and **double-gene** perturbation datasets across diverse cell lines, and this repository provides the necessary code and scripts to **reproduce the main experimental results and figures**.

---

## 📦 Installation

This project uses `conda` for environment management.

Clone the repository:

```bash
git clone https://github.com/BaiDing1234/PertAdapt.git
cd PertAdapt
```

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate pertadapt
```

Make sure the environment is correctly set up by running:

```bash
python -c "import torch; print(torch.__version__)"
```

---

## 📂 Dataset Download & Preparation

All preprocessed datasets and auxiliary files required to run PertAdapt are hosted on OneDrive:

🔗 **OneDrive link:**  
https://mbzuaiac-my.sharepoint.com/:f:/g/personal/ding_bai_mbzuai_ac_ae/Es93Oj9ghjFCtey-UmPlcYkBjcllLeXIaXH-xCC7MD5zaA?e=odmNPM

In this shared folder, the following files are **required** for running the model:

- `go.csv`
- `go_mask_19264.npz`
- `scfoundation_genes.txt`

These files contain:

- The GO term graph and annotations (`go.csv`)
- The GO-based attention mask for gene similarity (`go_mask_19264.npz`)
- The gene list used by scFoundation, ensuring consistent gene ordering (`scfoundation_genes.txt`)

You will also find several **preprocessed perturbation datasets** used for training and evaluation. Their filenames correspond to:

- **Norman (K562 double-gene perturbations)**  
  `gse133344_k562gi_oe_pert227_84986_19264_withtotalcount`
- **Replogle RPE1**  
  `replogle_rpe1_19264_withtotalcount`
- **Replogle K562**  
  `replogle_k562_19264_withtotalcount`
- **Nadig HepG2**  
  `nadig_hepg2_19264_withtotalcount_filtered`
- **Nadig Jurkat**  
  `nadig_jurkat_19264_withtotalcount_filtered`
- **Adamson K562**  
  `gse90546_k562_63587_19264_10k_log1p_withtotalcount`
- **Dixit K562**  
  `gse90063_k562_ko_tf20_37160_19264_withtotalcount`

📌 **Download the datasets you wish to use (e.g., Norman, Replogle, Adamson, Dixit) and place them in the appropriate data directories expected by the scripts (see comments in the training scripts for exact paths).**

---

## 🚀 Running Experiments

Below we describe how to run PertAdapt on the **Norman** dataset as an example. The same pattern can be extended to other datasets.

### 1. scFoundation + PertAdapt

1. Navigate to the scFoundation-based implementation:

```bash
cd scFoundation
```

2. Download the pretrained **scFoundation** checkpoint from:

🔗 https://hopebio2020.sharepoint.com/sites/PublicSharedfiles/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FPublicSharedfiles%2FShared%20Documents%2FPublic%20Shared%20files&amp;p=true&amp;ga=1

3. Place the downloaded checkpoint into:

```text
scFoundation/models/
```

4. Run PertAdapt on the Norman dataset:

```bash
cd PertAdapter
bash run_sh/run_norman.sh
```

This script will:

- Load the scFoundation backbone
- Attach the PertAdapt module
- Train/evaluate on the Norman dataset
- Save logs and results to the corresponding output directory

---

### 2. AIDO.Cell + PertAdapt

1. Navigate to the AIDO.Cell-based implementation:

```bash
cd AIDOCell
```

2. Download **AIDO.Cell-3M**, **AIDO.Cell-10M**, and/or **AIDO.Cell-100M** checkpoints from:

🔗 https://huggingface.co/genbio-ai/collections

3. Place all model weight files into:

```text
AIDOCell/AIDO.Cell-3M/
AIDOCell/AIDO.Cell-10M/
AIDOCell/AIDO.Cell-100M/
```

4. Navigate to the PertAdapt wrapper and run the Norman experiments:

```bash
cd PertAdapter

# AIDO.Cell-100M backbone
bash run_sh/run_norman.sh

# AIDO.Cell-3M backbone (for scaling trend experiments)
bash run_sh/run_norman_3m.sh

# AIDO.Cell-10M backbone (for scaling trend experiments)
bash run_sh/run_norman_10m.sh
```

These scripts reproduce the scaling experiments across different FM sizes reported in the paper.

---

## 📊 Reproducing Figures

To reproduce the figures in the paper, go to the result processing directory:

```bash
cd results_process
```

Download the following files from the OneDrive link:

- `norman_magnitude_mse.csv`
- `scaling_trend.pkl`
- `go_mask_19264.npz`

Then run the plotting scripts:

- **Figure 3 (Magnitude & distance-correlation MSE for gene interaction metrics):**

```bash
python magnitude_dcor_mse_plot.py
```

- **Figure 4 (Scaling trend across FM sizes):**

```bash
python scaling_trend_plot.py
```

- **Figure A1 (GO statistics based on the gene-similarity mask):**

```bash
python go_stats_fig.py
```

The generated figures correspond to those shown in the main manuscript and appendix.

---

## 📁 Repository Structure (High-Level)

A high-level view of the repository:

```text
PertAdapt/
├── scFoundation/           # Experiments with scFoundation backbone
│   ├── models/             # Pretrained scFoundation weights (to be placed here)
│   ├── PertAdapter/        # PertAdapt implementation + scripts for scFoundation
│   └── ...
├── AIDOCell/               # Experiments with AIDO.Cell backbone
│   ├── AIDO.Cell-3M/       # AIDO.Cell-3M weights (to be placed here)
│   ├── AIDO.Cell-10M/      # AIDO.Cell-10M weights
│   ├── AIDO.Cell-100M/     # AIDO.Cell-100M weights
│   └── PertAdapter/        # PertAdapt implementation + scripts for AIDO.Cell
├── results_process/        # Scripts for reproducing paper figures
├── img/                    # Model architecture illustration and other images
├── environment.yml         # Conda environment specification
└── README.md               # This document
```

---

## 🙏 Acknowledgements

This project makes use of content and code from:

- **[scFoundation](https://github.com/biomap-research/scFoundation)**  
- **[AIDO.Cell](https://github.com/genbio-ai/ModelGenerator)**  

These works were instrumental in the development of PertAdapt.  
We are deeply grateful to the original authors and contributors for making their models and code publicly available.
