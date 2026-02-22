# Limb Functional Predictors

A deep learning framework for predicting **genomic regulatory signals** directly from DNA sequence.

These are the **binary classification** models I trained in my publications:

- **bat limb MEIS2 binding affinity (401bp)**
- **bat forelimb-specific MEIS2 accessibility (501bp)**
- **mouse limb TWIST1 binding affinity (401bp)**
- **Boreoeutherian forelimb accessibility early (500bp)**
- **Boreoeutherian forelimb accessibility late (500bp)**

The two binding models were trained on accessible regions and should work on regulatory sequences (eg. ATAC-seq, TF ChIP-seq peaks, conserved elements). The MEIS accessibility model was only trained on regions with MEIS binding affinity. The forelimb accessibility models were trained on ATAC-seq peaks and their orthologs. Predictions outside the original data range will yield unreliable or unintepretable results.

---

## Quick Start

This project uses [`uv`](https://astral.sh/uv/) for fast and reproducible environment management.

### Prerequisites

Install `uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup Environment

Clone the repository and synchronize dependencies:

```bash
git clone https://github.com/BaiweiLo/Bat-MEIS-models.git
cd Bat-MEIS-models
uv sync
```

---

## Model Specification

### Input
- Fasta sequences 
- Fixed sequence length (see above description of the models)

### Output
A tab separated table. The first column is the name of the sequences, followed by predicted probabilities for classification tasks and scores for regression tasks. 


## Running Predictions

```bash
uv run python predict_fasta.py <model_path.h5> <input.fasta> <expected_length> > results.tsv
```

### Arguments

| Argument | Description |
|----------|-------------|
| `<model_path.h5>` | Trained model file |
| `<input.fasta>` | FASTA file containing sequences |
| `<expected_length>` | Required sequence length |

---
