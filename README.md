# Genomic Sequence Predictor

A deep learning framework for predicting **genomic regulatory signals** directly from DNA sequence.

The following scripts performs **binary classification** tasks for models I trained and reported in my publications:

- **bat limb MEIS2 binding affinity (401bp)**
- **bat forelimb-specific MEIS2 accessibility (501bp)**
- **mouse limb TWIST1 binding affinity (401bp)**
- **forelimb accessibility (E11.5) (500bp)**
- **forelimb accessibility (E13.5) (500bp)**

The two binding models were trained on accessible regions and should work on regulatory sequences (eg. conserved elements). The MEIS accessibility model was only trained on MEIS binding regions. The forelimb accessibility models were trained on ATAC-seq peaks and their orthologs. Predictions outside these data range will yield unreliable results.

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
git clone https://github.com/your-username/genomic-predictor.git
cd genomic-predictor
uv sync
```

---

## Model Specification

### Input
- Fasta sequences 
- Fixed sequence length (see above description of the models)

### Output
Predicted probabilities for the classification task.


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
