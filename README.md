# Machine Translate (GRU / Transformer / mT5)

This repository implements a machine translation system with three model families:

- **GRU-based Seq2Seq** with attention (dot / general / concat) and different training policies (teacher forcing / free running / mixed)
- **Transformer** with architecture ablations (Absolute vs ALiBi positional encoding, LayerNorm vs RMSNorm)
- **mT5** fine-tuning baseline

The project supports training, evaluation (BLEU), inference, and plotting validation curves for ablation analysis.

---

## 1. Project Structure
```
.
├── ckpts/ # Saved checkpoints
│ ├── gru/
│ ├── transformer/
│ ├── t5/
│ ├── t5_best/
│ └── t5_results/
├── data/ # Dataset files (jsonl + txt)
│ ├── train_10k.jsonl
│ ├── train_100k.jsonl
│ ├── valid.jsonl
│ ├── test.jsonl
│ ├── train_100k.en.txt
│ └── train_100k.zh.txt
├── fig/ # Figures for report
│ ├── gru/
│ ├── t5/
│ └── transformer_ablation_plots/
├── logs/ # Training logs
│ ├── gru_force_concat.log
│ ├── gru_force_dot.log
│ ├── gru_force_general.log
│ ├── gru_free_concat.log
│ ├── t5_log.txt
│ ├── transformer_abs_layernorm.txt
│ ├── transformer_abs_rmsnorm.txt
│ ├── transformer_alibi_layernorm.txt
│ └── transformer_alibi_rmsnorm.txt
├── models/ # Model definitions
│ ├── gru.py
│ ├── transformer.py
│ └── t5.py
├── scripts/ # Run scripts
├── spm/ # SentencePiece model + vocab
├── config.py # Main configuration
├── custom_data.py # Dataset loader
├── train_spm.py # Train SentencePiece model
├── prepare_spm_text.py # Prepare text for SPM training
├── main.py # Main entry: train/eval
├── inference.py # Inference pipeline
├── translate.py # Translation utils / decode functions
├── utils.py # Common utilities (BLEU, etc.)
└── README.md
```

---

## 2. Environment Setup

We recommend Python >= 3.9.

```bash
pip install -r requirements.txt
```


## 3. Data Format
Training/validation/test data are stored as jsonl files under data/.
Each line is one example:

```{"src": "...", "tgt": "..."}```

Example files:

- train_10k.jsonl / train_100k.jsonl

- valid.jsonl

- test.jsonl

## 4. SentencePiece Tokenization (Optional / Required for some settings)
Prepare text for SPM:

```bash
python prepare_spm_text.py
```
Train SPM model:

```bash
python train_spm.py
```
SPM model and vocab will be stored under spm/.

## 5. Training & Running (Using Scripts)

We provide runnable shell scripts under `scripts/` to reproduce training and inference for all three models:

- `gru.sh`: train GRU Seq2Seq + attention model
- `transformer.sh`: train scratch Transformer (with ablations)
- `t5.sh`: fine-tune pretrained mT5 model
- `inference.sh`: run inference / translation using a trained checkpoint

Before running scripts, make sure you are in the project root directory.

Our pretrained checkpoint weights are provided via the [link](https://drive.google.com/drive/folders/1vbxMiyksoxrdOPQCUKaglovyK-l5yOV-?usp=drive_link).

### 1) Train GRU Seq2Seq + Attention

```bash
bash scripts/gru.sh
```
This script trains the GRU-based Seq2Seq model (with attention) and saves checkpoints to ckpts/gru/.
Training logs are saved under logs/.

### 2) Train Transformer (Scratch)
```
bash scripts/transformer.sh
```

This script trains a lightweight Transformer from scratch and runs architectural ablations over:

- positional encoding (absolute vs ALiBi)

- normalization (LayerNorm vs RMSNorm)

Checkpoints are saved under ckpts/transformer/, and logs under logs/

### 3) Fine-tune mT5
```
bash scripts/t5.sh
```
This script fine-tunes a pretrained mT5-small model on the same dataset.
Checkpoints and results are saved under ckpts/t5/ and related directories (t5_best/, t5_results/).

### 4) Testing on the Test Set

For all three scripts (`gru.sh`, `transformer.sh`, `t5.sh`), you can switch from training to evaluation by modifying the script arguments and setting:

```bash
--mode test
```
### 5) Run Inference
```
bash scripts/inference.sh
```

This script loads a trained checkpoint and runs translation inference on test samples.
You may need to set the checkpoint path inside the script depending on which model you want to evaluate.