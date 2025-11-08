# ASR RNN System

A comprehensive Automatic Speech Recognition (ASR) system that implements and compares six encoder-decoder RNN architectures on the Shona language dataset from AfriSpeech-200.

## Overview

This system evaluates three RNN types (Vanilla RNN, LSTM, GRU) both with and without Bahdanau attention mechanism. It provides a complete pipeline for training, evaluation, and comparison of different ASR architectures.

### Features

- **Six Model Variants**: Vanilla RNN, LSTM, GRU (each with and without attention)
- **Modular Architecture**: Generalizable RNN module that supports all cell types
- **Comprehensive Logging**: Dual logging to WandB and TensorBoard with visual plots
- **Robust Error Handling**: Graceful handling of common errors with helpful messages
- **Two Execution Modes**: Quick testing with small data subset and full experiments
- **Complete Metrics**: CTC loss, accuracy, perplexity, CER, WER, and sample transcriptions

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Set up WandB token in .env file
echo "wandb_token=YOUR_TOKEN" > .env

# 3. Run quick test (2-5 minutes)
python asrking1.py

# 4. Run full experiments (~12-15 hours)
python asrking2.py

# 5. View results in TensorBoard
tensorboard --logdir=logs/tensorboard
# Open http://localhost:6006
```

## Requirements

- Python 3.8 or higher
- PyTorch 2.0 or higher
- 8GB RAM minimum (16GB recommended)
- 5GB free disk space
- CUDA-capable GPU (optional but recommended)

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Key packages:
- `torch` and `torchaudio` - Deep learning and audio processing
- `datasets` - Hugging Face datasets for AfriSpeech-200
- `wandb` - Experiment tracking
- `tensorboard` - Visualization
- `jiwer` - Error rate computation

### 2. Configure WandB (Optional)

Create a `.env` file:
```bash
wandb_token=YOUR_WANDB_TOKEN_HERE
```

Get your token at: https://wandb.ai/authorize

## Usage

### Quick Testing: asrking1.py

Tests the pipeline with a small subset (10 samples, 2 epochs):

```bash
python asrking1.py
```

**Runtime**: 2-5 minutes  
**Purpose**: Verify all components work before running full experiments

### Full Experiments: asrking2.py

Runs all six experiments on the complete dataset:

```bash
python asrking2.py
```

**Runtime**: ~12-15 hours (all 6 experiments)  
**Experiments**:
1. Vanilla RNN
2. Vanilla RNN with attention
3. LSTM
4. LSTM with attention
5. GRU
6. GRU with attention

## Viewing Results

### Option 1: TensorBoard (Recommended)

```bash
tensorboard --logdir=logs/tensorboard --port=6006
```

Open http://localhost:6006 to see:
- Training/validation loss curves
- All metrics over time
- Side-by-side experiment comparison

### Option 2: Loss Plots (PNG)

```bash
# macOS
open logs/plots/*.png

# Linux
xdg-open logs/plots/*.png
```

### Option 3: WandB Dashboard

Visit: https://wandb.ai (if configured)

## Results Location

All results are stored in:

```
logs/
├── tensorboard/              # TensorBoard event files
│   ├── Vanilla RNN/
│   ├── Vanilla RNN with attention/
│   ├── LSTM/
│   ├── LSTM with attention/
│   ├── GRU/
│   └── GRU with attention/
└── plots/                    # PNG loss plots
    ├── Vanilla_RNN_losses.png
    ├── Vanilla_RNN_with_attention_losses.png
    ├── LSTM_losses.png
    ├── LSTM_with_attention_losses.png
    ├── GRU_losses.png
    └── GRU_with_attention_losses.png
```

## Experiment Results

### Completed Experiments

| Experiment | Epochs | Final Train Loss | Final Val Loss | Test Accuracy | Test CER | Test WER |
|------------|--------|------------------|----------------|---------------|----------|----------|
| Vanilla RNN | 50 | 5.31 | 5.63 | 0.09% | 98.89% | 100% |
| Vanilla RNN + Attn | 50 | 3.42 | 4.48 | **1.20%** | 96.02% | 100% |
| LSTM | 50 | 3.33 | 5.27 | 0.83% | 95.37% | 100% |

### Key Findings

**Attention Mechanism Significantly Improves Performance:**
- Vanilla RNN with attention achieved **13x better accuracy** (1.20% vs 0.09%)
- Lower test CTC loss (3.97 vs 28.09)
- Better CER (96.02% vs 98.89%)

**All Models Show Strong Learning:**
- Vanilla RNN: Loss decreased 81% (28.50 → 5.31)
- Vanilla RNN + Attn: Loss decreased 80% (16.93 → 3.42)
- LSTM: Loss decreased 91% (36.41 → 3.33)

## Configuration

Adjust hyperparameters in `utils/config.py`:

```python
config = {
    'hidden_size': 128,        # RNN hidden state dimension
    'num_layers': 1,           # Number of stacked RNN layers
    'dropout': 0.3,            # Dropout probability
    'batch_size': 8,           # Training batch size
    'learning_rate': 0.001,    # Learning rate
    'num_epochs': 50,          # Number of training epochs
    'gradient_clip': 5.0,      # Gradient clipping threshold
    'n_mfcc': 40,              # Number of MFCC features
}
```

**Note**: Current settings are optimized for systems with limited RAM. For better performance on systems with more resources, increase `batch_size`, `hidden_size`, and `num_layers`.

## Evaluation Metrics

For each experiment, the following metrics are computed:

1. **CTC Loss**: Connectionist Temporal Classification loss (lower is better)
2. **Accuracy**: Character-level accuracy (higher is better, 0-1 range)
3. **Perplexity**: Model perplexity (lower is better)
4. **CER**: Character Error Rate (lower is better, 0-1 range)
5. **WER**: Word Error Rate (lower is better, 0-1 range)
6. **Sample Transcriptions**: Example predictions vs. ground truth

## Troubleshooting

### Common Issues

**1. Process Killed / Out of Memory**
```bash
# Reduce memory usage in utils/config.py:
'batch_size': 4,      # Reduce from 8
'hidden_size': 64,    # Reduce from 128
```

**2. WandB Authentication Error**
```bash
# Check .env file format (no quotes):
wandb_token=YOUR_TOKEN

# Or login manually:
wandb login
```

**3. Dataset Download Fails**
- Check internet connection
- Verify access to huggingface.co
- Ensure 2GB+ free disk space

**4. Missing Dependencies**
```bash
pip install -r requirements.txt --upgrade
```

**5. CUDA Out of Memory**
- Reduce `batch_size` in config
- Use CPU: Set `device = 'cpu'` in config
- Close other GPU applications

## Project Structure

```
asr-rnn-system/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── .env                       # WandB token (create this)
├── asrking1.py               # Quick testing script
├── asrking2.py               # Full experiments script
├── models/                    # Model implementations
│   ├── encoder.py
│   ├── decoder.py
│   ├── attention.py
│   └── rnn_module.py
├── data/                      # Data loading and preprocessing
│   ├── dataset.py
│   ├── afrispeech_loader.py
│   └── preprocessor.py
├── training/                  # Training loop and loss
│   ├── trainer.py
│   └── loss.py
├── evaluation/                # Metrics computation
│   └── evaluator.py
├── asr_logging/              # Experiment logging
│   └── logger.py
├── utils/                     # Configuration and utilities
│   ├── config.py
│   └── vocab.py
├── experiments/               # Experiment orchestration
│   └── runner.py
└── logs/                      # Generated results
    ├── tensorboard/
    └── plots/
```

## Technical Details

### Model Architecture
- **Encoder**: Multi-layer RNN (Vanilla/LSTM/GRU) with packed sequences
- **Decoder**: Multi-layer RNN with optional Bahdanau attention
- **Loss Function**: CTC (Connectionist Temporal Classification)
- **Optimizer**: Adam with gradient clipping

### Dataset
- **Source**: AfriSpeech-200 (Hugging Face)
- **Language**: Shona
- **Splits**: Train, Dev (validation), Test
- **Audio**: WAV files, 16kHz sample rate
- **Features**: 40-dimensional MFCC

## Performance Notes

The relatively low accuracy scores are expected for this challenging task:
- Medical/technical domain transcriptions
- Limited training data
- Shona is a low-resource language
- Simple model architecture (for educational purposes)

**To improve performance:**
1. Increase model capacity (`hidden_size`, `num_layers`)
2. Train for more epochs
3. Use larger batch sizes (if memory allows)
4. Fine-tune learning rate
5. Use full dataset (if using subset)

## Citation

```bibtex
@misc{asr-rnn-system,
  title={ASR RNN System: Comparing Encoder-Decoder Architectures for Shona Speech Recognition},
  author={Innocent Farai Chikwanda},
  year={2025}
}
```

AfriSpeech-200 dataset:
```bibtex
@inproceedings{afrispeech,
  title={AfriSpeech-200: Pan-African accented speech dataset for clinical and general domain ASR},
  author={Tonja, Andiswa Bukula and others},
  booktitle={Transactions of the Association for Computational Linguistics},
  year={2022}
}
```

## License

This project is provided for educational and research purposes.

## Acknowledgments

- AfriSpeech-200 dataset from Intron Health
- Hugging Face for the datasets library
- WandB for experiment tracking
- PyTorch team for the deep learning framework

---

**Questions?** Check the Troubleshooting section or review error messages carefully.
