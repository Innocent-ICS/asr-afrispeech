"""
asrking2.py - Full Experiment Script

This script runs all six ASR RNN experiments on the complete AfriSpeech-200
Shona dataset. It trains and evaluates all model variants (Vanilla RNN, LSTM,
GRU, each with and without attention) and logs results to WandB and TensorBoard.

Requirements addressed:
- 3.2: Provide script that downloads and processes complete dataset
- 3.4: Complete entire workflow using full dataset
- 3.5: Implement identical pipeline logic (differs only in data size)
- 1.5: Execute experiments in specified order
"""

import os
import sys
import torch
from pathlib import Path
from datetime import datetime

# Import experiment runner and configuration
from experiments.runner import ExperimentRunner
from utils.config import config


def main():
    """
    Main function to run all six experiments on the full dataset.
    
    Runs all experiments in the specified order:
    1. Vanilla RNN
    2. Vanilla RNN with attention
    3. LSTM
    4. LSTM with attention
    5. GRU
    6. GRU with attention
    """
    print("\n" + "="*80)
    print("ASR RNN SYSTEM - FULL EXPERIMENTS (asrking2.py)")
    print("="*80)
    print("\nThis script runs all six experiments on the complete dataset.")
    print("This will take significant time and computational resources.")
    print("="*80 + "\n")
    
    # Create a copy of config for full experiments
    full_config = config.copy()
    
    # Set subset_size to None to load full dataset
    full_config['subset_size'] = None  # Load complete dataset
    
    print("Configuration for full experiments:")
    print(f"  Subset size: Full dataset (all samples)")
    print(f"  Number of epochs: {full_config['num_epochs']}")
    print(f"  Batch size: {full_config['batch_size']}")
    print(f"  Hidden size: {full_config['hidden_size']}")
    print(f"  Number of layers: {full_config['num_layers']}")
    print(f"  Learning rate: {full_config['learning_rate']}")
    print(f"  Gradient clip: {full_config['gradient_clip']}")
    print(f"  Dropout: {full_config['dropout']}")
    print()
    
    # Load WandB token from .env file
    wandb_token = None
    env_path = Path('.env')
    if env_path.exists():
        try:
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('wandb_token'):
                        wandb_token = line.split('=', 1)[1].strip()
                        print("✓ WandB token loaded from .env file")
                        break
        except Exception as e:
            print(f"Warning: Failed to load WandB token from .env: {e}")
            print("Continuing without WandB logging...")
    else:
        print("Warning: .env file not found. Continuing without WandB logging...")
        print("To enable WandB logging, create a .env file with: wandb_token=YOUR_TOKEN")
    
    print()
    
    # Check for GPU availability
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠ No GPU available. Training will use CPU (this will be slow).")
    
    print()
    
    # Initialize experiment runner
    print("Initializing ExperimentRunner...")
    print("This will download the full dataset if not already cached...")
    print()
    
    try:
        runner = ExperimentRunner(config=full_config, wandb_token=wandb_token)
    except Exception as e:
        print(f"\n{'!'*80}")
        print("ERROR: Failed to initialize ExperimentRunner")
        print(f"{'!'*80}")
        print(f"\nError details: {e}")
        print("\nPlease check:")
        print("  1. Internet connection (for downloading dataset)")
        print("  2. Sufficient disk space (dataset is ~1-2 GB)")
        print("  3. Required dependencies are installed (see requirements.txt)")
        print(f"\n{'!'*80}\n")
        
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)
    
    # Record start time
    start_time = datetime.now()
    print(f"\nStarting experiments at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run all six experiments in specified order
    print("\n" + "="*80)
    print("RUNNING ALL SIX EXPERIMENTS")
    print("="*80)
    print("\nExperiments will be run in the following order:")
    print("  1. Vanilla RNN")
    print("  2. Vanilla RNN with attention")
    print("  3. LSTM")
    print("  4. LSTM with attention")
    print("  5. GRU")
    print("  6. GRU with attention")
    print("\nThis will take several hours depending on your hardware.")
    print("Results will be logged to WandB and TensorBoard.")
    print("="*80 + "\n")
    
    try:
        # Run all experiments
        all_results = runner.run_all_experiments()
        
        # Record end time
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*80)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\nStart time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total duration: {duration}")
        print()
        
        # Display summary of results
        print("="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print()
        
        for exp_name, results in all_results.items():
            print(f"{exp_name}:")
            print(f"  CTC Loss: {results.get('ctc_loss', 'N/A'):.4f}")
            print(f"  Accuracy: {results.get('accuracy', 'N/A'):.4f}")
            print(f"  Perplexity: {results.get('perplexity', 'N/A'):.4f}")
            print(f"  CER: {results.get('cer', 'N/A'):.4f}")
            print(f"  WER: {results.get('wer', 'N/A'):.4f}")
            print()
        
        print("="*80)
        print("\nResults have been logged to:")
        print("  - WandB (if token was provided)")
        print("  - TensorBoard (logs/tensorboard/)")
        print("  - Loss plots (logs/plots/)")
        print("\nTo view TensorBoard results, run:")
        print("  tensorboard --logdir=logs/tensorboard")
        print("="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("EXPERIMENTS INTERRUPTED BY USER")
        print("="*80)
        print("\nPartial results may have been saved.")
        print("You can check WandB and TensorBoard for completed experiments.")
        print("="*80 + "\n")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n{'!'*80}")
        print("ERROR: Experiments failed!")
        print(f"{'!'*80}")
        print(f"\nError details: {e}")
        print("\nPlease check:")
        print("  1. Sufficient memory is available (consider reducing batch_size)")
        print("  2. Sufficient disk space for checkpoints and logs")
        print("  3. CUDA is working properly if using GPU")
        print("  4. Dataset was loaded successfully")
        print(f"\n{'!'*80}\n")
        
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
