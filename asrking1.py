"""
asrking1.py - Quick Pipeline Test Script

This script runs a quick test of the ASR RNN system pipeline using a small
subset of the data. It's designed to verify that all components work together
correctly before running full experiments.

Requirements addressed:
- 3.1: Provide script that downloads and processes small subset of data
- 3.3: Complete entire workflow using small data subset
- 3.5: Implement identical pipeline logic (differs only in data size)
"""

import os
import sys
import torch
from pathlib import Path

# Import experiment runner and configuration
from experiments.runner import ExperimentRunner
from utils.config import config


def main():
    """
    Main function to run quick pipeline test.
    
    Runs one experiment (Vanilla RNN) on a small data subset to verify
    the entire pipeline works end-to-end.
    """
    print("\n" + "="*80)
    print("ASR RNN SYSTEM - QUICK PIPELINE TEST (asrking1.py)")
    print("="*80)
    print("\nThis script runs a quick test with a small data subset to verify")
    print("that the entire pipeline works correctly before running full experiments.")
    print("="*80 + "\n")
    
    # Create a copy of config for quick testing
    test_config = config.copy()
    
    # Set subset_size to small value for quick testing
    test_config['subset_size'] = 10  # Use only 10 samples per split
    
    # Reduce training parameters for quick testing
    test_config['num_epochs'] = 2  # Only 2 epochs for quick test
    test_config['batch_size'] = 4  # Smaller batch size
    
    print("Configuration for quick test:")
    print(f"  Subset size: {test_config['subset_size']} samples per split")
    print(f"  Number of epochs: {test_config['num_epochs']}")
    print(f"  Batch size: {test_config['batch_size']}")
    print(f"  Hidden size: {test_config['hidden_size']}")
    print(f"  Number of layers: {test_config['num_layers']}")
    print(f"  Learning rate: {test_config['learning_rate']}")
    print(f"  Gradient clip: {test_config['gradient_clip']}")
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
                        print("WandB token loaded from .env file")
                        break
        except Exception as e:
            print(f"Warning: Failed to load WandB token from .env: {e}")
            print("Continuing without WandB logging...")
    else:
        print("Warning: .env file not found. Continuing without WandB logging...")
    
    print()
    
    # Initialize experiment runner
    print("Initializing ExperimentRunner...")
    try:
        runner = ExperimentRunner(config=test_config, wandb_token=wandb_token)
    except Exception as e:
        print(f"\nERROR: Failed to initialize ExperimentRunner: {e}")
        print("\nPlease check:")
        print("  1. Internet connection (for downloading dataset)")
        print("  2. Disk space (for storing dataset)")
        print("  3. Required dependencies are installed (see requirements.txt)")
        sys.exit(1)
    
    # Run a single experiment to test the pipeline
    print("\n" + "="*80)
    print("Running single experiment to test pipeline...")
    print("="*80 + "\n")
    
    try:
        # Run Vanilla RNN experiment (simplest configuration)
        results = runner.run_experiment(
            cell_type='vanilla',
            use_attention=False,
            experiment_name='Vanilla RNN (Quick Test)'
        )
        
        print("\n" + "="*80)
        print("QUICK PIPELINE TEST COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nTest Results:")
        print(f"  CTC Loss: {results.get('ctc_loss', 'N/A'):.4f}")
        print(f"  Accuracy: {results.get('accuracy', 'N/A'):.4f}")
        print(f"  CER: {results.get('cer', 'N/A'):.4f}")
        print(f"  WER: {results.get('wer', 'N/A'):.4f}")
        print("\nThe pipeline is working correctly!")
        print("You can now run asrking2.py for full experiments with complete dataset.")
        print("="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        print("Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n{'!'*80}")
        print("ERROR: Experiment failed!")
        print(f"{'!'*80}")
        print(f"\nError details: {e}")
        print("\nPlease check:")
        print("  1. All required dependencies are installed")
        print("  2. Dataset was downloaded successfully")
        print("  3. Sufficient memory is available")
        print("  4. CUDA is available if using GPU")
        print(f"\n{'!'*80}\n")
        
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
