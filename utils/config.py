"""
Configuration management for ASR RNN System
"""

# Hyperparameters for all experiments
config = {
    # Model architecture
    'hidden_size': 128,  # Reduced from 256 to save memory
    'num_layers': 1,     # Reduced from 2 to save memory
    'dropout': 0.3,
    
    # Audio preprocessing
    'n_mfcc': 40,
    'sample_rate': 16000,
    
    # Training
    'batch_size': 8,     # Reduced from 32 to save memory
    'learning_rate': 0.001,
    'num_epochs': 50,
    'gradient_clip': 5.0,
    
    # Data
    'subset_size': None,  # None for full dataset, set to int for subset
    
    # Logging
    'log_interval': 10,  # Log every N batches
    'save_interval': 5,  # Save checkpoint every N epochs
}

# Experiment definitions
experiments = [
    {'name': 'Vanilla RNN', 'cell_type': 'vanilla', 'use_attention': False},
    {'name': 'Vanilla RNN with attention', 'cell_type': 'vanilla', 'use_attention': True},
    {'name': 'LSTM', 'cell_type': 'lstm', 'use_attention': False},
    {'name': 'LSTM with attention', 'cell_type': 'lstm', 'use_attention': True},
    {'name': 'GRU', 'cell_type': 'gru', 'use_attention': False},
    {'name': 'GRU with attention', 'cell_type': 'gru', 'use_attention': True},
]
