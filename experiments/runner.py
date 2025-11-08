"""
Experiment Runner for ASR RNN System

Orchestrates the execution of all six experiments comparing different RNN architectures
(Vanilla RNN, LSTM, GRU) with and without Bahdanau attention mechanism.

Requirements addressed:
- 1.5: Execute experiments in specified order
- 7.1: Name first experiment "Vanilla RNN"
- 7.2: Name second experiment "Vanilla RNN with attention"
- 7.3: Name third experiment "LSTM"
- 7.4: Name fourth experiment "LSTM with attention"
- 7.5: Name fifth experiment "GRU"
- 7.6: Name sixth experiment "GRU with attention"
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
import os

from models.encoder import Encoder
from models.decoder import Decoder
from training.trainer import Trainer
from training.loss import CTCLoss
from evaluation.evaluator import Evaluator
from asr_logging.logger import ExperimentLogger
from data.dataset import AfriSpeechDataset
from data.preprocessor import AudioPreprocessor
from utils.vocab import Vocabulary, build_vocabulary_from_dataset


class ASRModel(nn.Module):
    """
    Wrapper class for encoder-decoder ASR model.
    
    Combines encoder and decoder into a single model for easier management.
    """
    
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        """
        Initialize ASR model.
        
        Args:
            encoder: Encoder module
            decoder: Decoder module
        """
        super(ASRModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, audio_features, audio_lengths, target_lengths):
        """
        Forward pass through encoder and decoder.
        
        Args:
            audio_features: Audio features tensor
            audio_lengths: Actual lengths of audio sequences
            target_lengths: Actual lengths of target sequences
        
        Returns:
            logits: Output logits from decoder
        """
        encoder_outputs, encoder_hidden = self.encoder(audio_features, audio_lengths)
        logits = self.decoder(encoder_outputs, encoder_hidden, target_lengths)
        return logits


class ExperimentRunner:
    """
    Orchestrates the execution of all six ASR experiments.
    
    Manages dataset loading, model creation, training, and evaluation for
    each experiment configuration.
    """
    
    def __init__(self, config: Dict, wandb_token: Optional[str] = None):
        """
        Initialize the experiment runner.
        
        Args:
            config: Configuration dictionary with hyperparameters
            wandb_token: Optional WandB API token for logging
        """
        self.config = config
        self.wandb_token = wandb_token
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize datasets and vocabulary
        self._setup_data()
    
    def _setup_data(self):
        """
        Set up datasets, data loaders, and vocabulary.
        """
        print("\n" + "="*80)
        print("Setting up datasets and vocabulary")
        print("="*80)
        
        # Load datasets
        subset_size = self.config.get('subset_size', None)
        
        print(f"\nLoading datasets (subset_size: {subset_size})...")
        
        try:
            self.train_dataset = AfriSpeechDataset(split='train', subset_size=subset_size)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load training dataset.\n"
                f"Error: {str(e)}\n"
                f"Please check your internet connection and ensure the dataset can be downloaded."
            )
        
        try:
            self.dev_dataset = AfriSpeechDataset(split='dev', subset_size=subset_size)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load development dataset.\n"
                f"Error: {str(e)}\n"
                f"Please check your internet connection and ensure the dataset can be downloaded."
            )
        
        try:
            self.test_dataset = AfriSpeechDataset(split='test', subset_size=subset_size)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load test dataset.\n"
                f"Error: {str(e)}\n"
                f"Please check your internet connection and ensure the dataset can be downloaded."
            )
        
        # Validate that splits are non-empty
        if len(self.train_dataset) == 0:
            raise ValueError(
                "Training dataset is empty. Cannot proceed with training.\n"
                "Please check the dataset download and extraction."
            )
        
        if len(self.dev_dataset) == 0:
            raise ValueError(
                "Development dataset is empty. Cannot proceed with validation.\n"
                "Please check the dataset download and extraction."
            )
        
        if len(self.test_dataset) == 0:
            raise ValueError(
                "Test dataset is empty. Cannot proceed with evaluation.\n"
                "Please check the dataset download and extraction."
            )
        
        print(f"✓ Loaded {len(self.train_dataset)} training samples")
        print(f"✓ Loaded {len(self.dev_dataset)} development samples")
        print(f"✓ Loaded {len(self.test_dataset)} test samples")
        
        # Build vocabulary from training set
        print("\nBuilding vocabulary from training set...")
        try:
            self.vocab = build_vocabulary_from_dataset(self.train_dataset)
            print(f"Vocabulary size: {len(self.vocab)}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to build vocabulary from training dataset.\n"
                f"Error: {str(e)}"
            )
        
        # Create audio preprocessor
        try:
            self.preprocessor = AudioPreprocessor(
                feature_type='mfcc',
                n_features=self.config['n_mfcc'],
                sample_rate=self.config['sample_rate']
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to create audio preprocessor.\n"
                f"Error: {str(e)}"
            )
        
        # Create data loaders
        batch_size = self.config['batch_size']
        print(f"\nCreating data loaders (batch_size: {batch_size})...")
        
        try:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=self.preprocessor.collate_fn,
                num_workers=0  # Set to 0 to avoid multiprocessing issues
            )
            
            self.dev_loader = DataLoader(
                self.dev_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self.preprocessor.collate_fn,
                num_workers=0
            )
            
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self._test_collate_fn,
                num_workers=0
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to create data loaders.\n"
                f"Error: {str(e)}"
            )
        
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Dev batches: {len(self.dev_loader)}")
        print(f"Test batches: {len(self.test_loader)}")
        print("="*80 + "\n")
    
    def _test_collate_fn(self, batch):
        """
        Collate function for test set that returns dictionary format.
        """
        audio_features, transcriptions, audio_lengths, text_lengths = \
            self.preprocessor.collate_fn(batch)
        
        return {
            'audio_features': audio_features,
            'transcriptions': transcriptions,
            'audio_lengths': audio_lengths,
            'text_lengths': text_lengths
        }
    
    def run_experiment(self, cell_type: str, use_attention: bool, experiment_name: str):
        """
        Run a single experiment with specified configuration.
        
        Args:
            cell_type: Type of RNN cell ('vanilla', 'lstm', or 'gru')
            use_attention: Whether to use Bahdanau attention mechanism
            experiment_name: Name of the experiment for logging
        """
        print("\n" + "="*80)
        print(f"EXPERIMENT: {experiment_name}")
        print("="*80)
        print(f"Cell Type: {cell_type}")
        print(f"Attention: {'Enabled' if use_attention else 'Disabled'}")
        print("="*80 + "\n")
        
        logger = None
        
        try:
            # Create model
            print("Creating model...")
            encoder = Encoder(
                input_size=self.config['n_mfcc'],
                hidden_size=self.config['hidden_size'],
                num_layers=self.config['num_layers'],
                cell_type=cell_type,
                dropout=self.config['dropout']
            )
            
            decoder = Decoder(
                hidden_size=self.config['hidden_size'],
                vocab_size=len(self.vocab),
                num_layers=self.config['num_layers'],
                cell_type=cell_type,
                dropout=self.config['dropout'],
                use_attention=use_attention
            )
            
            model = ASRModel(encoder, decoder)
            
            # Move to device with error handling
            try:
                model.to(self.device)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    raise RuntimeError(
                        f"CUDA out of memory when loading model to device.\n"
                        f"Model size may be too large for available GPU memory.\n"
                        f"Suggestions:\n"
                        f"  - Reduce hidden_size (current: {self.config['hidden_size']})\n"
                        f"  - Reduce num_layers (current: {self.config['num_layers']})\n"
                        f"  - Use CPU instead (set device to 'cpu')\n"
                        f"Original error: {str(e)}"
                    )
                else:
                    raise
            
            # Create optimizer
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config['learning_rate']
            )
            
            # Create CTC loss
            ctc_loss = CTCLoss(blank_idx=self.vocab.blank_idx, reduction='mean')
            
            # Create logger
            logger = ExperimentLogger(
                experiment_name=experiment_name,
                wandb_token=self.wandb_token,
                log_dir='logs',
                project_name='asr-rnn-experiments'
            )
            
            # Create trainer
            trainer = Trainer(
                encoder=encoder,
                decoder=decoder,
                train_loader=self.train_loader,
                val_loader=self.dev_loader,
                optimizer=optimizer,
                ctc_loss=ctc_loss,
                logger=logger,
                vocab=self.vocab,
                device=self.device,
                experiment_name=experiment_name,
                gradient_clip=self.config['gradient_clip']
            )
            
            # Train model
            print("\nStarting training...")
            num_epochs = self.config['num_epochs']
            trainer.train(num_epochs=num_epochs)
            
            # Evaluate on test set
            print("\nEvaluating on test set...")
            evaluator = Evaluator(
                model=model,
                test_loader=self.test_loader,
                vocab=self.vocab,
                device=self.device,
                ctc_loss_fn=ctc_loss
            )
            
            results = evaluator.evaluate(num_samples=5)
            
            # Print results
            evaluator.print_results(results)
            
            # Log test results
            logger.log_test_results(results)
            
            # Log sample transcriptions
            if 'samples' in results:
                logger.log_sample_transcriptions(results['samples'])
            
            print(f"\n{'='*80}")
            print(f"EXPERIMENT COMPLETED: {experiment_name}")
            print(f"{'='*80}\n")
            
            return results
            
        except RuntimeError as e:
            error_msg = str(e)
            print(f"\n{'!'*80}")
            print(f"EXPERIMENT FAILED: {experiment_name}")
            print(f"{'!'*80}")
            print(f"Error: {error_msg}")
            print(f"{'!'*80}\n")
            return {'error': error_msg}
            
        except Exception as e:
            print(f"\n{'!'*80}")
            print(f"UNEXPECTED ERROR in experiment: {experiment_name}")
            print(f"{'!'*80}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error: {str(e)}")
            print(f"{'!'*80}\n")
            return {'error': f"{type(e).__name__}: {str(e)}"}
            
        finally:
            # Always close logger if it was created
            if logger is not None:
                try:
                    logger.close()
                except Exception as e:
                    print(f"Warning: Failed to close logger: {str(e)}")
    
    def run_all_experiments(self):
        """
        Run all six experiments in the specified order.
        
        Executes experiments in order:
        1. Vanilla RNN
        2. Vanilla RNN with attention
        3. LSTM
        4. LSTM with attention
        5. GRU
        6. GRU with attention
        """
        print("\n" + "="*80)
        print("STARTING ALL EXPERIMENTS")
        print("="*80)
        print("Will run 6 experiments:")
        print("  1. Vanilla RNN")
        print("  2. Vanilla RNN with attention")
        print("  3. LSTM")
        print("  4. LSTM with attention")
        print("  5. GRU")
        print("  6. GRU with attention")
        print("="*80 + "\n")
        
        # Define experiments in specified order
        experiments = [
            {
                'name': 'Vanilla RNN',
                'cell_type': 'vanilla',
                'use_attention': False
            },
            {
                'name': 'Vanilla RNN with attention',
                'cell_type': 'vanilla',
                'use_attention': True
            },
            {
                'name': 'LSTM',
                'cell_type': 'lstm',
                'use_attention': False
            },
            {
                'name': 'LSTM with attention',
                'cell_type': 'lstm',
                'use_attention': True
            },
            {
                'name': 'GRU',
                'cell_type': 'gru',
                'use_attention': False
            },
            {
                'name': 'GRU with attention',
                'cell_type': 'gru',
                'use_attention': True
            }
        ]
        
        # Store results from all experiments
        all_results = {}
        
        # Run each experiment
        for i, exp_config in enumerate(experiments, 1):
            print(f"\n{'#'*80}")
            print(f"# EXPERIMENT {i}/6")
            print(f"{'#'*80}\n")
            
            try:
                results = self.run_experiment(
                    cell_type=exp_config['cell_type'],
                    use_attention=exp_config['use_attention'],
                    experiment_name=exp_config['name']
                )
                all_results[exp_config['name']] = results
            except Exception as e:
                print(f"\n{'!'*80}")
                print(f"ERROR in experiment '{exp_config['name']}':")
                print(f"  {str(e)}")
                print(f"{'!'*80}\n")
                print("Continuing with next experiment...\n")
                all_results[exp_config['name']] = {'error': str(e)}
        
        # Print summary of all experiments
        self._print_summary(all_results)
        
        return all_results
    
    def _print_summary(self, all_results: Dict):
        """
        Print summary of all experiment results.
        
        Args:
            all_results: Dictionary mapping experiment names to results
        """
        print("\n" + "="*80)
        print("SUMMARY OF ALL EXPERIMENTS")
        print("="*80)
        
        # Print table header
        print(f"\n{'Experiment':<30} {'CTC Loss':<12} {'Accuracy':<12} {'CER':<12} {'WER':<12}")
        print("-"*80)
        
        # Print results for each experiment
        for exp_name, results in all_results.items():
            if 'error' in results:
                print(f"{exp_name:<30} {'ERROR':<12} {'-':<12} {'-':<12} {'-':<12}")
            else:
                ctc_loss = results.get('ctc_loss', float('inf'))
                accuracy = results.get('accuracy', 0.0)
                cer = results.get('cer', 1.0)
                wer = results.get('wer', 1.0)
                
                print(f"{exp_name:<30} {ctc_loss:<12.4f} {accuracy:<12.4f} {cer:<12.4f} {wer:<12.4f}")
        
        print("="*80 + "\n")
