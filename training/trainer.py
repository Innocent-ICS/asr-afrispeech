"""
Training Pipeline for ASR RNN System

Implements the training loop with forward pass, loss computation, backpropagation,
validation, and logging integration.

Requirements addressed:
- 4.3: Record CTC loss for training set during training
- 4.4: Record CTC loss for validation set during training
- 4.5: Generate visual plots of training and validation losses
- 5.2: Apply CTC loss to model predictions and target transcriptions during training
- 5.3: Optimize model parameters to minimize CTC loss
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict
from tqdm import tqdm


class Trainer:
    """
    Orchestrates the training process for ASR models.
    
    Handles training loop, validation, gradient clipping, and logging integration
    for encoder-decoder ASR models with CTC loss.
    """
    
    def __init__(self, encoder: nn.Module, decoder: nn.Module, 
                 train_loader: DataLoader, val_loader: DataLoader,
                 optimizer: torch.optim.Optimizer, ctc_loss: nn.Module,
                 logger, vocab, device: torch.device, 
                 experiment_name: str, gradient_clip: float = 5.0):
        """
        Initialize the trainer.
        
        Args:
            encoder: Encoder model
            decoder: Decoder model
            train_loader: DataLoader for training set
            val_loader: DataLoader for validation set
            optimizer: PyTorch optimizer
            ctc_loss: CTC loss module
            logger: Instance of ExperimentLogger
            vocab: Vocabulary object for encoding/decoding
            device: torch device (cpu/cuda)
            experiment_name: Name of current experiment
            gradient_clip: Maximum gradient norm for clipping (default: 5.0)
        """
        self.encoder = encoder
        self.decoder = decoder
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.ctc_loss = ctc_loss
        self.logger = logger
        self.vocab = vocab
        self.device = device
        self.experiment_name = experiment_name
        self.gradient_clip = gradient_clip
        
        # Move models to device
        self.encoder.to(device)
        self.decoder.to(device)
        
        # Track training history
        self.train_losses = []
        self.val_losses = []
        self.current_epoch = 0
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Performs forward pass, loss computation, and backpropagation for all
        batches in the training set.
        
        Returns:
            avg_train_loss: Average CTC loss on training set for this epoch
        """
        self.encoder.train()
        self.decoder.train()
        
        total_loss = 0.0
        num_batches = 0
        
        # Progress bar for training
        pbar = tqdm(self.train_loader, desc=f"[{self.experiment_name}] Training Epoch {self.current_epoch + 1}")
        
        for batch_idx, (audio_features, transcriptions, audio_lengths, text_lengths) in enumerate(pbar):
            # Move data to device
            audio_features = audio_features.to(self.device)
            audio_lengths = audio_lengths.to(self.device)
            
            # Encode text transcriptions to indices
            target_indices = []
            for text in transcriptions:
                indices = self.vocab.encode(text, add_sos=False, add_eos=False)
                target_indices.append(indices)
            
            # Pad target sequences
            max_target_len = max(len(seq) for seq in target_indices) if target_indices else 1
            padded_targets = []
            for seq in target_indices:
                padded_seq = seq + [self.vocab.pad_idx] * (max_target_len - len(seq))
                padded_targets.append(padded_seq)
            
            targets = torch.tensor(padded_targets, dtype=torch.long).to(self.device)
            target_lengths = torch.tensor([len(seq) for seq in target_indices], 
                                         dtype=torch.long).to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass through encoder
            encoder_outputs, encoder_hidden = self.encoder(audio_features, audio_lengths)
            
            # Forward pass through decoder
            # For CTC, we need logits for each time step in the encoder output
            # The decoder generates logits of shape (batch, max_time, vocab_size)
            logits = self.decoder(encoder_outputs, encoder_hidden, audio_lengths)
            
            # Compute CTC loss
            # logits: (batch, max_time, vocab_size)
            # targets: (batch, max_target_len)
            # audio_lengths: actual lengths of encoder outputs
            # target_lengths: actual lengths of target sequences
            try:
                loss = self.ctc_loss(logits, targets, audio_lengths, target_lengths)
            except RuntimeError as e:
                print(f"\nError computing CTC loss in batch {batch_idx}:")
                print(f"  logits shape: {logits.shape}")
                print(f"  targets shape: {targets.shape}")
                print(f"  audio_lengths: {audio_lengths}")
                print(f"  target_lengths: {target_lengths}")
                print(f"  Error: {str(e)}")
                continue
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"\nWarning: NaN loss detected in batch {batch_idx}, skipping batch")
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.gradient_clip)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.gradient_clip)
            
            # Optimizer step
            self.optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate average loss
        avg_train_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        return avg_train_loss
    
    def validate(self) -> float:
        """
        Validate on validation set.
        
        Evaluates the model on the validation set without updating parameters.
        
        Returns:
            avg_val_loss: Average CTC loss on validation set
        """
        self.encoder.eval()
        self.decoder.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"[{self.experiment_name}] Validation")
            
            for batch_idx, (audio_features, transcriptions, audio_lengths, text_lengths) in enumerate(pbar):
                # Move data to device
                audio_features = audio_features.to(self.device)
                audio_lengths = audio_lengths.to(self.device)
                
                # Encode text transcriptions to indices
                target_indices = []
                for text in transcriptions:
                    indices = self.vocab.encode(text, add_sos=False, add_eos=False)
                    target_indices.append(indices)
                
                # Pad target sequences
                max_target_len = max(len(seq) for seq in target_indices) if target_indices else 1
                padded_targets = []
                for seq in target_indices:
                    padded_seq = seq + [self.vocab.pad_idx] * (max_target_len - len(seq))
                    padded_targets.append(padded_seq)
                
                targets = torch.tensor(padded_targets, dtype=torch.long).to(self.device)
                target_lengths = torch.tensor([len(seq) for seq in target_indices], 
                                             dtype=torch.long).to(self.device)
                
                # Forward pass
                encoder_outputs, encoder_hidden = self.encoder(audio_features, audio_lengths)
                logits = self.decoder(encoder_outputs, encoder_hidden, audio_lengths)
                
                # Compute CTC loss
                try:
                    loss = self.ctc_loss(logits, targets, audio_lengths, target_lengths)
                except RuntimeError as e:
                    print(f"\nError computing CTC loss in validation batch {batch_idx}:")
                    print(f"  Error: {str(e)}")
                    continue
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"\nWarning: NaN loss detected in validation batch {batch_idx}, skipping batch")
                    continue
                
                # Accumulate loss
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate average loss
        avg_val_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        return avg_val_loss
    
    def train(self, num_epochs: int):
        """
        Full training loop for specified number of epochs.
        
        Trains the model, validates after each epoch, logs metrics, and generates
        visual plots of training progress.
        
        Args:
            num_epochs: Number of epochs to train
        """
        print(f"\n{'='*80}")
        print(f"Starting training: {self.experiment_name}")
        print(f"{'='*80}\n")
        
        best_val_loss = float('inf')
        nan_loss_count = 0
        max_nan_tolerance = 3  # Stop if we get 3 consecutive NaN losses
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"{'-'*80}")
            
            try:
                # Train for one epoch
                train_loss = self.train_epoch()
                self.train_losses.append(train_loss)
                
                # Check for NaN loss
                if train_loss == float('inf') or torch.isnan(torch.tensor(train_loss)):
                    nan_loss_count += 1
                    print(f"\n⚠ WARNING: NaN or infinite training loss detected (count: {nan_loss_count}/{max_nan_tolerance})")
                    
                    if nan_loss_count >= max_nan_tolerance:
                        raise RuntimeError(
                            f"Training stopped: {max_nan_tolerance} consecutive NaN/infinite losses detected.\n"
                            f"This may indicate:\n"
                            f"  - Learning rate is too high (current: {self.optimizer.param_groups[0]['lr']})\n"
                            f"  - Gradient explosion (try reducing learning rate or increasing gradient clipping)\n"
                            f"  - Numerical instability in the model\n"
                            f"Experiment: {self.experiment_name}"
                        )
                else:
                    nan_loss_count = 0  # Reset counter on successful epoch
                
                # Validate
                val_loss = self.validate()
                self.val_losses.append(val_loss)
                
                # Check for NaN validation loss
                if val_loss == float('inf') or torch.isnan(torch.tensor(val_loss)):
                    print(f"\n⚠ WARNING: NaN or infinite validation loss detected")
                
                # Log metrics to WandB and TensorBoard
                metrics = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }
                self.logger.log_metrics(metrics, step=epoch + 1)
                
                # Print epoch summary
                print(f"\nEpoch {epoch + 1} Summary:")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss:   {val_loss:.4f}")
                
                # Track best validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"  ✓ New best validation loss!")
                
                # Generate and save loss plots
                if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                    self.logger.plot_losses(self.train_losses, self.val_losses)
                    
            except RuntimeError as e:
                error_msg = str(e).lower()
                if 'out of memory' in error_msg or 'cuda' in error_msg:
                    raise RuntimeError(
                        f"CUDA out of memory error during training.\n"
                        f"Suggestions:\n"
                        f"  - Reduce batch size (current: {self.train_loader.batch_size})\n"
                        f"  - Reduce model size (hidden_size, num_layers)\n"
                        f"  - Use gradient accumulation\n"
                        f"  - Use mixed precision training (torch.cuda.amp)\n"
                        f"Experiment: {self.experiment_name}\n"
                        f"Original error: {str(e)}"
                    )
                else:
                    # Re-raise other runtime errors
                    raise
        
        # Final summary
        print(f"\n{'='*80}")
        print(f"Training completed: {self.experiment_name}")
        if self.train_losses and self.val_losses:
            print(f"  Best Validation Loss: {best_val_loss:.4f}")
            print(f"  Final Train Loss: {self.train_losses[-1]:.4f}")
            print(f"  Final Val Loss: {self.val_losses[-1]:.4f}")
        print(f"{'='*80}\n")
        
        # Generate final loss plot
        if self.train_losses and self.val_losses:
            self.logger.plot_losses(self.train_losses, self.val_losses)
    
    def get_training_history(self) -> Dict[str, list]:
        """
        Get training history.
        
        Returns:
            Dictionary containing training and validation loss history
        """
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def save_checkpoint(self, filepath: str):
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'experiment_name': self.experiment_name
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to load checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        print(f"Checkpoint loaded from {filepath}")
