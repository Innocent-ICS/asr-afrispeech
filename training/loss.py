"""
CTC Loss Wrapper

Implements Connectionist Temporal Classification (CTC) loss computation
for sequence-to-sequence ASR training with variable-length sequences.

Requirements addressed:
- 5.1: Use CTC loss as the cost function for all models
- 5.2: Apply CTC loss to model predictions and target transcriptions during training
- 5.3: Optimize model parameters to minimize CTC loss
"""

import torch
import torch.nn as nn
from typing import Tuple


class CTCLoss(nn.Module):
    """
    Wrapper for PyTorch's CTCLoss with proper handling of variable-length sequences.
    
    CTC loss allows training sequence-to-sequence models without requiring
    frame-level alignment between input and output sequences. It handles
    variable-length inputs and targets by marginalizing over all possible
    alignments.
    """
    
    def __init__(self, blank_idx: int = 0, reduction: str = 'mean', 
                 zero_infinity: bool = True):
        """
        Initialize CTC loss wrapper.
        
        Args:
            blank_idx: Index of the blank token in vocabulary (default: 0)
            reduction: Specifies the reduction to apply to the output:
                      'none' | 'mean' | 'sum'. Default: 'mean'
            zero_infinity: Whether to zero infinite losses and gradients.
                          Infinite losses can occur when input is too short
                          to produce target. Default: True
        """
        super(CTCLoss, self).__init__()
        
        self.blank_idx = blank_idx
        self.reduction = reduction
        
        # Initialize PyTorch's CTCLoss
        # log_softmax will be applied to logits before passing to CTCLoss
        self.ctc_loss = nn.CTCLoss(
            blank=blank_idx,
            reduction=reduction,
            zero_infinity=zero_infinity
        )
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute CTC loss for a batch of predictions and targets.
        
        Args:
            logits: Model output logits of shape (batch, max_time, vocab_size)
                   These are raw unnormalized scores from the model
            targets: Target sequences of shape (batch, max_target_len)
                    Contains character indices (should not include blank tokens)
            input_lengths: Actual lengths of input sequences (before padding)
                          Shape: (batch,) with values <= max_time
            target_lengths: Actual lengths of target sequences (before padding)
                           Shape: (batch,) with values <= max_target_len
        
        Returns:
            loss: Scalar tensor containing the CTC loss value
        
        Note:
            PyTorch's CTCLoss expects inputs in the format:
            - log_probs: (T, N, C) where T=time, N=batch, C=classes
            - targets: (N, S) or (sum(target_lengths),) where S=max target length
            - input_lengths: (N,)
            - target_lengths: (N,)
        """
        # Validate input shapes
        batch_size, max_time, vocab_size = logits.shape
        
        if targets.dim() != 2:
            raise ValueError(f"Expected targets to be 2D (batch, max_target_len), "
                           f"got shape {targets.shape}")
        
        if input_lengths.shape[0] != batch_size:
            raise ValueError(f"input_lengths batch size {input_lengths.shape[0]} "
                           f"doesn't match logits batch size {batch_size}")
        
        if target_lengths.shape[0] != batch_size:
            raise ValueError(f"target_lengths batch size {target_lengths.shape[0]} "
                           f"doesn't match logits batch size {batch_size}")
        
        # Apply log_softmax to convert logits to log probabilities
        # Shape: (batch, max_time, vocab_size)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Transpose to (max_time, batch, vocab_size) as required by CTCLoss
        log_probs = log_probs.transpose(0, 1)
        
        # Flatten targets by removing padding
        # CTCLoss can accept either (N, S) or flattened (sum(target_lengths),)
        # We'll use the (N, S) format which is simpler
        
        # Ensure input_lengths and target_lengths are on CPU for CTCLoss
        # (PyTorch CTCLoss requires length tensors on CPU)
        input_lengths_cpu = input_lengths.cpu()
        target_lengths_cpu = target_lengths.cpu()
        
        # Compute CTC loss
        try:
            loss = self.ctc_loss(
                log_probs,
                targets,
                input_lengths_cpu,
                target_lengths_cpu
            )
        except RuntimeError as e:
            # Provide helpful error message if CTC loss computation fails
            print(f"CTC Loss computation failed:")
            print(f"  log_probs shape: {log_probs.shape}")
            print(f"  targets shape: {targets.shape}")
            print(f"  input_lengths: {input_lengths_cpu}")
            print(f"  target_lengths: {target_lengths_cpu}")
            print(f"  Error: {str(e)}")
            raise
        
        return loss
    
    def compute_loss_with_logits(self, logits: torch.Tensor, targets: torch.Tensor,
                                 input_lengths: torch.Tensor, 
                                 target_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute CTC loss and also return log probabilities for potential reuse.
        
        This method is useful when you want both the loss and the log probabilities
        (e.g., for decoding or analysis).
        
        Args:
            logits: Model output logits of shape (batch, max_time, vocab_size)
            targets: Target sequences of shape (batch, max_target_len)
            input_lengths: Actual lengths of input sequences, shape (batch,)
            target_lengths: Actual lengths of target sequences, shape (batch,)
        
        Returns:
            loss: Scalar tensor containing the CTC loss value
            log_probs: Log probabilities of shape (batch, max_time, vocab_size)
        """
        # Apply log_softmax to convert logits to log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Transpose for CTCLoss
        log_probs_transposed = log_probs.transpose(0, 1)
        
        # Ensure lengths are on CPU
        input_lengths_cpu = input_lengths.cpu()
        target_lengths_cpu = target_lengths.cpu()
        
        # Compute CTC loss
        loss = self.ctc_loss(
            log_probs_transposed,
            targets,
            input_lengths_cpu,
            target_lengths_cpu
        )
        
        return loss, log_probs


def create_ctc_loss(blank_idx: int = 0, reduction: str = 'mean') -> CTCLoss:
    """
    Factory function to create a CTC loss instance.
    
    Args:
        blank_idx: Index of the blank token in vocabulary
        reduction: Reduction method ('none', 'mean', or 'sum')
    
    Returns:
        CTCLoss instance
    """
    return CTCLoss(blank_idx=blank_idx, reduction=reduction)


def validate_ctc_inputs(logits: torch.Tensor, targets: torch.Tensor,
                       input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> bool:
    """
    Validate that inputs to CTC loss have correct shapes and values.
    
    Args:
        logits: Model output logits
        targets: Target sequences
        input_lengths: Input sequence lengths
        target_lengths: Target sequence lengths
    
    Returns:
        True if inputs are valid, raises ValueError otherwise
    
    Raises:
        ValueError: If any validation check fails
    """
    # Check dimensions
    if logits.dim() != 3:
        raise ValueError(f"logits must be 3D (batch, time, vocab), got shape {logits.shape}")
    
    if targets.dim() != 2:
        raise ValueError(f"targets must be 2D (batch, max_target_len), got shape {targets.shape}")
    
    if input_lengths.dim() != 1:
        raise ValueError(f"input_lengths must be 1D, got shape {input_lengths.shape}")
    
    if target_lengths.dim() != 1:
        raise ValueError(f"target_lengths must be 1D, got shape {target_lengths.shape}")
    
    # Check batch sizes match
    batch_size = logits.shape[0]
    if targets.shape[0] != batch_size:
        raise ValueError(f"Batch size mismatch: logits {batch_size}, targets {targets.shape[0]}")
    
    if input_lengths.shape[0] != batch_size:
        raise ValueError(f"Batch size mismatch: logits {batch_size}, "
                        f"input_lengths {input_lengths.shape[0]}")
    
    if target_lengths.shape[0] != batch_size:
        raise ValueError(f"Batch size mismatch: logits {batch_size}, "
                        f"target_lengths {target_lengths.shape[0]}")
    
    # Check that lengths are within bounds
    max_time = logits.shape[1]
    if (input_lengths > max_time).any():
        raise ValueError(f"Some input_lengths exceed max_time {max_time}")
    
    max_target_len = targets.shape[1]
    if (target_lengths > max_target_len).any():
        raise ValueError(f"Some target_lengths exceed max_target_len {max_target_len}")
    
    # Check that lengths are positive
    if (input_lengths <= 0).any():
        raise ValueError("All input_lengths must be positive")
    
    if (target_lengths <= 0).any():
        raise ValueError("All target_lengths must be positive")
    
    return True
