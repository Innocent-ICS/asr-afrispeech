"""
Bahdanau Attention Mechanism

Implements the Bahdanau (additive) attention mechanism for encoder-decoder models.
This allows the decoder to focus on relevant parts of the encoded input sequence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """
    Bahdanau attention mechanism (also known as additive attention).
    
    This attention mechanism computes a context vector as a weighted sum of encoder outputs,
    where the weights are determined by the compatibility between the decoder hidden state
    and each encoder output.
    
    Reference: Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate"
    """
    
    def __init__(self, hidden_size: int):
        """
        Initialize the Bahdanau attention mechanism.
        
        Args:
            hidden_size: Dimension of encoder and decoder hidden states
        """
        super(BahdanauAttention, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Linear layers for computing attention scores
        # W_a for encoder outputs
        self.W_a = nn.Linear(hidden_size, hidden_size, bias=False)
        # U_a for decoder hidden state
        self.U_a = nn.Linear(hidden_size, hidden_size, bias=False)
        # v_a for computing scalar attention scores
        self.v_a = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        Compute attention context vector and attention weights.
        
        Args:
            decoder_hidden: Current decoder hidden state of shape (batch, hidden_size)
            encoder_outputs: All encoder outputs of shape (batch, max_time, hidden_size)
            mask: Optional boolean mask of shape (batch, max_time) where True indicates
                  positions to ignore (e.g., padding positions)
        
        Returns:
            context: Weighted sum of encoder outputs of shape (batch, hidden_size)
            attention_weights: Attention distribution of shape (batch, max_time)
        """
        batch_size = encoder_outputs.size(0)
        max_time = encoder_outputs.size(1)
        
        # Expand decoder_hidden to match encoder_outputs time dimension
        # decoder_hidden: (batch, hidden_size) -> (batch, 1, hidden_size) -> (batch, max_time, hidden_size)
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).expand(batch_size, max_time, self.hidden_size)
        
        # Compute attention scores using additive attention
        # score = v_a^T * tanh(W_a * encoder_outputs + U_a * decoder_hidden)
        # Shape: (batch, max_time, hidden_size)
        scores = torch.tanh(self.W_a(encoder_outputs) + self.U_a(decoder_hidden_expanded))
        
        # Project to scalar scores
        # Shape: (batch, max_time, 1) -> (batch, max_time)
        attention_scores = self.v_a(scores).squeeze(-1)
        
        # Apply mask if provided (set masked positions to large negative value)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask, float('-inf'))
        
        # Compute attention weights using softmax
        # Shape: (batch, max_time)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Compute context vector as weighted sum of encoder outputs
        # attention_weights: (batch, max_time) -> (batch, max_time, 1)
        # encoder_outputs: (batch, max_time, hidden_size)
        # context: (batch, max_time, hidden_size) -> (batch, hidden_size)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attention_weights
