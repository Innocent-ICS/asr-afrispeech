"""
Decoder Component

Implements the decoder that generates text transcriptions from encoded audio
representations using RNN modules with optional Bahdanau attention mechanism.
"""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Decoder that generates text transcriptions from encoded representations.
    
    The decoder uses an RNN module (vanilla RNN, LSTM, or GRU) to generate
    output sequences. It can optionally use Bahdanau attention to focus on
    relevant parts of the encoder outputs.
    """
    
    def __init__(self, hidden_size: int, vocab_size: int, num_layers: int,
                 cell_type: str = 'lstm', dropout: float = 0.0,
                 use_attention: bool = False):
        """
        Initialize the decoder.
        
        Args:
            hidden_size: Dimension of RNN hidden state (must match encoder hidden size)
            vocab_size: Size of character/token vocabulary
            num_layers: Number of stacked RNN layers (should match encoder)
            cell_type: Type of RNN cell ('vanilla', 'lstm', or 'gru')
            dropout: Dropout probability between RNN layers
            use_attention: Whether to use Bahdanau attention mechanism
        """
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.use_attention = use_attention
        
        # Import modules here to avoid circular imports
        from models.rnn_module import RNNModule
        from models.attention import BahdanauAttention
        
        # Determine RNN input size based on whether attention is used
        # With attention: input is concatenation of embedding and context vector
        # Without attention: input is just the embedding
        rnn_input_size = hidden_size * 2 if use_attention else hidden_size
        
        # Create RNN module with specified cell type
        self.rnn = RNNModule(
            cell_type=cell_type,
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Create attention mechanism if enabled
        if use_attention:
            self.attention = BahdanauAttention(hidden_size=hidden_size)
        else:
            self.attention = None
        
        # Output projection layer to vocabulary size
        # Projects from hidden_size to vocab_size for character predictions
        self.output_projection = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, encoder_outputs, encoder_hidden, audio_lengths):
        """
        Generate output logits for CTC loss.
        
        For CTC, we need to generate logits at every time step of the encoder output.
        The decoder processes the encoder outputs and produces character probabilities
        at each time step.
        
        Args:
            encoder_outputs: Outputs from encoder of shape (batch, max_time, hidden_size)
            encoder_hidden: Final encoder hidden state
                          - For vanilla RNN and GRU: (num_layers, batch, hidden_size)
                          - For LSTM: tuple of (h_n, c_n), each (num_layers, batch, hidden_size)
            audio_lengths: Actual lengths of audio sequences (encoder output lengths)
        
        Returns:
            logits: Output logits of shape (batch, max_time, vocab_size)
        """
        batch_size = encoder_outputs.size(0)
        max_time = encoder_outputs.size(1)
        
        # For CTC, we can simply project encoder outputs to vocabulary size
        # This is more efficient and appropriate for CTC-based ASR
        if not self.use_attention:
            # Without attention: directly project encoder outputs
            # encoder_outputs: (batch, max_time, hidden_size)
            # logits: (batch, max_time, vocab_size)
            logits = self.output_projection(encoder_outputs)
            return logits
        
        # With attention: process through RNN with attention at each step
        decoder_hidden = encoder_hidden
        all_logits = []
        
        # Process each time step of encoder output
        for t in range(max_time):
            # Get encoder output for current time step
            current_input = encoder_outputs[:, t:t+1, :]  # (batch, 1, hidden_size)
            
            # Get decoder hidden state for attention
            # For LSTM, use the hidden state (not cell state)
            if self.cell_type == 'lstm':
                h_for_attention = decoder_hidden[0][-1]  # Last layer hidden state
            else:
                h_for_attention = decoder_hidden[-1]  # Last layer hidden state
            
            # Compute attention context over all encoder outputs
            context, attention_weights = self.attention(
                h_for_attention, 
                encoder_outputs
            )
            
            # Concatenate current input with context vector
            # current_input: (batch, 1, hidden_size)
            # context: (batch, hidden_size) -> (batch, 1, hidden_size)
            context = context.unsqueeze(1)
            rnn_input = torch.cat([current_input, context], dim=-1)
            
            # Process through RNN
            rnn_output, decoder_hidden = self.rnn(rnn_input, decoder_hidden)
            
            # Project to vocabulary size
            # rnn_output: (batch, 1, hidden_size) -> (batch, 1, vocab_size)
            logits = self.output_projection(rnn_output)
            
            all_logits.append(logits)
        
        # Concatenate all time steps
        # List of (batch, 1, vocab_size) -> (batch, max_time, vocab_size)
        output_logits = torch.cat(all_logits, dim=1)
        
        return output_logits
    
    def get_cell_type(self):
        """
        Get the RNN cell type used by this decoder.
        
        Returns:
            str: The cell type ('vanilla', 'lstm', or 'gru')
        """
        return self.rnn.get_cell_type()
