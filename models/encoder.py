"""
Encoder Component

Implements the encoder that processes variable-length audio features into
fixed-size representations using RNN modules with packed sequences.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    """
    Encoder that processes audio features into latent representations.
    
    The encoder uses an RNN module (vanilla RNN, LSTM, or GRU) to encode
    variable-length audio feature sequences. It uses packed sequences for
    efficient processing of variable-length inputs.
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 cell_type: str = 'lstm', dropout: float = 0.0):
        """
        Initialize the encoder.
        
        Args:
            input_size: Dimension of input audio features (e.g., number of MFCC coefficients)
            hidden_size: Dimension of RNN hidden state
            num_layers: Number of stacked RNN layers
            cell_type: Type of RNN cell ('vanilla', 'lstm', or 'gru')
            dropout: Dropout probability between RNN layers
        """
        super(Encoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        
        # Import RNNModule here to avoid circular imports
        from models.rnn_module import RNNModule
        
        # Create RNN module with specified cell type
        self.rnn = RNNModule(
            cell_type=cell_type,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
    
    def forward(self, audio_features, lengths):
        """
        Encode audio features into latent representations.
        
        Args:
            audio_features: Padded audio features of shape (batch, max_time, input_size)
            lengths: Actual lengths of each sequence in the batch (before padding)
                    as a tensor of shape (batch,) or list
        
        Returns:
            encoder_outputs: Full sequence of encoder outputs of shape (batch, max_time, hidden_size)
            encoder_hidden: Final hidden state
                          - For vanilla RNN and GRU: (num_layers, batch, hidden_size)
                          - For LSTM: tuple of (h_n, c_n), each (num_layers, batch, hidden_size)
        """
        batch_size = audio_features.size(0)
        
        # Convert lengths to CPU tensor if needed (required by pack_padded_sequence)
        if isinstance(lengths, list):
            lengths = torch.tensor(lengths, dtype=torch.long)
        if lengths.is_cuda:
            lengths = lengths.cpu()
        
        # Pack the padded sequences for efficient RNN processing
        # This removes padding and allows the RNN to process only actual data
        packed_input = pack_padded_sequence(
            audio_features,
            lengths,
            batch_first=True,
            enforce_sorted=False  # Allow unsorted lengths
        )
        
        # Process through RNN
        packed_output, encoder_hidden = self.rnn(packed_input)
        
        # Unpack the sequences back to padded format
        # encoder_outputs: (batch, max_time, hidden_size)
        encoder_outputs, _ = pad_packed_sequence(
            packed_output,
            batch_first=True
        )
        
        return encoder_outputs, encoder_hidden
    
    def get_cell_type(self):
        """
        Get the RNN cell type used by this encoder.
        
        Returns:
            str: The cell type ('vanilla', 'lstm', or 'gru')
        """
        return self.rnn.get_cell_type()

