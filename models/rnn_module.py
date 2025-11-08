"""
RNN Module Factory

Provides a generalizable RNN class that supports multiple cell types
(Vanilla RNN, LSTM, GRU) with a consistent interface.
"""

import torch
import torch.nn as nn


class RNNModule(nn.Module):
    """
    Generalizable RNN class that supports multiple cell types.
    
    This module provides a unified interface for creating different RNN cell types
    (vanilla RNN, LSTM, GRU) with support for multi-layer stacking and dropout.
    """
    
    def __init__(self, cell_type: str, input_size: int, hidden_size: int, 
                 num_layers: int = 1, dropout: float = 0.0, batch_first: bool = True):
        """
        Initialize the RNN module.
        
        Args:
            cell_type: One of ['vanilla', 'lstm', 'gru']
            input_size: Dimension of input features
            hidden_size: Dimension of hidden state
            num_layers: Number of stacked RNN layers (default: 1)
            dropout: Dropout probability between layers (default: 0.0)
            batch_first: If True, input/output tensors are (batch, seq, feature) (default: True)
        
        Raises:
            ValueError: If cell_type is not one of the supported types
        """
        super(RNNModule, self).__init__()
        
        self.cell_type = cell_type.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first
        
        # Create the appropriate RNN cell type
        if self.cell_type == 'vanilla':
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=batch_first
            )
        elif self.cell_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=batch_first
            )
        elif self.cell_type == 'gru':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=batch_first
            )
        else:
            raise ValueError(
                f"Unsupported cell_type: {cell_type}. "
                f"Must be one of ['vanilla', 'lstm', 'gru']"
            )
    
    def forward(self, x, hidden=None):
        """
        Forward pass through the RNN.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size) if batch_first=True,
               or (seq_len, batch, input_size) if batch_first=False
            hidden: Optional initial hidden state. If None, defaults to zeros.
                   - For vanilla RNN and GRU: (num_layers, batch, hidden_size)
                   - For LSTM: tuple of (h_0, c_0), each (num_layers, batch, hidden_size)
        
        Returns:
            output: Output tensor of shape (batch, seq_len, hidden_size) if batch_first=True,
                   or (seq_len, batch, hidden_size) if batch_first=False
            hidden: Final hidden state
                   - For vanilla RNN and GRU: (num_layers, batch, hidden_size)
                   - For LSTM: tuple of (h_n, c_n), each (num_layers, batch, hidden_size)
        """
        output, hidden = self.rnn(x, hidden)
        return output, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device = None):
        """
        Initialize hidden state with zeros.
        
        Args:
            batch_size: Batch size
            device: Device to create tensors on (default: None, uses CPU)
        
        Returns:
            hidden: Initialized hidden state
                   - For vanilla RNN and GRU: (num_layers, batch, hidden_size)
                   - For LSTM: tuple of (h_0, c_0), each (num_layers, batch, hidden_size)
        """
        if device is None:
            device = torch.device('cpu')
        
        if self.cell_type == 'lstm':
            # LSTM requires both hidden state and cell state
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            return (h_0, c_0)
        else:
            # Vanilla RNN and GRU only need hidden state
            return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
    
    def get_cell_type(self):
        """
        Get the cell type of this RNN module.
        
        Returns:
            str: The cell type ('vanilla', 'lstm', or 'gru')
        """
        return self.cell_type
