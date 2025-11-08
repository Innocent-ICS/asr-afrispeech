"""
Vocabulary builder for character-level ASR system.
Handles building vocabulary from dataset and encoding/decoding text.
"""

from typing import List, Dict, Set
import torch


class Vocabulary:
    """Character-level vocabulary for ASR system with special tokens."""
    
    # Special token constants
    BLANK_TOKEN = '<blank>'  # CTC blank token
    PAD_TOKEN = '<pad>'      # Padding token
    SOS_TOKEN = '<sos>'      # Start of sequence
    EOS_TOKEN = '<eos>'      # End of sequence
    
    def __init__(self):
        """Initialize vocabulary with special tokens."""
        self.char2idx: Dict[str, int] = {}
        self.idx2char: Dict[int, str] = {}
        
        # Add special tokens first to ensure consistent indices
        self._add_special_tokens()
        
    def _add_special_tokens(self):
        """Add special tokens to vocabulary."""
        special_tokens = [
            self.BLANK_TOKEN,
            self.PAD_TOKEN,
            self.SOS_TOKEN,
            self.EOS_TOKEN
        ]
        
        for token in special_tokens:
            idx = len(self.char2idx)
            self.char2idx[token] = idx
            self.idx2char[idx] = token
    
    def build_from_texts(self, texts: List[str]):
        """
        Build vocabulary from a list of text transcriptions.
        
        Args:
            texts: List of text transcriptions from dataset
        """
        # Collect all unique characters
        unique_chars: Set[str] = set()
        for text in texts:
            unique_chars.update(text)
        
        # Add characters to vocabulary (special tokens already added)
        for char in sorted(unique_chars):
            if char not in self.char2idx:
                idx = len(self.char2idx)
                self.char2idx[char] = idx
                self.idx2char[idx] = char
    
    def encode(self, text: str, add_sos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Encode text string to list of indices.
        
        Args:
            text: Text string to encode
            add_sos: Whether to add start-of-sequence token
            add_eos: Whether to add end-of-sequence token
            
        Returns:
            List of character indices
        """
        indices = []
        
        if add_sos:
            indices.append(self.char2idx[self.SOS_TOKEN])
        
        for char in text:
            if char in self.char2idx:
                indices.append(self.char2idx[char])
            else:
                # Handle unknown characters by skipping them
                # In production, might want to use an <unk> token
                continue
        
        if add_eos:
            indices.append(self.char2idx[self.EOS_TOKEN])
        
        return indices
    
    def decode(self, indices: List[int], remove_special: bool = True) -> str:
        """
        Decode list of indices to text string.
        
        Args:
            indices: List of character indices
            remove_special: Whether to remove special tokens from output
            
        Returns:
            Decoded text string
        """
        special_tokens = {
            self.char2idx[self.BLANK_TOKEN],
            self.char2idx[self.PAD_TOKEN],
            self.char2idx[self.SOS_TOKEN],
            self.char2idx[self.EOS_TOKEN]
        }
        
        chars = []
        for idx in indices:
            if idx in self.idx2char:
                if remove_special and idx in special_tokens:
                    continue
                chars.append(self.idx2char[idx])
        
        return ''.join(chars)
    
    def decode_ctc(self, indices: List[int]) -> str:
        """
        Decode CTC output by removing blanks and consecutive duplicates.
        
        Args:
            indices: List of character indices from CTC output
            
        Returns:
            Decoded text string
        """
        blank_idx = self.char2idx[self.BLANK_TOKEN]
        
        # Remove consecutive duplicates and blanks
        decoded_indices = []
        prev_idx = None
        
        for idx in indices:
            if idx == blank_idx:
                prev_idx = None
                continue
            
            if idx != prev_idx:
                decoded_indices.append(idx)
                prev_idx = idx
        
        return self.decode(decoded_indices, remove_special=True)
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.char2idx)
    
    @property
    def blank_idx(self) -> int:
        """Return index of blank token."""
        return self.char2idx[self.BLANK_TOKEN]
    
    @property
    def pad_idx(self) -> int:
        """Return index of padding token."""
        return self.char2idx[self.PAD_TOKEN]
    
    @property
    def sos_idx(self) -> int:
        """Return index of start-of-sequence token."""
        return self.char2idx[self.SOS_TOKEN]
    
    @property
    def eos_idx(self) -> int:
        """Return index of end-of-sequence token."""
        return self.char2idx[self.EOS_TOKEN]


def build_vocabulary_from_dataset(dataset) -> Vocabulary:
    """
    Build vocabulary from a dataset object.
    
    Args:
        dataset: Dataset object with transcriptions
        
    Returns:
        Vocabulary object built from dataset
    """
    vocab = Vocabulary()
    
    # Collect all transcriptions
    texts = []
    for i in range(len(dataset)):
        item = dataset[i]
        # Handle different dataset formats
        if isinstance(item, dict):
            text = item.get('transcription', item.get('text', ''))
        else:
            # Assume tuple format (audio, transcription, ...)
            text = item[1] if len(item) > 1 else ''
        
        if text:
            texts.append(text)
    
    # Build vocabulary from collected texts
    vocab.build_from_texts(texts)
    
    return vocab


def encode_batch(texts: List[str], vocab: Vocabulary, 
                 add_sos: bool = False, add_eos: bool = False) -> torch.Tensor:
    """
    Encode a batch of texts to padded tensor.
    
    Args:
        texts: List of text strings
        vocab: Vocabulary object
        add_sos: Whether to add start-of-sequence token
        add_eos: Whether to add end-of-sequence token
        
    Returns:
        Padded tensor of shape (batch_size, max_length)
    """
    encoded = [vocab.encode(text, add_sos=add_sos, add_eos=add_eos) for text in texts]
    
    # Find max length
    max_length = max(len(seq) for seq in encoded) if encoded else 0
    
    # Pad sequences
    padded = []
    for seq in encoded:
        padded_seq = seq + [vocab.pad_idx] * (max_length - len(seq))
        padded.append(padded_seq)
    
    return torch.tensor(padded, dtype=torch.long)


def get_sequence_lengths(texts: List[str], vocab: Vocabulary,
                        add_sos: bool = False, add_eos: bool = False) -> torch.Tensor:
    """
    Get actual lengths of encoded sequences (before padding).
    
    Args:
        texts: List of text strings
        vocab: Vocabulary object
        add_sos: Whether to add start-of-sequence token
        add_eos: Whether to add end-of-sequence token
        
    Returns:
        Tensor of sequence lengths
    """
    lengths = [len(vocab.encode(text, add_sos=add_sos, add_eos=add_eos)) for text in texts]
    return torch.tensor(lengths, dtype=torch.long)
