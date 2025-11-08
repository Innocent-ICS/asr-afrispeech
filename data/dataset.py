"""
AfriSpeech-200 Dataset Loader for Shona Language ASR

This module provides a PyTorch Dataset class for loading the Shona subset
from the AfriSpeech-200 dataset hosted on Hugging Face.
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Optional, Dict, Any
from data.afrispeech_loader import AfriSpeechShona


class AfriSpeechDataset(Dataset):
    """
    Dataset class for AfriSpeech-200 Shona language subset.
    
    Loads audio and transcription data from Hugging Face datasets library
    and provides PyTorch Dataset interface for training ASR models.
    
    Requirements addressed:
    - 2.1: Load Shona subset from AfriSpeech-200
    - 2.2: Support train split for model training
    - 2.3: Support dev split for validation
    - 2.4: Support test split for final evaluation
    - 3.1: Support small subset for quick testing (asrking1.py)
    - 3.2: Support full dataset loading (asrking2.py)
    """
    
    def __init__(self, split: str, subset_size: Optional[int] = None):
        """
        Initialize the AfriSpeech dataset loader.
        
        Args:
            split: One of ['train', 'dev', 'test'] specifying which data split to load
            subset_size: Optional number of samples to load. If None, loads full split.
                        Used for quick pipeline testing (asrking1.py uses small subset,
                        asrking2.py uses None for full dataset)
        
        Raises:
            ValueError: If split is not one of ['train', 'dev', 'test']
            RuntimeError: If dataset cannot be loaded from Hugging Face
        """
        if split not in ['train', 'dev', 'test']:
            raise ValueError(f"Split must be one of ['train', 'dev', 'test'], got '{split}'")
        
        self.split = split
        self.subset_size = subset_size
        self.use_custom_loader = False
        
        try:
            # Load the Shona subset from AfriSpeech-200
            # Dataset URL: https://huggingface.co/datasets/intronhealth/afrispeech-200
            print(f"Loading AfriSpeech-200 Shona dataset, split: {split}...")
            
            # Try loading with trust_remote_code for datasets with loading scripts
            try:
                self.dataset = load_dataset(
                    "intronhealth/afrispeech-200",
                    "shona",  # Shona accent config
                    split=split,
                    trust_remote_code=True
                )
                print(f"Successfully loaded {len(self.dataset)} samples from {split} split using Hugging Face loader")
            except Exception as script_error:
                # If loading script fails, try without trust_remote_code
                print(f"Warning: Could not load with loading script: {script_error}")
                print("Attempting alternative loading method...")
                try:
                    self.dataset = load_dataset(
                        "intronhealth/afrispeech-200",
                        "shona",
                        split=split
                    )
                    print(f"Successfully loaded {len(self.dataset)} samples from {split} split")
                except Exception as hf_error:
                    # If Hugging Face loader fails completely, use custom loader
                    print(f"Warning: Hugging Face loader failed: {hf_error}")
                    print("Using custom AfriSpeech loader as fallback...")
                    self.dataset = AfriSpeechShona(split=split, subset_size=subset_size)
                    self.use_custom_loader = True
                    print(f"Successfully loaded {len(self.dataset)} samples using custom loader")
            
            # Validate that the split is not empty
            if len(self.dataset) == 0:
                raise RuntimeError(f"Loaded dataset split '{split}' is empty")
            
            # Apply subset if specified (for quick testing) - only for HF loader
            if subset_size is not None and not self.use_custom_loader:
                if subset_size > len(self.dataset):
                    print(f"Warning: subset_size ({subset_size}) is larger than dataset size "
                          f"({len(self.dataset)}). Using full dataset.")
                    subset_size = len(self.dataset)
                
                self.dataset = self.dataset.select(range(subset_size))
                print(f"Using subset of {subset_size} samples for quick testing")
        
        except Exception as e:
            raise RuntimeError(
                f"Failed to load AfriSpeech-200 dataset. "
                f"Please check your internet connection and Hugging Face access. "
                f"Error: {str(e)}"
            )
    
    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
        
        Returns:
            Dictionary containing:
                - 'audio': Raw audio waveform as numpy array or tensor
                - 'transcription': Text transcription as string
                - 'sample_rate': Audio sample rate in Hz
        """
        sample = self.dataset[idx]
        
        # Handle custom loader format
        if self.use_custom_loader:
            # Custom loader returns dict with 'audio', 'transcription', 'sample_rate'
            return {
                'audio': sample['audio'],
                'transcription': sample['transcription'],
                'sample_rate': sample['sample_rate']
            }
        
        # Handle Hugging Face loader format
        # Extract audio data
        # AfriSpeech-200 stores audio in 'audio' field with 'array' and 'sampling_rate'
        audio_data = sample['audio']
        audio_array = audio_data['array']
        sample_rate = audio_data['sampling_rate']
        
        # Extract transcription
        # The transcription field may be named 'text' or 'transcription'
        transcription = sample.get('text', sample.get('transcription', ''))
        
        return {
            'audio': audio_array,
            'transcription': transcription,
            'sample_rate': sample_rate
        }
    
    def get_split_name(self) -> str:
        """
        Get the name of the current split.
        
        Returns:
            str: Name of the split ('train', 'dev', or 'test')
        """
        return self.split
    
    def is_subset(self) -> bool:
        """
        Check if this dataset is using a subset of the full data.
        
        Returns:
            bool: True if using a subset, False if using full split
        """
        return self.subset_size is not None
