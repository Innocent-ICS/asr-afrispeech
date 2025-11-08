"""
Audio Preprocessing Module for ASR System

This module provides audio preprocessing functionality including MFCC feature extraction
and batch collation for variable-length sequences.
"""

import torch
import torchaudio
import numpy as np
from typing import List, Tuple, Dict, Any


class AudioPreprocessor:
    """
    Audio preprocessor for converting raw audio waveforms into features suitable for RNN input.
    
    Supports MFCC feature extraction and handles variable-length sequences through
    padding and length tracking for packed sequences.
    
    Requirements addressed:
    - 2.5: Preprocess audio data into a format suitable for RNN input
    """
    
    def __init__(self, feature_type: str = 'mfcc', n_features: int = 40, sample_rate: int = 16000):
        """
        Initialize the audio preprocessor.
        
        Args:
            feature_type: Type of audio features to extract ('mfcc' or 'mel_spectrogram')
            n_features: Number of feature dimensions (e.g., number of MFCC coefficients)
            sample_rate: Target sample rate for audio processing
        """
        self.feature_type = feature_type
        self.n_features = n_features
        self.sample_rate = sample_rate
        
        # Initialize MFCC transform
        if feature_type == 'mfcc':
            self.mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=n_features,
                melkwargs={
                    'n_fft': 400,
                    'hop_length': 160,
                    'n_mels': 80,
                    'center': False
                }
            )
        elif feature_type == 'mel_spectrogram':
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=400,
                hop_length=160,
                n_mels=n_features,
                center=False
            )
        else:
            raise ValueError(f"Unsupported feature_type: {feature_type}. Use 'mfcc' or 'mel_spectrogram'")
    
    def process(self, audio: np.ndarray, sample_rate: int) -> torch.Tensor:
        """
        Process raw audio waveform into feature representation.
        
        Args:
            audio: Raw audio waveform as numpy array
            sample_rate: Sample rate of the input audio
        
        Returns:
            features: Tensor of shape (time_steps, n_features)
        """
        try:
            # Convert numpy array to torch tensor
            if isinstance(audio, np.ndarray):
                audio_tensor = torch.from_numpy(audio).float()
            else:
                audio_tensor = torch.tensor(audio).float()
            
            # Check for NaN or infinite values in audio
            if torch.isnan(audio_tensor).any() or torch.isinf(audio_tensor).any():
                raise ValueError("Audio contains NaN or infinite values")
            
            # Ensure audio is 1D
            if audio_tensor.dim() == 0:
                audio_tensor = audio_tensor.unsqueeze(0)
            elif audio_tensor.dim() > 1:
                # Handle multi-channel audio (e.g., stereo)
                # Convert to mono by averaging channels
                if audio_tensor.shape[1] == 2:  # Shape is (samples, channels)
                    audio_tensor = audio_tensor.mean(dim=1)
                elif audio_tensor.shape[0] == 2:  # Shape is (channels, samples)
                    audio_tensor = audio_tensor.mean(dim=0)
                else:
                    # Take first channel if more than 2 channels
                    audio_tensor = audio_tensor[0] if audio_tensor.shape[0] < audio_tensor.shape[1] else audio_tensor[:, 0]
            
            # Resample if necessary
            if sample_rate != self.sample_rate:
                try:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sample_rate,
                        new_freq=self.sample_rate
                    )
                    audio_tensor = resampler(audio_tensor)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to resample audio from {sample_rate}Hz to {self.sample_rate}Hz.\n"
                        f"Error: {str(e)}"
                    )
            
            # Handle very short audio by padding to minimum length
            min_length = 400  # Minimum length for n_fft=400
            if audio_tensor.shape[0] < min_length:
                padding = min_length - audio_tensor.shape[0]
                audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))
            
            # Extract features
            try:
                if self.feature_type == 'mfcc':
                    # MFCC transform expects shape (channel, time)
                    audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension
                    features = self.mfcc_transform(audio_tensor)  # Shape: (channel, n_mfcc, time)
                    features = features.squeeze(0)  # Remove channel dimension: (n_mfcc, time)
                    features = features.transpose(0, 1)  # Transpose to (time, n_mfcc)
                elif self.feature_type == 'mel_spectrogram':
                    audio_tensor = audio_tensor.unsqueeze(0)
                    features = self.mel_transform(audio_tensor)
                    features = torch.log(features + 1e-9)  # Log mel spectrogram
                    features = features.squeeze(0)
                    features = features.transpose(0, 1)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to extract {self.feature_type} features from audio.\n"
                    f"Audio shape: {audio_tensor.shape}\n"
                    f"Error: {str(e)}"
                )
            
            # Check for NaN or infinite values in features
            if torch.isnan(features).any() or torch.isinf(features).any():
                raise ValueError("Extracted features contain NaN or infinite values")
            
            return features
            
        except Exception as e:
            # Re-raise with more context if not already a custom error
            if isinstance(e, (ValueError, RuntimeError)):
                raise
            else:
                raise RuntimeError(
                    f"Unexpected error during audio preprocessing.\n"
                    f"Error type: {type(e).__name__}\n"
                    f"Error: {str(e)}"
                )
    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, List[str], torch.Tensor, torch.Tensor]:
        """
        Collate a batch of samples with padding for variable-length sequences.
        
        This function is designed to be used as the collate_fn parameter in PyTorch DataLoader.
        It processes raw audio into features, pads sequences to the same length, and tracks
        actual sequence lengths for packed sequence processing.
        
        Args:
            batch: List of dictionaries, each containing:
                - 'audio': Raw audio waveform
                - 'transcription': Text transcription
                - 'sample_rate': Audio sample rate
        
        Returns:
            Tuple containing:
                - audio_features: Padded tensor of shape (batch, max_time, n_features)
                - transcriptions: List of text transcriptions
                - audio_lengths: Tensor of actual audio sequence lengths (batch,)
                - text_lengths: Tensor of actual text sequence lengths (batch,)
        """
        if not batch:
            raise ValueError("Cannot collate empty batch")
        
        # Process each audio sample into features
        features_list = []
        transcriptions = []
        failed_indices = []
        
        for idx, sample in enumerate(batch):
            try:
                # Validate sample structure
                if not isinstance(sample, dict):
                    raise ValueError(f"Sample at index {idx} is not a dictionary")
                
                if 'audio' not in sample:
                    raise ValueError(f"Sample at index {idx} missing 'audio' key")
                
                if 'transcription' not in sample:
                    raise ValueError(f"Sample at index {idx} missing 'transcription' key")
                
                if 'sample_rate' not in sample:
                    raise ValueError(f"Sample at index {idx} missing 'sample_rate' key")
                
                # Extract audio features
                features = self.process(sample['audio'], sample['sample_rate'])
                features_list.append(features)
                transcriptions.append(sample['transcription'])
                
            except Exception as e:
                # Log the error but continue with other samples
                print(f"Warning: Failed to process sample {idx} in batch: {str(e)}")
                failed_indices.append(idx)
        
        # Check if we have any valid samples
        if not features_list:
            raise RuntimeError(
                f"All samples in batch failed to process.\n"
                f"Failed indices: {failed_indices}\n"
                f"Please check audio data quality and format."
            )
        
        if failed_indices:
            print(f"Warning: Skipped {len(failed_indices)} samples in batch due to processing errors")
        
        # Get actual lengths before padding
        audio_lengths = torch.tensor([features.shape[0] for features in features_list], dtype=torch.long)
        text_lengths = torch.tensor([len(text) for text in transcriptions], dtype=torch.long)
        
        # Pad audio features to max length in batch
        max_audio_len = max(features.shape[0] for features in features_list)
        batch_size = len(features_list)
        
        # Create padded tensor
        audio_features = torch.zeros(batch_size, max_audio_len, self.n_features)
        
        for i, features in enumerate(features_list):
            audio_len = features.shape[0]
            audio_features[i, :audio_len, :] = features
        
        return audio_features, transcriptions, audio_lengths, text_lengths
