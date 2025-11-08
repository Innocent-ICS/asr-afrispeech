"""
Custom loader for AfriSpeech-200 Shona dataset.

This module downloads and loads the Shona language data directly from the 
Hugging Face repository without using the deprecated loading script.
"""

import os
import tarfile
import csv
import requests
from pathlib import Path
from typing import Dict, List, Optional
import soundfile as sf
from tqdm import tqdm


class AfriSpeechShona:
    """
    Custom loader for AfriSpeech-200 Shona dataset.
    Downloads audio files and transcripts directly from Hugging Face.
    """
    
    BASE_URL = "https://huggingface.co/datasets/intronhealth/afrispeech-200/resolve/main"
    
    def __init__(self, data_dir: str = "./data/afrispeech_shona", split: str = "train", subset_size: Optional[int] = None):
        """
        Initialize the Shona dataset loader.
        
        Args:
            data_dir: Directory to store downloaded data
            split: One of ['train', 'dev', 'test']
            subset_size: Optional number of samples to load (for quick testing)
        """
        if split not in ['train', 'dev', 'test']:
            raise ValueError(f"Split must be one of ['train', 'dev', 'test'], got '{split}'")
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.subset_size = subset_size
        
        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Define file paths
        self.audio_dir = self.data_dir / "audio" / split
        self.transcript_file = self.data_dir / "transcripts" / f"{split}.csv"
        
        # Download and extract data if needed
        self._download_and_extract()
        
        # Load transcripts
        self.samples = self._load_transcripts()
        
        print(f"Loaded {len(self.samples)} samples from {split} split")
    
    def _download_file(self, url: str, dest_path: Path) -> None:
        """Download a file from URL to destination path."""
        if dest_path.exists():
            print(f"File already exists: {dest_path.name}")
            return
        
        print(f"Downloading {dest_path.name}...")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Failed to download {dest_path.name}. Please check your internet connection.\n"
                f"URL: {url}\n"
                f"Error: {str(e)}"
            )
        except requests.exceptions.Timeout as e:
            raise TimeoutError(
                f"Download timed out for {dest_path.name}. Please try again.\n"
                f"URL: {url}\n"
                f"Error: {str(e)}"
            )
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(
                f"HTTP error downloading {dest_path.name}.\n"
                f"URL: {url}\n"
                f"Status code: {response.status_code}\n"
                f"Error: {str(e)}"
            )
        
        total_size = int(response.headers.get('content-length', 0))
        
        try:
            with open(dest_path, 'wb') as f, tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=dest_path.name
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        except IOError as e:
            raise IOError(
                f"Failed to write downloaded file to {dest_path}.\n"
                f"Please check disk space and write permissions.\n"
                f"Error: {str(e)}"
            )
    
    def _extract_tar(self, tar_path: Path, extract_dir: Path) -> None:
        """Extract a tar.gz file."""
        if extract_dir.exists() and any(extract_dir.iterdir()):
            print(f"Already extracted: {tar_path.name}")
            return
        
        print(f"Extracting {tar_path.name}...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(extract_dir)
    
    def _download_and_extract(self) -> None:
        """Download and extract audio files and transcripts."""
        # Download transcript
        transcript_url = f"{self.BASE_URL}/transcripts/shona/{self.split}.csv"
        self.transcript_file.parent.mkdir(parents=True, exist_ok=True)
        self._download_file(transcript_url, self.transcript_file)
        
        # Download and extract audio
        audio_tar_name = f"{self.split}_shona_0.tar.gz"
        audio_tar_url = f"{self.BASE_URL}/audio/shona/{self.split}/{audio_tar_name}"
        audio_tar_path = self.data_dir / "audio_archives" / audio_tar_name
        
        self._download_file(audio_tar_url, audio_tar_path)
        self._extract_tar(audio_tar_path, self.audio_dir)
    
    def _load_transcripts(self) -> List[Dict]:
        """Load transcripts from CSV file."""
        samples = []
        
        if not self.transcript_file.exists():
            raise FileNotFoundError(
                f"Transcript file not found: {self.transcript_file}\n"
                f"Please ensure the dataset was downloaded correctly."
            )
        
        try:
            with open(self.transcript_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    samples.append({
                        'audio_path': row['audio_paths'],
                        'transcription': row['transcript'],
                        'speaker_id': row.get('user_ids', ''),
                        'gender': row.get('gender', ''),
                        'accent': row.get('accent', ''),
                        'audio_id': row.get('audio_ids', ''),
                        'duration': row.get('duration', '')
                    })
                    
                    if self.subset_size and len(samples) >= self.subset_size:
                        break
        except KeyError as e:
            raise ValueError(
                f"Missing required column in transcript file: {e}\n"
                f"Expected columns: audio_paths, transcript\n"
                f"File: {self.transcript_file}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load transcripts from {self.transcript_file}\n"
                f"Error: {str(e)}"
            )
        
        if len(samples) == 0:
            raise ValueError(
                f"No samples found in {self.split} split.\n"
                f"Transcript file: {self.transcript_file}\n"
                f"Please check if the file is empty or corrupted."
            )
        
        return samples
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.
        
        Args:
            idx: Index of the sample
        
        Returns:
            Dictionary containing:
                - 'audio': Audio waveform as numpy array
                - 'transcription': Text transcription
                - 'sample_rate': Audio sample rate
                - 'speaker_id': Speaker ID
                - 'gender': Speaker gender
                - 'accent': Speaker accent
        """
        sample = self.samples[idx]
        
        # Find the audio file
        audio_filename = os.path.basename(sample['audio_path'])
        
        # Search for the audio file in the extracted directory
        audio_file = None
        for root, dirs, files in os.walk(self.audio_dir):
            if audio_filename in files:
                audio_file = Path(root) / audio_filename
                break
        
        if audio_file is None or not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_filename}")
        
        # Load audio
        try:
            audio, sample_rate = sf.read(audio_file)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load audio file: {audio_file}\n"
                f"The file may be corrupted or in an unsupported format.\n"
                f"Error: {str(e)}"
            )
        
        return {
            'audio': audio,
            'transcription': sample['transcription'],
            'sample_rate': sample_rate,
            'speaker_id': sample['speaker_id'],
            'gender': sample['gender'],
            'accent': sample['accent']
        }
    
    def get_split_name(self) -> str:
        """Get the name of the current split."""
        return self.split
    
    def is_subset(self) -> bool:
        """Check if using a subset of the data."""
        return self.subset_size is not None
