"""
Metrics Evaluator

Implements comprehensive evaluation metrics for ASR models including:
- CTC loss computation on test set
- Character-level accuracy
- Perplexity
- Character Error Rate (CER)
- Word Error Rate (WER)
- Sample transcription generation

Requirements addressed:
- 6.1: Compute and log CTC loss on test set
- 6.2: Compute and log accuracy on test set
- 6.3: Compute and log perplexity on test set
- 6.4: Compute and log Character Error Rate (CER) on test set
- 6.5: Compute and log Word Error Rate (WER) on test set
- 6.6: Generate and log sample text transcriptions
- 6.7: Compute metrics for each of the six model variants
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
from jiwer import wer, cer


class Evaluator:
    """
    Evaluator for computing comprehensive metrics on the test set.
    
    Computes CTC loss, accuracy, perplexity, CER, WER, and generates
    sample transcriptions for qualitative evaluation.
    """
    
    def __init__(self, model: nn.Module, test_loader, vocab, device: torch.device,
                 ctc_loss_fn=None):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained encoder-decoder model (should have encoder and decoder attributes)
            test_loader: DataLoader for test set
            vocab: Vocabulary object for encoding/decoding
            device: torch device (cpu/cuda)
            ctc_loss_fn: CTC loss function (if None, will create one)
        """
        self.model = model
        self.test_loader = test_loader
        self.vocab = vocab
        self.device = device
        
        # Create CTC loss function if not provided
        if ctc_loss_fn is None:
            from training.loss import CTCLoss
            self.ctc_loss_fn = CTCLoss(blank_idx=vocab.blank_idx, reduction='mean')
        else:
            self.ctc_loss_fn = ctc_loss_fn
        
        # Move model to device and set to evaluation mode
        self.model.to(device)
        self.model.eval()
    
    def evaluate(self, num_samples: int = 5) -> Dict:
        """
        Compute all metrics on test set.
        
        Args:
            num_samples: Number of sample transcriptions to generate
        
        Returns:
            results: Dictionary containing:
                - ctc_loss: CTC loss on test set
                - accuracy: Character-level accuracy
                - perplexity: Model perplexity
                - cer: Character Error Rate
                - wer: Word Error Rate
                - samples: List of (reference, hypothesis) pairs
        """
        total_loss = 0.0
        total_correct_chars = 0
        total_chars = 0
        all_references = []
        all_hypotheses = []
        sample_pairs = []
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                # Unpack batch
                audio_features = batch['audio_features'].to(self.device)
                transcriptions = batch['transcriptions']
                audio_lengths = batch['audio_lengths'].to(self.device)
                text_lengths = batch['text_lengths'].to(self.device)
                
                # Encode targets
                targets = self._encode_targets(transcriptions).to(self.device)
                
                # Forward pass through encoder
                encoder_outputs, encoder_hidden = self.model.encoder(
                    audio_features, audio_lengths
                )
                
                # Forward pass through decoder
                logits = self.model.decoder(
                    encoder_outputs, encoder_hidden, text_lengths
                )
                
                # Get actual output sequence length from logits
                output_lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long)
                
                # Compute CTC loss
                batch_loss = self.ctc_loss_fn(
                    logits, targets, output_lengths, text_lengths
                )
                total_loss += batch_loss.item()
                num_batches += 1
                
                # Decode predictions
                predictions = self._decode_predictions(logits)
                
                # Compute character-level accuracy
                correct, total = self._compute_char_accuracy(
                    predictions, transcriptions
                )
                total_correct_chars += correct
                total_chars += total
                
                # Collect references and hypotheses for CER/WER
                all_references.extend(transcriptions)
                all_hypotheses.extend(predictions)
                
                # Collect samples
                if len(sample_pairs) < num_samples:
                    for ref, hyp in zip(transcriptions, predictions):
                        if len(sample_pairs) < num_samples:
                            sample_pairs.append((ref, hyp))
        
        # Compute average loss
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        # Compute accuracy
        accuracy = total_correct_chars / total_chars if total_chars > 0 else 0.0
        
        # Compute perplexity
        perplexity = np.exp(avg_loss) if avg_loss < 100 else float('inf')
        
        # Compute CER and WER
        cer_score = self.compute_cer(all_references, all_hypotheses)
        wer_score = self.compute_wer(all_references, all_hypotheses)
        
        results = {
            'ctc_loss': avg_loss,
            'accuracy': accuracy,
            'perplexity': perplexity,
            'cer': cer_score,
            'wer': wer_score,
            'samples': sample_pairs
        }
        
        return results
    
    def _encode_targets(self, transcriptions: List[str]) -> torch.Tensor:
        """
        Encode target transcriptions to tensor.
        
        Args:
            transcriptions: List of text transcriptions
        
        Returns:
            Padded tensor of target indices
        """
        encoded = [self.vocab.encode(text) for text in transcriptions]
        
        # Find max length
        max_length = max(len(seq) for seq in encoded) if encoded else 0
        
        # Pad sequences
        padded = []
        for seq in encoded:
            padded_seq = seq + [self.vocab.pad_idx] * (max_length - len(seq))
            padded.append(padded_seq)
        
        return torch.tensor(padded, dtype=torch.long)
    
    def _decode_predictions(self, logits: torch.Tensor) -> List[str]:
        """
        Decode model predictions to text strings.
        
        Args:
            logits: Model output logits of shape (batch, max_time, vocab_size)
        
        Returns:
            List of decoded text strings
        """
        # Get most likely character at each time step
        # Shape: (batch, max_time)
        predictions = torch.argmax(logits, dim=-1)
        
        # Decode each sequence using CTC decoding
        decoded_texts = []
        for pred_seq in predictions:
            # Convert to list of indices
            indices = pred_seq.cpu().tolist()
            # Decode using CTC (removes blanks and duplicates)
            text = self.vocab.decode_ctc(indices)
            decoded_texts.append(text)
        
        return decoded_texts
    
    def _compute_char_accuracy(self, predictions: List[str], 
                               references: List[str]) -> Tuple[int, int]:
        """
        Compute character-level accuracy.
        
        Args:
            predictions: List of predicted text strings
            references: List of reference text strings
        
        Returns:
            Tuple of (correct_chars, total_chars)
        """
        correct_chars = 0
        total_chars = 0
        
        for pred, ref in zip(predictions, references):
            # Count matching characters at each position
            min_len = min(len(pred), len(ref))
            for i in range(min_len):
                if pred[i] == ref[i]:
                    correct_chars += 1
            
            # Total characters is the length of reference
            total_chars += len(ref)
        
        return correct_chars, total_chars
    
    def compute_cer(self, references: List[str], hypotheses: List[str]) -> float:
        """
        Compute Character Error Rate (CER) using edit distance.
        
        CER measures the edit distance between reference and hypothesis
        at the character level, normalized by the total number of characters
        in the reference.
        
        Args:
            references: List of reference text strings
            hypotheses: List of hypothesis text strings
        
        Returns:
            CER score (lower is better, 0.0 is perfect)
        """
        if not references or not hypotheses:
            return 1.0
        
        # Filter out empty references to avoid division by zero
        valid_pairs = [(ref, hyp) for ref, hyp in zip(references, hypotheses) 
                      if ref.strip()]
        
        if not valid_pairs:
            return 1.0
        
        valid_refs, valid_hyps = zip(*valid_pairs)
        
        try:
            # Use jiwer library to compute CER
            # jiwer expects lists of strings
            cer_score = cer(list(valid_refs), list(valid_hyps))
            return cer_score
        except Exception as e:
            print(f"Warning: CER computation failed: {e}")
            return 1.0
    
    def compute_wer(self, references: List[str], hypotheses: List[str]) -> float:
        """
        Compute Word Error Rate (WER) using edit distance.
        
        WER measures the edit distance between reference and hypothesis
        at the word level, normalized by the total number of words
        in the reference.
        
        Args:
            references: List of reference text strings
            hypotheses: List of hypothesis text strings
        
        Returns:
            WER score (lower is better, 0.0 is perfect)
        """
        if not references or not hypotheses:
            return 1.0
        
        # Filter out empty references to avoid division by zero
        valid_pairs = [(ref, hyp) for ref, hyp in zip(references, hypotheses) 
                      if ref.strip()]
        
        if not valid_pairs:
            return 1.0
        
        valid_refs, valid_hyps = zip(*valid_pairs)
        
        try:
            # Use jiwer library to compute WER
            # jiwer expects lists of strings
            wer_score = wer(list(valid_refs), list(valid_hyps))
            return wer_score
        except Exception as e:
            print(f"Warning: WER computation failed: {e}")
            return 1.0
    
    def generate_sample_transcriptions(self, num_samples: int = 5) -> List[Tuple[str, str]]:
        """
        Generate sample transcriptions for qualitative evaluation.
        
        Args:
            num_samples: Number of samples to generate
        
        Returns:
            List of (reference, hypothesis) tuples
        """
        samples = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                if len(samples) >= num_samples:
                    break
                
                # Unpack batch
                audio_features = batch['audio_features'].to(self.device)
                transcriptions = batch['transcriptions']
                audio_lengths = batch['audio_lengths'].to(self.device)
                text_lengths = batch['text_lengths'].to(self.device)
                
                # Forward pass
                encoder_outputs, encoder_hidden = self.model.encoder(
                    audio_features, audio_lengths
                )
                logits = self.model.decoder(
                    encoder_outputs, encoder_hidden, text_lengths
                )
                
                # Decode predictions
                predictions = self._decode_predictions(logits)
                
                # Collect samples
                for ref, hyp in zip(transcriptions, predictions):
                    if len(samples) < num_samples:
                        samples.append((ref, hyp))
        
        return samples
    
    def print_results(self, results: Dict):
        """
        Print evaluation results in a formatted way.
        
        Args:
            results: Dictionary of evaluation results
        """
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"CTC Loss:    {results['ctc_loss']:.4f}")
        print(f"Accuracy:    {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"Perplexity:  {results['perplexity']:.4f}")
        print(f"CER:         {results['cer']:.4f} ({results['cer']*100:.2f}%)")
        print(f"WER:         {results['wer']:.4f} ({results['wer']*100:.2f}%)")
        print("="*60)
        
        if 'samples' in results and results['samples']:
            print("\nSAMPLE TRANSCRIPTIONS:")
            print("-"*60)
            for i, (ref, hyp) in enumerate(results['samples'], 1):
                print(f"\nSample {i}:")
                print(f"  Reference:  {ref}")
                print(f"  Hypothesis: {hyp}")
            print("-"*60)


def evaluate_model(model: nn.Module, test_loader, vocab, device: torch.device,
                   num_samples: int = 5) -> Dict:
    """
    Convenience function to evaluate a model on test set.
    
    Args:
        model: Trained encoder-decoder model
        test_loader: DataLoader for test set
        vocab: Vocabulary object
        device: torch device
        num_samples: Number of sample transcriptions to generate
    
    Returns:
        Dictionary of evaluation results
    """
    evaluator = Evaluator(model, test_loader, vocab, device)
    results = evaluator.evaluate(num_samples=num_samples)
    return results
