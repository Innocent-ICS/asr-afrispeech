"""
Experiment logging module for ASR RNN System.
Handles logging to WandB, TensorBoard, and matplotlib visualizations.
"""

import os
import warnings
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("WandB not available. Install with: pip install wandb")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    warnings.warn("TensorBoard not available. Install with: pip install tensorboard")


class ExperimentLogger:
    """
    Logs training metrics to WandB, TensorBoard, and generates visual plots.
    
    Handles experiment tracking across multiple logging backends with graceful
    fallback if any backend is unavailable.
    """
    
    def __init__(self, experiment_name: str, wandb_token: Optional[str] = None,
                 log_dir: str = "logs", project_name: str = "asr-rnn-experiments"):
        """
        Initialize experiment logger with WandB and TensorBoard.
        
        Args:
            experiment_name: Name of the experiment for logging
            wandb_token: WandB API token for authentication (optional, loads from .env if not provided)
            log_dir: Directory for TensorBoard logs and plots
            project_name: WandB project name
        """
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.project_name = project_name
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Load WandB token from .env if not provided
        if wandb_token is None:
            wandb_token = self._load_wandb_token_from_env()
        
        # Initialize WandB
        self.wandb_enabled = False
        if WANDB_AVAILABLE:
            self._init_wandb(wandb_token)
        else:
            print(f"[{experiment_name}] WandB not available, skipping WandB logging")
        
        # Initialize TensorBoard
        self.tensorboard_enabled = False
        self.tb_writer = None
        if TENSORBOARD_AVAILABLE:
            self._init_tensorboard()
        else:
            print(f"[{experiment_name}] TensorBoard not available, skipping TensorBoard logging")
        
        # Storage for plotting
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
    
    def _load_wandb_token_from_env(self) -> Optional[str]:
        """Load WandB token from .env file."""
        env_path = Path('.env')
        if env_path.exists():
            try:
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('wandb_token'):
                            # Parse "wandb_token = <token>" or "wandb_token=<token>"
                            token = line.split('=', 1)[1].strip()
                            if token:
                                print(f"[{self.experiment_name}] Loaded WandB token from .env file")
                                return token
            except Exception as e:
                print(f"[{self.experiment_name}] Failed to load WandB token from .env: {e}")
        return None
    
    def _init_wandb(self, wandb_token: Optional[str] = None):
        """Initialize WandB with authentication."""
        try:
            # Login with token if provided
            if wandb_token:
                try:
                    wandb.login(key=wandb_token)
                except wandb.errors.AuthenticationError as e:
                    raise RuntimeError(
                        f"WandB authentication failed. Invalid API token.\n"
                        f"Please check your WandB token in the .env file.\n"
                        f"You can find your token at: https://wandb.ai/authorize\n"
                        f"Error: {str(e)}"
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"WandB login failed: {str(e)}\n"
                        f"Please check your internet connection and WandB token."
                    )
            
            # Initialize run
            try:
                wandb.init(
                    project=self.project_name,
                    name=self.experiment_name,
                    reinit=True
                )
                self.wandb_enabled = True
                print(f"[{self.experiment_name}] WandB initialized successfully")
            except Exception as e:
                raise RuntimeError(
                    f"WandB initialization failed: {str(e)}\n"
                    f"Please check your internet connection and WandB configuration."
                )
                
        except RuntimeError as e:
            print(f"\n{'!'*80}")
            print(f"[{self.experiment_name}] WandB Error:")
            print(f"  {str(e)}")
            print(f"{'!'*80}")
            print(f"[{self.experiment_name}] Continuing with TensorBoard only\n")
            self.wandb_enabled = False
        except Exception as e:
            print(f"\n{'!'*80}")
            print(f"[{self.experiment_name}] Unexpected WandB error: {str(e)}")
            print(f"{'!'*80}")
            print(f"[{self.experiment_name}] Continuing with TensorBoard only\n")
            self.wandb_enabled = False
    
    def _init_tensorboard(self):
        """Initialize TensorBoard writer."""
        try:
            tb_log_dir = os.path.join(self.log_dir, "tensorboard", self.experiment_name)
            os.makedirs(tb_log_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=tb_log_dir)
            self.tensorboard_enabled = True
            print(f"[{self.experiment_name}] TensorBoard initialized at {tb_log_dir}")
        except Exception as e:
            print(f"[{self.experiment_name}] TensorBoard initialization failed: {e}")
            self.tensorboard_enabled = False
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """
        Log metrics to both WandB and TensorBoard.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Current step/epoch number
        """
        # Log to WandB
        if self.wandb_enabled:
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                print(f"[{self.experiment_name}] WandB logging failed at step {step}: {e}")
        
        # Log to TensorBoard
        if self.tensorboard_enabled and self.tb_writer:
            try:
                for metric_name, metric_value in metrics.items():
                    self.tb_writer.add_scalar(metric_name, metric_value, step)
                self.tb_writer.flush()
            except Exception as e:
                print(f"[{self.experiment_name}] TensorBoard logging failed at step {step}: {e}")
        
        # Store for plotting
        if 'train_loss' in metrics:
            self.train_losses.append(metrics['train_loss'])
        if 'val_loss' in metrics:
            self.val_losses.append(metrics['val_loss'])
        if 'epoch' in metrics:
            self.epochs.append(metrics['epoch'])
    
    def plot_losses(self, train_losses: Optional[List[float]] = None, 
                   val_losses: Optional[List[float]] = None,
                   save_path: Optional[str] = None):
        """
        Create and save visual plots of training and validation losses.
        
        Args:
            train_losses: List of training losses (uses stored if None)
            val_losses: List of validation losses (uses stored if None)
            save_path: Path to save plot (auto-generated if None)
        """
        # Use provided losses or stored losses
        train_losses = train_losses if train_losses is not None else self.train_losses
        val_losses = val_losses if val_losses is not None else self.val_losses
        
        if not train_losses and not val_losses:
            print(f"[{self.experiment_name}] No losses to plot")
            return
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        epochs = list(range(1, len(train_losses) + 1)) if train_losses else list(range(1, len(val_losses) + 1))
        
        if train_losses:
            plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        
        if val_losses:
            plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('CTC Loss', fontsize=12)
        plt.title(f'{self.experiment_name} - Training Progress', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            plots_dir = os.path.join(self.log_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            save_path = os.path.join(plots_dir, f"{self.experiment_name.replace(' ', '_')}_losses.png")
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[{self.experiment_name}] Loss plot saved to {save_path}")
        
        # Log to WandB
        if self.wandb_enabled:
            try:
                wandb.log({"loss_plot": wandb.Image(save_path)})
            except Exception as e:
                print(f"[{self.experiment_name}] Failed to log plot to WandB: {e}")
        
        # Log to TensorBoard
        if self.tensorboard_enabled and self.tb_writer:
            try:
                # Convert plot to image array
                fig = plt.gcf()
                fig.canvas.draw()
                
                # Use buffer_rgba() for newer matplotlib versions
                try:
                    img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                    img_array = img_array[:, :, :3]  # Remove alpha channel
                except AttributeError:
                    # Fallback for older matplotlib versions
                    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                
                # Log to TensorBoard (HWC format, need to convert to CHW)
                img_array = np.transpose(img_array, (2, 0, 1))
                self.tb_writer.add_image('loss_plot', img_array, global_step=len(train_losses))
                self.tb_writer.flush()
            except Exception as e:
                print(f"[{self.experiment_name}] Failed to log plot to TensorBoard: {e}")
        
        plt.close()
    
    def log_test_results(self, results: Dict[str, float]):
        """
        Log final test set results.
        
        Args:
            results: Dictionary of test metrics (ctc_loss, accuracy, perplexity, cer, wer, etc.)
        """
        print(f"\n[{self.experiment_name}] Test Results:")
        for metric_name, metric_value in results.items():
            if isinstance(metric_value, (int, float)):
                print(f"  {metric_name}: {metric_value:.4f}")
        
        # Log to WandB
        if self.wandb_enabled:
            try:
                test_metrics = {f"test/{k}": v for k, v in results.items() if isinstance(v, (int, float))}
                wandb.log(test_metrics)
            except Exception as e:
                print(f"[{self.experiment_name}] Failed to log test results to WandB: {e}")
        
        # Log to TensorBoard
        if self.tensorboard_enabled and self.tb_writer:
            try:
                for metric_name, metric_value in results.items():
                    if isinstance(metric_value, (int, float)):
                        self.tb_writer.add_scalar(f"test/{metric_name}", metric_value, 0)
                self.tb_writer.flush()
            except Exception as e:
                print(f"[{self.experiment_name}] Failed to log test results to TensorBoard: {e}")
    
    def log_sample_transcriptions(self, samples: List[tuple]):
        """
        Log sample transcriptions for qualitative evaluation.
        
        Args:
            samples: List of (reference, hypothesis) tuples
        """
        if not samples:
            return
        
        print(f"\n[{self.experiment_name}] Sample Transcriptions:")
        for i, (ref, hyp) in enumerate(samples[:5], 1):  # Show first 5
            print(f"  Sample {i}:")
            print(f"    Reference:  {ref}")
            print(f"    Hypothesis: {hyp}")
        
        # Log to WandB as table
        if self.wandb_enabled:
            try:
                table = wandb.Table(columns=["Reference", "Hypothesis"])
                for ref, hyp in samples:
                    table.add_data(ref, hyp)
                wandb.log({"sample_transcriptions": table})
            except Exception as e:
                print(f"[{self.experiment_name}] Failed to log samples to WandB: {e}")
    
    def close(self):
        """Close logging resources."""
        if self.tensorboard_enabled and self.tb_writer:
            self.tb_writer.close()
        
        if self.wandb_enabled:
            try:
                wandb.finish()
            except Exception as e:
                print(f"[{self.experiment_name}] Failed to close WandB: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
