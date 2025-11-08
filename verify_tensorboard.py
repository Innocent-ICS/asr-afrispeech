#!/usr/bin/env python
"""
TensorBoard Verification Script

This script verifies that TensorBoard logs exist and contain data,
then provides instructions for viewing them.
"""

import os
import sys
from pathlib import Path

def check_tensorboard_logs():
    """Check if TensorBoard logs exist and contain data."""
    
    print("="*80)
    print("TensorBoard Logging Verification")
    print("="*80)
    print()
    
    # Check if logs directory exists
    logs_dir = Path("logs/tensorboard")
    
    if not logs_dir.exists():
        print("❌ ERROR: TensorBoard logs directory not found!")
        print(f"   Expected location: {logs_dir.absolute()}")
        print()
        print("   This means no experiments have been run yet.")
        print("   Run asrking1.py or asrking2.py to generate logs.")
        return False
    
    print(f"✅ TensorBoard logs directory found: {logs_dir.absolute()}")
    print()
    
    # Find all experiment directories
    experiment_dirs = [d for d in logs_dir.iterdir() if d.is_dir()]
    
    if not experiment_dirs:
        print("❌ ERROR: No experiment directories found!")
        print("   Run asrking1.py or asrking2.py to generate logs.")
        return False
    
    print(f"✅ Found {len(experiment_dirs)} experiment(s):")
    print()
    
    total_events = 0
    
    for exp_dir in sorted(experiment_dirs):
        # Find event files
        event_files = list(exp_dir.glob("events.out.tfevents.*"))
        
        if event_files:
            print(f"   📊 {exp_dir.name}")
            print(f"      Event files: {len(event_files)}")
            
            # Get file size
            total_size = sum(f.stat().st_size for f in event_files)
            print(f"      Total size: {total_size / 1024:.2f} KB")
            
            # Try to read metrics
            try:
                from tensorboard.backend.event_processing import event_accumulator
                
                ea = event_accumulator.EventAccumulator(str(event_files[0]))
                ea.Reload()
                
                scalars = ea.Tags()['scalars']
                if scalars:
                    print(f"      Metrics: {', '.join(scalars)}")
                    
                    # Get number of logged steps
                    if 'train_loss' in scalars:
                        train_loss_events = ea.Scalars('train_loss')
                        print(f"      Epochs logged: {len(train_loss_events)}")
                        
                        if train_loss_events:
                            latest = train_loss_events[-1]
                            print(f"      Latest train_loss: {latest.value:.4f} (step {latest.step})")
                
                total_events += 1
                
            except ImportError:
                print(f"      ⚠ Cannot read metrics (tensorboard not installed)")
            except Exception as e:
                print(f"      ⚠ Error reading metrics: {e}")
            
            print()
    
    if total_events == 0:
        print("⚠ WARNING: Event files found but couldn't read any metrics")
        print("   TensorBoard may still work - try launching it to verify")
        print()
    
    return True


def print_launch_instructions():
    """Print instructions for launching TensorBoard."""
    
    print("="*80)
    print("How to View TensorBoard Logs")
    print("="*80)
    print()
    
    print("Option 1: Use the launch script (easiest)")
    print("-" * 40)
    print("   ./launch_tensorboard.sh")
    print()
    
    print("Option 2: Manual launch")
    print("-" * 40)
    print("   conda activate asr-rnn")
    print("   tensorboard --logdir=logs/tensorboard --port=6006")
    print()
    
    print("Then open your browser to:")
    print("   🌐 http://localhost:6006")
    print()
    
    print("="*80)
    print()


def main():
    """Main function."""
    
    # Check if we're in the right directory
    if not Path("asrking1.py").exists() and not Path("asrking2.py").exists():
        print("❌ ERROR: This script must be run from the project root directory")
        print("   (the directory containing asrking1.py and asrking2.py)")
        sys.exit(1)
    
    # Check TensorBoard logs
    logs_exist = check_tensorboard_logs()
    
    if logs_exist:
        print_launch_instructions()
        
        # Ask if user wants to launch TensorBoard
        try:
            response = input("Would you like to launch TensorBoard now? (y/n): ").strip().lower()
            
            if response in ['y', 'yes']:
                print()
                print("Launching TensorBoard...")
                print("Press Ctrl+C to stop TensorBoard")
                print()
                
                import subprocess
                try:
                    subprocess.run(['tensorboard', '--logdir=logs/tensorboard', '--port=6006'])
                except KeyboardInterrupt:
                    print("\n\nTensorBoard stopped.")
                except FileNotFoundError:
                    print("❌ ERROR: tensorboard command not found")
                    print("   Install it with: pip install tensorboard")
                    print("   Or use: conda activate asr-rnn && pip install tensorboard")
        except KeyboardInterrupt:
            print("\n\nCancelled.")
    else:
        print("Run asrking1.py or asrking2.py first to generate TensorBoard logs.")
        print()


if __name__ == "__main__":
    main()

