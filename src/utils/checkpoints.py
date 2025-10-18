import os
from typing import Optional, Any
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as StepLR
from torch.optim import Optimizer
from torch.amp import GradScaler
import optuna
import numpy as np
import wandb
from torch.serialization import add_safe_globals


class Checkpoint:
    """
    Class for Checkpoints of a 3D CNN model

    Parameters
    ----------
    model : nn.Module
        Model to train.
    config : Any
        Configuration object parsed from YAML.
    """

    def __init__(self, model: nn.Module, optimizer: Optimizer, scaler: GradScaler, scheduler: StepLR, config: Any, device: torch.device, start_epoch: int = 1, trial: Optional[optuna.trial.Trial] = None, early_stopping: Optional[Any] = None) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = scheduler
        self.device = device
        self.start_epoch = start_epoch
        self.config = config
        self.trial = trial
        self.early_stopping = early_stopping
        
        if torch.cuda.is_available():
            _ = torch.randn(1).to(self.device)        

    def load_checkpoint(self) -> None:
        """
        Load the last checkpoint if available and resuming is allowed.
        """
        add_safe_globals({
            'numpy.core.multiarray._reconstruct': np.core.multiarray._reconstruct
        })
        
        checkpoint_dir = os.path.dirname(self.config.output.checkpoints_path)
        path = os.path.join(checkpoint_dir, "last_checkpoint.pth")
        
        print("Config path:", self.config.output.checkpoints_path)
        print("Final checkpoint path:", path)
        print("Exists:", os.path.exists(path))
        if os.path.exists(path) and self.trial is None:
            try:
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                
                print("Key of the checkpoint:", checkpoint.keys())
                print("Epoch saved:", checkpoint.get("epoch"))
                
                
                self.model.load_state_dict(checkpoint["model_state"], strict=False)
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
                self.scaler.load_state_dict(checkpoint["scaler_state"])
                
                if self.scheduler and checkpoint.get("scheduler_state"):
                    self.scheduler.load_state_dict(checkpoint["scheduler_state"])
                # Torch RNG
                if "torch_rng_state" in checkpoint:
                    try:
                        state = checkpoint["torch_rng_state"]
                        if isinstance(state, torch.ByteTensor):
                            torch.set_rng_state(state)
                        else:
                            print("Could not restore PyTorch RNG state: Incorrect type, got", type(state))
                    except Exception as e:
                        print("Error restoring torch RNG state:", e)

                # CUDA RNG
                if "cuda_rng_state" in checkpoint and checkpoint["cuda_rng_state"] is not None:
                    try:
                        # Debe ser lista de ByteTensor
                        stateCuda = checkpoint["cuda_rng_state"]
                        if isinstance(checkpoint["cuda_rng_state"], list) and all(isinstance(t, torch.ByteTensor) for t in checkpoint["cuda_rng_state"]):
                            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
                        else:
                            print("Could not restore CUDA RNG state: Bad format or type")
                            print("Could not restore CUDA RNG: Incorrect type, got", type(stateCuda))
                    except Exception as e:
                        print(f"Could not restore CUDA RNG state: {e}")

                # NumPy RNG
                if "numpy_rng_state" in checkpoint:
                    try:
                        np.random.set_state(checkpoint["numpy_rng_state"])
                    except Exception as e:
                        print(f"Could not restore NumPy RNG state: {e}")

                # EarlyStopping state
                if "early_stopping_state" in checkpoint and self.early_stopping:
                    print("Ladding early_stopping_state:", checkpoint.get("early_stopping_state"))
                    self.early_stopping.counter = checkpoint["early_stopping_state"]["counter"]
                    self.early_stopping.best_score = checkpoint["early_stopping_state"]["best_score"]
                    print("best_score en checkpoint:", checkpoint.get("early_stopping_state", {}).get("best_score"))


                # WandB run ID (reconnect)
                if "wandb_run_id" in checkpoint and checkpoint["wandb_run_id"]:
                    os.environ["WANDB_RUN_ID"] = checkpoint["wandb_run_id"]                    
                
                if "epoch" in checkpoint:
                    self.start_epoch = checkpoint["epoch"] + 1
                print(f"Resumed from checkpoint at epoch {self.start_epoch}")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")

    def save_last_checkpoint(self, epoch: int) -> None:
        """
        Save current training checkpoint.

        Parameters
        ----------
        epoch : int
            Current epoch.
        """
        checkpoint = {
        "epoch": epoch,
        "model_state": self.model.state_dict(),
        "optimizer_state": self.optimizer.state_dict(),
        "scaler_state": self.scaler.state_dict(),
        "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
        "torch_rng_state": torch.get_rng_state().clone().detach().cpu(),
        "cuda_rng_state": [t.clone().detach().to(torch.uint8) for t in torch.cuda.get_rng_state_all()],
        "numpy_rng_state": np.random.get_state(),
        "wandb_run_id": wandb.run.id if wandb.run else None,
        }

            
        if self.early_stopping and self.early_stopping.best_score is not None:
            checkpoint["early_stopping_state"] = {
                "counter": self.early_stopping.counter,
                "best_score": self.early_stopping.best_score
            }
        else:
            print("No se guardó early_stopping_state porque best_score era None.")


        path = os.path.join(os.path.dirname(
            self.config.output.checkpoints_path), "last_checkpoint.pth")
        
        try:
            torch.save(checkpoint, path)
            print(f"Checkpoint saved at: {path}")
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")       
        
        
    def get_start_epoch(self):
            return self.start_epoch
