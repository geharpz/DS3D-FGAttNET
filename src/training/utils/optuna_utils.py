
from typing import Optional, Dict, Any
import optuna

class OptunaUtils:
    
    def __init__(self, config: Any, trial: Optional[optuna.trial.Trial] = None) -> None:
        self.config = config
        self.trial = trial
        
    def report_optuna(self, metrics: Dict[str, float], epoch: int) -> None:
        """
        Report multiple metrics to Optuna for pruning decision, using dynamic metrics_to_monitor.

        Parameters
        ----------
        metrics : Dict[str, float]
            Dictionary with computed metrics (f1, precision, recall, etc.).
        epoch : int
            Current epoch.
        """
        if not isinstance(metrics, dict):
            print("Provided metrics is not a dictionary.")
            return
        if self.trial:
            # Decide primary metric to report from config
            primary_metric = getattr(
                self.config.training, "early_stopping_metric", "val_loss")
            metric_value = metrics.get(primary_metric, None)
            if metric_value is not None:
                self.trial.report(metric_value, epoch)
                if self.trial.should_prune():
                    raise optuna.TrialPruned()
            else:
                print(
                    f"Metric '{primary_metric}' not found in metrics. Check your compute logic.")