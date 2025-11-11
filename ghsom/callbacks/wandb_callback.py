"""WandB tracking callback for GHSOM training."""

from ghsom.callbacks.base import TrackingCallback


class WandBCallback(TrackingCallback):
    """
    Callback for tracking GHSOM training with Weights & Biases (WandB).

    This callback integrates GHSOM training with WandB for experiment tracking,
    logging metrics, and visualizing results.

    Example usage:
        >>> from ghsom import GHSOM
        >>> from ghsom.callbacks import WandBCallback
        >>>
        >>> ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05, ...)
        >>> callback = WandBCallback(project="GHSOM", name="experiment_1")
        >>> model = ghsom.train(epochs_number=50, callbacks=[callback])
    """

    def __init__(self, project="GHSOM", name=None, **wandb_kwargs):
        """
        Initialize WandB callback.

        :param project: str, optional (default="GHSOM")
            WandB project name.

        :param name: str, optional (default=None)
            Name for this run. If None, WandB will auto-generate.

        :param wandb_kwargs: Additional keyword arguments passed to wandb.init()
        """
        try:
            import wandb

            self.wandb = wandb
        except ImportError:
            raise ImportError(
                "WandB is not installed. Install it with: pip install wandb"
            )

        self.project = project
        self.name = name
        self.wandb_kwargs = wandb_kwargs
        self.run = None

    def on_train_begin(self, config):
        """
        Initialize WandB run at the beginning of training.

        :param config: dict
            Training configuration to log.
        """
        self.run = self.wandb.init(
            project=self.project, name=self.name, config=config, **self.wandb_kwargs
        )

    def on_map_created(self, metrics):
        """
        Log metrics when a new map is created.

        :param metrics: dict
            Metrics to log (e.g., quantization error).
        """
        if self.run is not None:
            self.wandb.log(metrics)

    def on_train_end(self, results):
        """
        Finalize WandB run at the end of training.

        :param results: dict
            Final training results to log.
        """
        if self.run is not None:
            self.wandb.log(results)
            self.wandb.finish()
