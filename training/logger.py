from typing import Optional, Dict, Any
import datetime
import wandb
import pandas as pd
import os
import torch
import yaml
from utils.misc import skip_save
from omegaconf import OmegaConf
from omegaconf import DictConfig


class WandBAstroLogger:
    def __init__(
            self,
            project: str,
            config: DictConfig,
            entity: Optional[str] = None,
            name: Optional[str] = None,
            notes: Optional[str] = None,
            save_path: Optional[str] = None,
    ) -> None:

        self._now: str = datetime.datetime.now().strftime("%d%b-%H_%M_%S")

        self.save_path = save_path if save_path else "./"

        self.config = config
        print(f"Logging wandb to: project={project}, entity={entity}", )
        self.experiment = wandb.init(
            project=project, entity=entity, name=name, notes=notes, config=OmegaConf.to_container(config)
        )
        self.name = self.experiment.name

    def log_metrics(
            self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        if step is not None:
            metrics.update({"global_step": step})
        self.experiment.log(metrics)

    def save_config(self) -> None:
        dir = self.save_path
        if not os.path.exists(dir):
            os.makedirs(dir)

        args_save_name = os.path.join(self.save_path, f"{self._now}_{self.name}_args.yaml")
        OmegaConf.save(config=self.config, f=args_save_name)