from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def create_trainer(config: dict[str, Any]) -> pl.Trainer:
    """Trainer function

    Creates a PyTorch Lightning Trainer with:
      - EarlyStopping on config['early_stop_monitor'] (default 'val_r2')
      - Checkpoint callback for best and last model
    """
    # ------------------------
    # EarlyStopping callback
    # ------------------------
    early_stop_callback = EarlyStopping(
        monitor=config.get("monitor_metric", "val_r2"),
        patience=config.get("early_stop_patience", 5),
        verbose=True,
        mode="max",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=config["output_dir"],
        filename="best-{epoch:02d}-{val_r2:.4f}",
        monitor=config.get("monitor_metric", "val_r2"),
        save_top_k=1,
        save_last=True,
        mode="max",
        every_n_epochs=1,
    )

    # ------------------------
    # Create the Trainer
    # ------------------------
    trainer = pl.Trainer(
        max_epochs=config.get("max_epochs", 100),
        accelerator="auto",
        devices="auto",
        enable_checkpointing=True,
        enable_model_summary=True,
        gradient_clip_val=config.get("gradient_clip_val", 0.0),
        callbacks=[
            early_stop_callback,
            checkpoint_callback,
        ],
    )

    return trainer
