import json
import os

import numpy as np
from lightning import seed_everything
from tap import Tap

from bactgraph.modeling.data_reader import preprocess_data_for_training
from bactgraph.modeling.model import BactGraphModel
from bactgraph.modeling.trainer import create_trainer


def run(args):
    """Run training and evaluation of the BactGraph model."""
    # get the data
    data_reader_output = preprocess_data_for_training(
        input_dir=args.input_dir,
        transform_norm_expression_fn=np.log10,
        train_size=args.train_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        num_workers=4,
        random_seed=args.random_state,
    )

    # read config from args
    model = BactGraphModel(args.as_dict())
    print("Nr of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # get the trainer
    trainer = create_trainer(args)

    # train the model
    trainer.fit(
        model,
        data_reader_output["train_dataloader"],
        data_reader_output["val_dataloader"],
    )

    # if test data is available, evaluate the model
    if not args.test:
        return

    test_metrics = trainer.test(model, data_reader_output["test_dataloader"], ckpt_path="best")
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f)


class TrainArgumentParser(Tap):
    """Argument parser for training Bacformer."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    input_dir: str
    output_dir: str
    train_size: float = 0.7
    test_size: float = 0.2
    random_state: int = 42
    test: bool = False
    input_dim: int = 480
    hidden_dim: int = 480
    output_dim: int = 1
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.2
    lr: float = 0.001
    weight_decay: float = 0.01
    max_epochs: int = 100
    batch_size: int = 32
    monitor_metric: str = "val_r2"
    early_stop_patience: int = 5
    gradient_clip_val: float = 0.0


def main(args):
    """Train the model."""
    seed_everything(args.random_state)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # write the arguments for reproducibility
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(args.as_dict(), f)

    # run training
    run(args)


if __name__ == "__main__":
    args = TrainArgumentParser().parse_args()
    print("Args:", args.as_dict())
    main(args)
