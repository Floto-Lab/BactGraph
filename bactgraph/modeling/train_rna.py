import json
import os

import numpy as np
import pandas as pd
from lightning import seed_everything
from tap import Tap

from bactgraph.modeling.data_reader import preprocess_data_for_training_rna
from bactgraph.modeling.model import BactGraphModel
from bactgraph.modeling.trainer import create_trainer


def run(args, random_state: int):
    """Run training and evaluation of the BactGraph model."""
    # get the data
    config = args.as_dict()
    data_reader_output = preprocess_data_for_training_rna(
        input_dir=args.input_dir,
        transform_norm_expression_fn=np.log,
        train_size=args.train_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        num_workers=4,
        random_seed=random_state,
        randomize_network=args.randomize_network,
    )
    config["n_genes"] = len(data_reader_output["gene2idx"])

    # read config from args
    model = BactGraphModel(config)
    print("Nr of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # get the trainer
    trainer = create_trainer(config)

    # train the model
    trainer.fit(
        model,
        data_reader_output["train_dataloader"],
        data_reader_output["val_dataloader"],
    )

    # if test data is available, evaluate the model
    if not args.test:
        return

    val_metrics = trainer.test(model, data_reader_output["val_dataloader"], ckpt_path="best")
    with open(os.path.join(args.output_dir, "val_metrics.json"), "w") as f:
        json.dump(val_metrics, f)

    test_metrics = trainer.test(model, data_reader_output["test_dataloader"], ckpt_path="best")
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f)
    return test_metrics


class TrainArgumentParser(Tap):
    """Argument parser for training Bacformer."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    input_dir: str = "/Users/maciejwiatrak/Downloads/data/"
    output_dir: str = "/tmp/gnn-output/"
    train_size: float = 0.6
    test_size: float = 0.2
    random_state: int = 42
    test: bool = False
    input_dim: int = 50
    hidden_dim: int = 50
    output_dim: int = 1
    num_layers: int = 2
    num_heads: int = 2
    dropout: float = 0.2
    lr: float = 0.001
    weight_decay: float = 0.01
    max_epochs: int = 100
    batch_size: int = 32
    monitor_metric: str = "val_r2"
    early_stop_patience: int = 10
    gradient_clip_val: float = 0.0
    randomize_network: bool = False
    # t_max: int = 10


def main(args):
    """Train the model."""
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # write the arguments for reproducibility
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(args.as_dict(), f)

    # run training
    seeds = [1, 2, 3, 4, 5]
    metrics_arr = []
    for seed in seeds:
        seed_everything(seed)
        test_metrics = run(args, random_state=seed)[0]
        test_metrics["seed"] = seed
        metrics_arr.append(test_metrics)
    out_df = pd.DataFrame(metrics_arr)
    out_df["randomize_network"] = args.randomize_network
    out_df.to_csv(os.path.join(args.output_dir, "test_metrics_across_seeds.csv"))


if __name__ == "__main__":
    args = TrainArgumentParser().parse_args()
    print("Args:", args.as_dict())
    main(args)
