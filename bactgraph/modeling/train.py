import json
import os

import numpy as np
from lightning import seed_everything
from tap import Tap


def run(args):
    """Run training and evaluation of the BactGraph model."""
    # get the data
    # TODO: create data reader class
    data_reader_output = preprocess_data_for_training(  # noqa
        input_dir=args.input_dir,
        transform_norm_expression_fn=np.log10,
        train_size=args.train_size,
        test_size=args.test_size,
    )

    # read config from args
    # TODO: create config class
    config = None
    model = BactGraphModel(config)  # noqa
    print("Nr of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # get the trainer
    # TODO: create trainer class
    trainer = get_trainer(args)  # noqa

    # train the model
    # TODO: load best model at the end
    trainer.fit(
        model,
        train_dataloader=data_reader_output["train_dataloader"],
        val_dataloader=data_reader_output["val_dataloader"],
    )

    # if test data is available, evaluate the model
    if not args.test:
        return

    test_metrics = trainer.test(model, test_dataloader=data_reader_output["test_dataloader"])
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
    n_gat_layers: int = 2
    n_heads: int = 2
    hidden_size: int = 480
    dropout: float = 0.2
    lr: float = 0.001
    weight_decay: float = 0.01
    max_epochs: int = 20
    batch_size: int = 32


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
