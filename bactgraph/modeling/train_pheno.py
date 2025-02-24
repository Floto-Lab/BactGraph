import json
import os

from lightning import seed_everything

from bactgraph.modeling.data_reader import preprocess_data_for_training_pheno
from bactgraph.modeling.model import BactGraphModel
from bactgraph.modeling.train_rna import TrainArgumentParser
from bactgraph.modeling.trainer import create_trainer


def run(args):
    """Run training and evaluation of the BactGraph model."""
    # get the data
    config = args.as_dict()
    data_reader_output = preprocess_data_for_training_pheno(
        input_dir=args.input_dir,
        train_size=args.train_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        num_workers=4,
        random_seed=args.random_state,
        randomize_network=args.randomize_network,
        label_col="label",
    )
    config["n_genes"] = len(data_reader_output["gene2idx"])

    # read config from args
    model = BactGraphModel(config, phenotype_prediction=True)
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
