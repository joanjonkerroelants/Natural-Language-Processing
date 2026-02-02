import argparse
import yaml

def load_config(config_path) -> dict:
    "loads configuration from a yaml file"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_parser(config) -> argparse.ArgumentParser:
    "builds the parser based on config"
    parser = argparse.ArgumentParser(
        description="Pipeline for multiple models"
    )

    subparsers = parser.add_subparsers(dest="model", required=True)

    for model_name, model_cfg in config["model"].items():
        sp = subparsers.add_parser(model_name, help=f"{model_name} model")

        # Architecture choices
        sp.add_argument(
            "arc",
            choices=model_cfg["architectures"],
            help=f"Architecture for {model_name}"
        )

        # Dataset only change for final 
        sp.add_argument(
            "--path",
            default=config["dataset"]["train"]["path"],
            help="Dataset path (default: train)"
        )

        # Training hyperparameters 
        sp.add_argument("--lr", 
                        type=float, \
                        default=config["training"]["lr"], 
                        help="Learning rate (default: 0.001)")
        sp.add_argument("--batch_size", 
                        type=int, 
                        default=config["training"]["batch_size"], 
                        help="Batch size (default: 4)")
        sp.add_argument("--epochs", 
                        type=int, 
                        default=config["training"]["epochs"], 
                        help="Number of epochs (default: 80)")
        sp.add_argument("--patience", 
                        type=int, 
                        default=config["training"]["patience"], 
                        help="Early stopping patience (default: 10)")

    parser.add_argument(
        "--verbose",
        required=False,
        help="prints all model details (default: False)",
    )
    return parser

def details_model(config) -> None:
    """
    Prints details about the models and their architectures.
    """
    if config["verbose"]: 
        print(f"Model: {args.model}")
        print(f"Architecture: {args.arc}")
        print(f"Dataset path: {args.path}")
        print(f"Learning rate: {args.lr}")
        print(f"Batch size: {args.batch_size}")
        print(f"Epochs: {args.epochs}")
        print(f"Patience: {args.patience}")
    else:
        print(f"Model: {args.model}")
        print(f"Architecture: {args.arc}")


if __name__ == "__main__":
    config = load_config("config.yaml")
    parser = build_parser(config)
    args = parser.parse_args()
    details_model(config)