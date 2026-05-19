import argparse

from .model import ModelConfig
from .workflows import AssignmentConfig, workflow_auto_cluster, workflow_classify_singletons, workflow_merge_training


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-path", default="models/char_transformer_cache.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-len", type=int, default=80)
    parser.add_argument("--d-model", type=int, default=192)


def model_config(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(epochs=args.epochs, batch_size=args.batch_size, max_len=args.max_len, d_model=args.d_model)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="name-cluster", description="Character Transformer workflow for firm-name clustering.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    auto = subparsers.add_parser("auto-cluster", help="Workflow A: learn from training set and assign ids to the full sample.")
    auto.add_argument("--train", required=True)
    auto.add_argument("--full", required=True)
    auto.add_argument("--output", required=True)
    auto.add_argument("--name-col")
    auto.add_argument("--threshold", type=float, default=0.95)
    auto.add_argument("--min-train-cluster-size", type=int, default=2)
    add_model_args(auto)

    classify = subparsers.add_parser("classify-singletons", help="Workflow B: assign pending singleton names to existing clusters.")
    classify.add_argument("--train", required=True)
    classify.add_argument("--pending", required=True)
    classify.add_argument("--output", required=True)
    classify.add_argument("--name-col")
    classify.add_argument("--headerless", action="store_true")
    classify.add_argument("--threshold", type=float, default=0.95)
    classify.add_argument("--min-train-cluster-size", type=int, default=3)
    add_model_args(classify)

    merge = subparsers.add_parser("merge-training", help="Purify training clusters by high-threshold centroid merge candidates.")
    merge.add_argument("--train", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--threshold", type=float, default=0.93)
    merge.add_argument("--topk", type=int, default=20)
    add_model_args(merge)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = model_config(args)

    if args.command == "auto-cluster":
        assignment = AssignmentConfig(args.threshold, args.min_train_cluster_size, args.cache_path)
        workflow_auto_cluster(args.train, args.full, args.output, args.name_col, assignment, cfg)
    elif args.command == "classify-singletons":
        assignment = AssignmentConfig(args.threshold, args.min_train_cluster_size, args.cache_path)
        workflow_classify_singletons(args.train, args.pending, args.output, args.name_col, args.headerless, assignment, cfg)
    elif args.command == "merge-training":
        workflow_merge_training(args.train, args.output, args.threshold, args.topk, args.cache_path, cfg)


if __name__ == "__main__":
    main()
