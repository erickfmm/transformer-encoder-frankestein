from __future__ import annotations

import argparse


def _run_train(args: argparse.Namespace) -> int:
    from .training.main import main as train_main

    argv = [
        "--config-name",
        args.config_name,
        "--device",
        args.device,
    ]
    if args.config:
        argv.extend(["--config", args.config])
    if args.list_configs:
        argv.append("--list-configs")
    if args.batch_size is not None:
        argv.extend(["--batch-size", str(args.batch_size)])
    if args.model_mode:
        argv.extend(["--model-mode", args.model_mode])
    result = train_main(argv)
    return int(result) if isinstance(result, int) else 0


def _run_deploy(args: argparse.Namespace) -> int:
    from .deploy.deploy import main as deploy_main

    argv = [
        "--checkpoint",
        args.checkpoint,
        "--output",
        args.output,
        "--format",
        args.format,
        "--device",
        args.device,
    ]
    if args.validate:
        argv.append("--validate")
    if args.config:
        argv.extend(["--config", args.config])
    result = deploy_main(argv)
    return int(result) if isinstance(result, int) else 0


def _run_quantize(args: argparse.Namespace) -> int:
    from .deploy.deploy import main as deploy_main

    argv = [
        "--checkpoint",
        args.checkpoint,
        "--output",
        args.output,
        "--format",
        "quantized",
        "--device",
        args.device,
    ]
    if args.validate:
        argv.append("--validate")
    if args.config:
        argv.extend(["--config", args.config])
    result = deploy_main(argv)
    return int(result) if isinstance(result, int) else 0


def _run_infer(args: argparse.Namespace) -> int:
    from .deploy.inference import main as infer_main

    argv = [
        "--model",
        args.model,
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
    ]
    if args.text:
        argv.extend(["--text", args.text])
    if args.input:
        argv.extend(["--input", args.input])
    if args.output:
        argv.extend(["--output", args.output])
    if args.fp16:
        argv.append("--fp16")
    if args.benchmark:
        argv.append("--benchmark")
    result = infer_main(argv)
    return int(result) if isinstance(result, int) else 0


def _run_sbert_train(args: argparse.Namespace) -> int:
    from .sbert.train_sbert import main as sbert_train_main

    argv = [
        "--output_dir",
        args.output_dir,
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--learning_rate",
        str(args.learning_rate),
        "--max_eval_samples",
        str(args.max_eval_samples),
        "--hidden_size",
        str(args.hidden_size),
        "--num_layers",
        str(args.num_layers),
        "--pooling_mode",
        args.pooling_mode,
        "--resample_std",
        str(args.resample_std),
        "--device",
        args.device,
    ]
    if args.pretrained:
        argv.extend(["--pretrained", args.pretrained])
    if args.max_train_samples is not None:
        argv.extend(["--max_train_samples", str(args.max_train_samples)])
    if args.no_amp:
        argv.append("--no_amp")
    if args.no_resample:
        argv.append("--no_resample")
    result = sbert_train_main(argv)
    return int(result) if isinstance(result, int) else 0


def _run_sbert_infer(args: argparse.Namespace) -> int:
    from .sbert.inference_sbert import main as sbert_infer_main

    argv = [
        "--model_path",
        args.model_path,
        "--mode",
        args.mode,
        "--batch_size",
        str(args.batch_size),
        "--device",
        args.device,
        "--top_k",
        str(args.top_k),
        "--n_clusters",
        str(args.n_clusters),
    ]
    if args.sentence1:
        argv.extend(["--sentence1", args.sentence1])
    if args.sentence2:
        argv.extend(["--sentence2", args.sentence2])
    if args.query:
        argv.extend(["--query", args.query])
    if args.corpus_file:
        argv.extend(["--corpus_file", args.corpus_file])
    if args.sentences_file:
        argv.extend(["--sentences_file", args.sentences_file])
    if args.input_file:
        argv.extend(["--input_file", args.input_file])
    if args.output_file:
        argv.extend(["--output_file", args.output_file])
    result = sbert_infer_main(argv)
    return int(result) if isinstance(result, int) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frankestein-transformer",
        description="Configurable training library and CLI for Transformer Encoder Frankenstein",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run main training")
    train_parser.add_argument("--config", type=str, default=None)
    train_parser.add_argument("--config-name", type=str, default="mini")
    train_parser.add_argument("--list-configs", action="store_true")
    train_parser.add_argument("--batch-size", type=int, default=None)
    train_parser.add_argument("--model-mode", choices=["frankenstein", "mini"], default=None)
    train_parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    train_parser.set_defaults(func=_run_train)

    deploy_parser = subparsers.add_parser("deploy", help="Convert checkpoint to deployment artifacts")
    deploy_parser.add_argument("--checkpoint", type=str, required=True)
    deploy_parser.add_argument("--output", type=str, required=True)
    deploy_parser.add_argument("--format", type=str, choices=["quantized", "standard"], default="quantized")
    deploy_parser.add_argument("--validate", action="store_true")
    deploy_parser.add_argument("--config", type=str, default=None)
    deploy_parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    deploy_parser.set_defaults(func=_run_deploy)

    quantize_parser = subparsers.add_parser("quantize", help="Export checkpoint in quantized deployment format")
    quantize_parser.add_argument("--checkpoint", type=str, required=True)
    quantize_parser.add_argument("--output", type=str, required=True)
    quantize_parser.add_argument("--validate", action="store_true")
    quantize_parser.add_argument("--config", type=str, default=None)
    quantize_parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    quantize_parser.set_defaults(func=_run_quantize)

    infer_parser = subparsers.add_parser("infer", help="Run deployed model inference")
    infer_parser.add_argument("--model", type=str, required=True)
    infer_parser.add_argument("--text", type=str, default=None)
    infer_parser.add_argument("--input", type=str, default=None)
    infer_parser.add_argument("--output", type=str, default=None)
    infer_parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    infer_parser.add_argument("--fp16", action="store_true")
    infer_parser.add_argument("--batch-size", type=int, default=8)
    infer_parser.add_argument("--benchmark", action="store_true")
    infer_parser.set_defaults(func=_run_infer)

    sbert_train_parser = subparsers.add_parser("sbert-train", help="Train SBERT model")
    sbert_train_parser.add_argument("--pretrained", type=str, default=None)
    sbert_train_parser.add_argument("--output_dir", type=str, default="./output/sbert_tormented_v2")
    sbert_train_parser.add_argument("--batch_size", type=int, default=16)
    sbert_train_parser.add_argument("--epochs", type=int, default=4)
    sbert_train_parser.add_argument("--learning_rate", type=float, default=2e-5)
    sbert_train_parser.add_argument("--max_train_samples", type=int, default=None)
    sbert_train_parser.add_argument("--max_eval_samples", type=int, default=10000)
    sbert_train_parser.add_argument("--hidden_size", type=int, default=768)
    sbert_train_parser.add_argument("--num_layers", type=int, default=12)
    sbert_train_parser.add_argument("--pooling_mode", choices=["mean", "cls", "max"], default="mean")
    sbert_train_parser.add_argument("--no_amp", action="store_true")
    sbert_train_parser.add_argument("--no_resample", action="store_true")
    sbert_train_parser.add_argument("--resample_std", type=float, default=0.3)
    sbert_train_parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    sbert_train_parser.set_defaults(func=_run_sbert_train)

    sbert_infer_parser = subparsers.add_parser("sbert-infer", help="Run SBERT inference tasks")
    sbert_infer_parser.add_argument("--model_path", type=str, required=True)
    sbert_infer_parser.add_argument("--mode", choices=["similarity", "search", "cluster", "encode"], required=True)
    sbert_infer_parser.add_argument("--sentence1", type=str, default=None)
    sbert_infer_parser.add_argument("--sentence2", type=str, default=None)
    sbert_infer_parser.add_argument("--query", type=str, default=None)
    sbert_infer_parser.add_argument("--corpus_file", type=str, default=None)
    sbert_infer_parser.add_argument("--top_k", type=int, default=5)
    sbert_infer_parser.add_argument("--sentences_file", type=str, default=None)
    sbert_infer_parser.add_argument("--n_clusters", type=int, default=5)
    sbert_infer_parser.add_argument("--input_file", type=str, default=None)
    sbert_infer_parser.add_argument("--output_file", type=str, default=None)
    sbert_infer_parser.add_argument("--batch_size", type=int, default=32)
    sbert_infer_parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    sbert_infer_parser.set_defaults(func=_run_sbert_infer)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
