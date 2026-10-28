"""Export a saved checkpoint as a validated ONNX model and encoder bundle."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from dwp.ml.export import export_onnx


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-version")
    args = parser.parse_args(argv)
    print(export_onnx(args.checkpoint, args.output, model_version=args.model_version))


if __name__ == "__main__":
    main()
