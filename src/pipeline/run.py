"""Sequential orchestration without changing the existing command contracts."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline.config import PreprocessingConfig, load_config
from src.pipeline.registry import ModelRegistry


def _run_preprocessing(config: PreprocessingConfig) -> int:
    if config.extraction is not None and config.extraction.run:
        from src.preprocessing.parameters.run import run as extract

        stage = config.extraction
        result = extract(
            argparse.Namespace(
                input=list(stage.inputs),
                config=stage.config,
                output_dir=stage.output_dir,
            )
        )
        if result:
            return result
    if config.plotting is not None and config.plotting.run:
        from src.preprocessing.plotting.run import run as plot

        result = plot(argparse.Namespace(config=config.plotting.config))
        if result:
            return result
    if config.inputs_model is not None and config.inputs_model.run:
        from src.preprocessing.inputsModel.run import run as build_inputs

        stage_inputs = config.inputs_model
        result = build_inputs(
            argparse.Namespace(
                config=stage_inputs.config,
                unlabeled=stage_inputs.unlabeled,
            )
        )
        if result:
            return result
    return 0


def run_pipeline(config_path: str | Path, registry: ModelRegistry | None = None) -> int:
    """Validate every XML entry, then run enabled stages and models in order.

    Enabled model INIs are preflighted, disabled references are not opened.
    Clinical INIs are left to their stages. Matrix existence is checked only
    after preprocessing, immediately before that model's handler is invoked.
    """
    config = load_config(config_path, registry)
    if config.preprocessing is not None:
        stage_result = _run_preprocessing(config.preprocessing)
        if stage_result:
            return stage_result
    for step in config.models:
        if not step.config.run:
            continue
        if not step.config.input_matrix.is_file():
            raise ValueError(f"Input matrix is not a file for model {step.config.id}")
        result = step.execute()
        if result:
            return result
    return 0


def run(args: argparse.Namespace) -> int:
    return run_pipeline(args.config)


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "pipeline", help="Run enabled steps from an XML configuration"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/pipeline.xml"),
        help="Pipeline XML; paths in XML are relative to this file.",
    )
    parser.set_defaults(handler=run)
