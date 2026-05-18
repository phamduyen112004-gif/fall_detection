#!/usr/bin/env python3
"""
Fall Detection Project — Single Entry Point.
========================================

A unified CLI for the fall detection pipeline supporting:
    - Preprocessing: Convert raw videos to .npy feature matrices
    - Training: Train the HybridFallTransformer model
    - Evaluation: Benchmark trained model with metrics and plots
    - Application: Launch the real-time PyQt5 GUI

Environment Variables (optional):
    TELEGRAM_BOT_TOKEN    Telegram bot token for alerts
    TELEGRAM_CHAT_ID     Telegram chat ID for alerts

Examples:
    python main.py --mode preprocess   # Preprocess datasets
    python main.py --mode train       # Train model
    python main.py --mode evaluate    # Benchmark model
    python main.py --mode app        # Launch GUI
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import traceback
from typing import Dict, Tuple, Optional


# =============================================================================
# MODULE-LEVEL LOGGING
# =============================================================================

def _setup_module_logger(name: str) -> logging.Logger:
    """
    Configure and return a module-level logger.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


_logger: logging.Logger = _setup_module_logger(__name__)


# =============================================================================
# MODE REGISTRY
# =============================================================================

_ModeEntry = Tuple[str, str, str]
_MODES: Dict[str, _ModeEntry] = {
    "preprocess": (
        "Convert raw videos (CaucaFall, MCFD) into preprocessed .npy matrices.",
        "src.data_prep",
        "run_preprocessing",
    ),
    "train": (
        "Train the HybridFallTransformer with online augmentation.",
        "src.trainer",
        "run_training",
    ),
    "evaluate": (
        "Run full benchmark: accuracy metrics, GFLOPs, FPS, plots.",
        "src.evaluator",
        "run_evaluation",
    ),
    "app": (
        "Launch the PyQt5 real-time fall detection GUI.",
        "src.gui_app",
        "run_app",
    ),
}


def _build_parser() -> argparse.ArgumentParser:
    """
    Build and return the argument parser for the CLI.

    Returns:
        Configured ArgumentParser with all CLI arguments.
    """
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Fall Detection: YOLOv11n-Pose + PIFR + Transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --mode preprocess   # Preprocess datasets\n"
            "  python main.py --mode train       # Train model\n"
            "  python main.py --mode evaluate    # Benchmark model\n"
            "  python main.py --mode app         # Launch GUI\n"
        ),
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=list(_MODES.keys()),
        required=True,
        help="Execution mode. See descriptions below.",
    )
    return parser


def main() -> int:
    """
    Main entry point for the CLI application.

    Parses arguments, dispatches to the appropriate module based on mode,
    and handles all execution errors gracefully.

    Returns:
        Exit code: 0 for success, 1 for errors, 2 for user interruption.
    """
    parser: argparse.ArgumentParser = _build_parser()

    try:
        args: argparse.Namespace = parser.parse_args()

    except SystemExit as e:
        # argparse exits on --help or errors
        return e.code if e.code is not None else 1

    except Exception as e:
        _logger.error("Failed to parse arguments: %s", e)
        return 1

    mode_key: str = args.mode
    description: str
    module_name: str
    entry_func_name: str
    description, module_name, entry_func_name = _MODES[mode_key]

    _logger.info("=" * 60)
    _logger.info("Mode: %s", mode_key)
    _logger.info("=" * 60)
    _logger.info("  %s", description)

    try:
        # GUI mode: run directly without subprocess (avoids PyQt5 fork issues)
        if mode_key == "app":
            from src.gui_app import run_app
            exit_code: int = run_app()
            return exit_code

        # Other modes: import and call entry function dynamically
        module = importlib.import_module(module_name)
        entry_func: Optional[callable] = getattr(module, entry_func_name, None)

        if entry_func is None:
            _logger.error(
                "Module '%s' has no '%s' function",
                module_name,
                entry_func_name
            )
            return 1

        # Call the entry function
        result: Optional[object] = entry_func()

        # Log completion with optional result
        if result is not None:
            _logger.info("%s completed with result: %s", entry_func_name, result)
        else:
            _logger.info("%s completed successfully", entry_func_name)

        return 0

    except ImportError as e:
        _logger.error("Import error: %s", e)
        _logger.error("Make sure all dependencies are installed:")
        _logger.error("  pip install -r requirements.txt")
        return 1

    except KeyboardInterrupt:
        _logger.warning("Interrupted by user.")
        return 2

    except SystemExit as e:
        # Propagate intentional sys.exit() calls from submodules
        return e.code if e.code is not None else 0

    except Exception as e:
        _logger.error("Unexpected error: %s", e)
        _logger.debug("Traceback:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
