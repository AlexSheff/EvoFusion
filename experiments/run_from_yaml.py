"""
Run EvoFusion using a YAML configuration file.
"""

from __future__ import annotations

from pathlib import Path

from utils.config_loader import load_config
from utils.reproducibility import set_seed, get_env_info
from core.fusion_engine import FusionEngine, FusionConfig
from core.evaluator import Evaluator


def main(config_path: str = "configs/example_custom.yaml"):
    project_root = Path(__file__).resolve().parents[1]
    set_seed(42)
    print("Environment:", get_env_info())

    cfg = load_config(project_root / config_path)

    fusion_cfg = FusionConfig(**cfg["fusion"])

    evaluator = Evaluator(
        metrics=fusion_cfg.metrics,
        dataset_config=cfg.get("dataset"),
        save_model_dir=str(project_root / "results" / "best_models"),
    )

    engine = FusionEngine(fusion_cfg, evaluator=evaluator)
    summary = engine.run()

    print("Best:", summary.get("best_pair"))
    print("Report:", summary.get("report_path"))


if __name__ == "__main__":
    main()