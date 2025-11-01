"""
FusionEngine - main co-evolution cycle for activations (EvoActiv) and loss functions (EvoLoss).

Tasks:
- Initialization of activation and loss populations
- Joint evaluation of (activation, loss) pairs
- Selection, crossover and mutations
- Logging and saving best results
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from utils.reproducibility import get_env_info
from utils.visualization import plot_evolution_progress


@dataclass
class FusionConfig:
    generations: int
    activ_population_size: int
    loss_population_size: int
    metrics: List[str]
    results_dir: str


class FusionEngine:
    def __init__(self, config: FusionConfig, evaluator=None):
        self.config = config
        from .population_manager import PopulationManager
        from .evaluator import Evaluator
        self.population_manager = PopulationManager(
            activ_population_size=config.activ_population_size,
            loss_population_size=config.loss_population_size,
        )
        self.evaluator = evaluator if evaluator is not None else Evaluator(metrics=config.metrics)

        # Ensure results directories exist
        os.makedirs(os.path.join(config.results_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(config.results_dir, "best_models"), exist_ok=True)
        os.makedirs(os.path.join(config.results_dir, "reports"), exist_ok=True)

    def run(self) -> Dict[str, Any]:
        """Runs the coevolution cycle and returns a brief final report."""
        self.population_manager.initialize_populations()

        best_pair: Tuple[Dict[str, Any], Dict[str, Any]] | None = None
        best_fitness: float = float("-inf")
        history: List[Dict[str, Any]] = []

        evolution_scores: List[float] = []
        average_scores: List[float] = []
        best_acts: List[str] = []
        best_losses: List[str] = []

        for gen in range(self.config.generations):
            pairs = self.population_manager.enumerate_pairs()

            gen_log = self.evaluator.evaluate_population(self.population_manager.activations, self.population_manager.losses)

            for result in gen_log:
                if "failed" in result and result["failed"]:
                    continue
                # Resolve score from fitness or metrics
                score = None
                if "fitness" in result:
                    score = result["fitness"].get("score")
                if score is None and "metrics" in result:
                    score = result["metrics"].get("score")
                if score is None:
                    continue
                if score > best_fitness:
                    best_fitness = score
                    best_pair = (result["activation"], result["loss"]) 

            # Selection and evolution
            self.population_manager.evolve(gen_log)

            # Generation log
            history.append({
                "generation": gen,
                "results": gen_log,
                "best_fitness": best_fitness,
            })

            # Aggregate per-generation stats for visualization
            gen_scores = []
            gen_best_name = (None, None)
            for res in gen_log:
                s = None
                if "fitness" in res:
                    s = res["fitness"].get("score")
                if s is None and "metrics" in res:
                    s = res["metrics"].get("score")
                if s is not None:
                    gen_scores.append(float(s))
                    if s == best_fitness:
                        gen_best_name = (res["activation"].get("name"), res["loss"].get("name"))
            evolution_scores.append(best_fitness if best_fitness != float("-inf") else 0.0)
            average_scores.append(sum(gen_scores) / len(gen_scores) if gen_scores else 0.0)
            best_acts.append(gen_best_name[0] or "")
            best_losses.append(gen_best_name[1] or "")

        # Save report
        report_path = os.path.join(self.config.results_dir, "reports", "final_report.json")
        config_snapshot = {
            "generations": self.config.generations,
            "activ_population_size": self.config.activ_population_size,
            "loss_population_size": self.config.loss_population_size,
            "metrics": self.config.metrics,
            "results_dir": self.config.results_dir,
        }
        env_info = get_env_info()

        # Save evolution progress plot
        try:
            progress_path = os.path.join(self.config.results_dir, "reports", "evolution_progress.png")
            plot_evolution_progress({
                "generations": list(range(self.config.generations)),
                "best_scores": evolution_scores,
                "avg_scores": average_scores,
                "best_activations": best_acts,
                "best_losses": best_losses,
            }, save_path=progress_path)
        except Exception:
            progress_path = None

        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            elif hasattr(obj, '__str__'):
                return str(obj)
            else:
                return obj

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(convert_to_serializable({
                "best_pair": best_pair,
                "best_fitness": best_fitness,
                "history_size": len(history),
                "config": config_snapshot,
                "env": env_info,
                "progress_plot": progress_path,
            }), f, ensure_ascii=False, indent=2)

        return {
            "best_pair": best_pair,
            "best_fitness": best_fitness,
            "report_path": report_path,
        }