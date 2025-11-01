from core.fusion_engine import FusionEngine, FusionConfig
from utils.reproducibility import set_seed, get_env_info
from core.evaluator import Evaluator


def main():
    # Fix seed and log environment
    set_seed(42)
    print("Environment:", get_env_info())
    config = FusionConfig(
        generations=5,
        activ_population_size=3,
        loss_population_size=3,
        metrics=["accuracy", "f1", "convergence"],
        results_dir="results",
    )

    # Configure dataset and artifact saving
    dataset_cfg = {
        "name": "mnist",
        "root": "./data",
        "batch_size": 64,
        "download": True,
    }
    evaluator = Evaluator(metrics=config.metrics, dataset_config=dataset_cfg, save_model_dir="results/best_models")
    engine = FusionEngine(config, evaluator=evaluator)
    report = engine.run()

    best = report.get("best_pair")
    print("Best fitness:", report.get("best_fitness"))
    if best:
        act, loss = best
        print("Best activation:", act.get("name"), act.get("expr"))
        print("Best loss:", loss.get("name"), loss.get("expr"))
    print("Report saved to:", report.get("report_path"))


if __name__ == "__main__":
    main()