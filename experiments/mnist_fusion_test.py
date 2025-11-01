"""
Example of running co-evolution on MNIST with real training.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from core.fusion_engine import FusionEngine, FusionConfig
from core.evaluator import Evaluator


def main():
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    
    # Create configuration with smaller populations for quick testing
    config = FusionConfig(
        generations=2,  # Reduced for quick testing
        activ_population_size=3,
        loss_population_size=3,
        metrics=["accuracy", "f1", "convergence"],
        results_dir=str(results_dir),
    )

    # Create Evaluator instance with settings for quick testing
    evaluator = Evaluator()
    evaluator.metrics = {"accuracy": evaluator._calculate_accuracy, "f1_score": evaluator._calculate_f1_score}

    
    # Test a single activation-loss pair to verify functionality
    print("Testing a single activation-loss pair...")
    test_activation = {
        "name": "relu",
        "params": {}
    }
    test_loss = {
        "name": "cross_entropy",
        "params": {}
    }
    
    start_time = time.time()
    result = evaluator.evaluate_pair(
        activation=test_activation,
        loss=test_loss,
        epochs=1  # For quick testing
    )
    print(f"Test completed in {time.time() - start_time:.2f} sec.")
    print(f"Results: {result}")
    
    # Launch the full co-evolution process
    print("\nStarting full co-evolution process...")
    engine = FusionEngine(config, evaluator=evaluator)
    summary = engine.run()
    print("Summary:", summary)


if __name__ == "__main__":
    main()