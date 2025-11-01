"""
PopulationManager - management of activation and loss function populations.
Creation, pair enumeration, evolution (selection, crossover, mutations).
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

from .crossover_mutation import crossover_entities, mutate_entity
# Using absolute imports instead of relative imports
from modules.evoactiv_interface import generate_initial_activations
from modules.evoloss_interface import generate_initial_losses


class PopulationManager:
    def __init__(self, activ_population_size: int, loss_population_size: int):
        self.activ_population_size = activ_population_size
        self.loss_population_size = loss_population_size
        self.activations: List[Dict[str, Any]] = []
        self.losses: List[Dict[str, Any]] = []

    def initialize_populations(self) -> None:
        self.activations = generate_initial_activations(self.activ_population_size)
        self.losses = generate_initial_losses(self.loss_population_size)

    def enumerate_pairs(self) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for a in self.activations:
            for l in self.losses:
                pairs.append((a, l))
        random.shuffle(pairs)
        return pairs

    def evolve(self, gen_log: List[Dict[str, Any]]) -> None:
        """
        Simple algorithm:
        - sort results by score
        - select top 50%
        - crossover and mutation to restore population size
        """
        # Selection for activations
        successful_results = [r for r in gen_log if not ("failed" in r and r["failed"])]

        def extract_score(r: Dict[str, Any]) -> float:
            # Prefer fitness.score, fall back to metrics.score
            if "fitness" in r and isinstance(r["fitness"], dict):
                s = r["fitness"].get("score")
                if isinstance(s, (int, float)):
                    return float(s)
            if "metrics" in r and isinstance(r["metrics"], dict):
                s = r["metrics"].get("score")
                if isinstance(s, (int, float)):
                    return float(s)
            return 0.0

        top_pairs = sorted(successful_results, key=extract_score, reverse=True)
        if not top_pairs:
            return

        top_acts = [p["activation"] for p in top_pairs[: max(1, len(self.activations) // 2)]]
        top_losses = [p["loss"] for p in top_pairs[: max(1, len(self.losses) // 2)]]

        # Restore activation population
        new_acts: List[Dict[str, Any]] = top_acts.copy()
        while len(new_acts) < self.activ_population_size:
            a, b = random.sample(top_acts, 2) if len(top_acts) >= 2 else (top_acts[0], top_acts[0])
            child1, child2 = crossover_entities(a, b)
            new_acts.append(mutate_entity(child1, mutation_rate=0.2))
            if len(new_acts) < self.activ_population_size:
                new_acts.append(mutate_entity(child2, mutation_rate=0.2))
        self.activations = new_acts

        # Restore loss population
        new_losses: List[Dict[str, Any]] = top_losses.copy()
        while len(new_losses) < self.loss_population_size:
            a, b = random.sample(top_losses, 2) if len(top_losses) >= 2 else (top_losses[0], top_losses[0])
            child1, child2 = crossover_entities(a, b)
            new_losses.append(mutate_entity(child1, mutation_rate=0.2))
            if len(new_losses) < self.loss_population_size:
                new_losses.append(mutate_entity(child2, mutation_rate=0.2))
        self.losses = new_losses