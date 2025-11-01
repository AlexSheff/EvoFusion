"""
Test script for symbolic function generation, crossover, and mutation.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import random
import sympy as sp
from modules.evoactiv_interface import generate_initial_activations, x
from modules.evoloss_interface import generate_initial_losses, y_true, y_pred
from core.crossover_mutation import crossover_entities, mutate_entity

def test_activation_generation():
    """Test generation of activation functions."""
    print("\n=== Testing Activation Function Generation ===")
    
    # Generate a population of activation functions
    population_size = 5
    activations = generate_initial_activations(population_size)
    
    print(f"Generated {len(activations)} activation functions:")
    for i, activation in enumerate(activations):
        print(f"{i+1}. {activation['name']}: {activation['expr']}")
    
    return activations

def test_loss_generation():
    """Test generation of loss functions."""
    print("\n=== Testing Loss Function Generation ===")
    
    # Generate a population of loss functions
    population_size = 5
    losses = generate_initial_losses(population_size)
    
    print(f"Generated {len(losses)} loss functions:")
    for i, loss in enumerate(losses):
        print(f"{i+1}. {loss['name']}: {loss['expr']}")
    
    return losses

def test_crossover(entities):
    """Test crossover operation on entities."""
    print("\n=== Testing Crossover Operation ===")
    
    if len(entities) < 2:
        print("Need at least 2 entities for crossover test")
        return
    
    # Select two random entities
    entity1 = random.choice(entities)
    entity2 = random.choice([e for e in entities if e != entity1])
    
    print(f"Parent 1: {entity1['name']} - {entity1['expr']}")
    print(f"Parent 2: {entity2['name']} - {entity2['expr']}")
    
    # Perform crossover
    child1, child2 = crossover_entities(entity1, entity2)
    
    print(f"Child 1: {child1['name']} - {child1['expr']}")
    print(f"Child 2: {child2['name']} - {child2['expr']}")
    
    return child1, child2

def test_mutation(entity):
    """Test mutation operation on an entity."""
    print("\n=== Testing Mutation Operation ===")
    
    print(f"Original: {entity['name']} - {entity['expr']}")
    
    # Perform mutation
    mutated = mutate_entity(entity, mutation_rate=1.0)  # Force mutation
    
    print(f"Mutated: {mutated['name']} - {mutated['expr']}")
    
    return mutated

def main():
    """Run all tests."""
    random.seed(42)  # For reproducibility
    
    print("TESTING SYMBOLIC FUNCTION GENERATION, CROSSOVER, AND MUTATION")
    print("=" * 60)
    
    # Test activation functions
    activations = test_activation_generation()
    
    # Test loss functions
    losses = test_loss_generation()
    
    # Test crossover on activations
    child_activations = test_crossover(activations)
    
    # Test crossover on losses
    child_losses = test_crossover(losses)
    
    # Test mutation on an activation function
    mutated_activation = test_mutation(activations[0])
    
    # Test mutation on a loss function
    mutated_loss = test_mutation(losses[0])
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    main()