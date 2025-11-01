# EvoFusion Architecture

## Project Structure

- core/
  - fusion_engine.py — co-evolution control cycle
  - evaluator.py — evaluation of activation-loss pairs
  - crossover_mutation.py — evolution operators for symbolic expressions
  - population_manager.py — population management
- modules/
  - evoactiv_interface.py — interface for symbolic activation function generation
  - evoloss_interface.py — interface for symbolic loss function generation
- experiments/ — example runs
- results/ — logs, best models, reports
- utils/ — visualization, metrics, config loading
- configs/ — YAML configurations
- tests/ — unit and integration tests

## Core Mechanics

In each generation, (activation, loss) pairs receive fitness scores, the best ones undergo selection, crossover, and mutations.

## Symbolic Function Representation

EvoFusion now uses symbolic representation (SymPy) for both activation and loss functions:

1. **Symbolic Expression Trees**:
   - Functions are represented as expression trees
   - Allows for complex mathematical operations
   - Supports automatic differentiation

2. **Dynamic Generation**:
   - Random generation of diverse initial populations
   - Templates and operations for creating novel functions
   - Ensures mathematical validity of expressions

3. **Tree-Based Genetic Operations**:
   - Subtree crossover: swaps subtrees between parent expressions
   - Mutation operations:
     - Replace subtree
     - Change operator
     - Add new operation
   - Parameter mutation for fine-tuning

4. **Benefits**:
   - Unlimited function space exploration
   - Novel activation/loss functions discovery
   - Interpretable mathematical expressions
   - Easy conversion to executable code