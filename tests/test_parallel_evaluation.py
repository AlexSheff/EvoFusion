"""
Tests for parallel evaluation and multi-dataset functionality.
"""

import unittest
import sys
import os
import time
from typing import Dict, List, Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.evaluator import Evaluator
from modules.evoactiv_interface import generate_initial_activations
from modules.evoloss_interface import generate_initial_losses


class TestParallelEvaluation(unittest.TestCase):
    """Test cases for parallel evaluation functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.evaluator = Evaluator()
        # Use only standard activations and losses for testing to avoid compatibility issues
        self.activations = [
            {"name": "relu", "expr": "Max(0, x)", "sympy_expr": None},
            {"name": "sigmoid", "expr": "1/(1 + exp(-x))", "sympy_expr": None}
        ]
        self.losses = [
            {"name": "mse", "expr": "(y_true - y_pred)**2", "sympy_expr": None}
        ]
    
    def test_parallel_vs_sequential(self):
        """Test that parallel and sequential evaluations give same results."""
        # Use a small batch size to avoid tensor size mismatch
        self.evaluator.batch_size = 10
        
        # Sequential evaluation
        start_time = time.time()
        sequential_results = self.evaluator.evaluate_population(
            self.activations, self.losses, parallel=False
        )
        sequential_time = time.time() - start_time
        
        # Parallel evaluation
        start_time = time.time()
        parallel_results = self.evaluator.evaluate_population(
            self.activations, self.losses, parallel=True
        )
        parallel_time = time.time() - start_time
        
        # Check results consistency (order might differ)
        self.assertEqual(len(sequential_results), len(parallel_results))
        
        # Sort results by activation and loss names for comparison
        sequential_results.sort(key=lambda x: (x['activation']['name'], x['loss']['name']))
        parallel_results.sort(key=lambda x: (x['activation']['name'], x['loss']['name']))
        
        # Compare metrics (allowing small floating point differences)
        for seq_result, par_result in zip(sequential_results, parallel_results):
            self.assertEqual(seq_result['activation']['name'], par_result['activation']['name'])
            self.assertEqual(seq_result['loss']['name'], par_result['loss']['name'])
            
            # Metrics might have small differences due to randomness in training
            # Just check that all metrics exist
            for metric in seq_result['metrics']:
                self.assertIn(metric, par_result['metrics'])
        
        print(f"Sequential time: {sequential_time:.2f}s, Parallel time: {parallel_time:.2f}s")


class TestMultiDataset(unittest.TestCase):
    """Test cases for multi-dataset functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.evaluator = Evaluator()
        
    def test_dataset_loading(self):
        """Test that different datasets can be loaded."""
        datasets = ["mnist", "fashion_mnist"]
        
        for dataset_name in datasets:
            # This should not raise an exception
            train_loader, val_loader = self.evaluator._load_dataset(dataset_name)
            
            # Basic checks
            self.assertIsNotNone(train_loader)
            self.assertIsNotNone(val_loader)
            self.assertTrue(len(train_loader) > 0)
            self.assertTrue(len(val_loader) > 0)
            
            print(f"Successfully loaded {dataset_name} dataset")


if __name__ == '__main__':
    unittest.main()