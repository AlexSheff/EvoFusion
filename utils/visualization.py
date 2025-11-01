"""
Visualization utilities for EvoFusion.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import json
import os
import sympy as sp
from typing import Dict, List, Any, Optional


def plot_evolution_progress(evolution_history: Dict[str, List], 
                           save_path: Optional[str] = None):
    """
    Plot the progress of evolution across generations.
    
    Args:
        evolution_history: Dictionary with evolution history data
        save_path: Path to save the plot (if None, plot is displayed)
    """
    generations = evolution_history['generations']
    best_scores = evolution_history['best_scores']
    avg_scores = evolution_history['avg_scores']
    
    plt.figure(figsize=(12, 8))
    
    # Plot best and average scores
    plt.plot(generations, best_scores, 'b-', label='Best Score', linewidth=2)
    plt.plot(generations, avg_scores, 'r--', label='Average Score', linewidth=2)
    
    # Add annotations for best activation and loss functions
    for i, gen in enumerate(generations):
        if i % max(1, len(generations) // 5) == 0 or i == len(generations) - 1:
            plt.annotate(
                f"{evolution_history['best_activations'][i]}\n{evolution_history['best_losses'][i]}",
                (gen, best_scores[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
            )
    
    plt.xlabel('Generation')
    plt.ylabel('Score')
    plt.title('Evolution Progress')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_symbolic_function(expr_str: str, function_type: str = 'activation',
                               x_range: List[float] = [-5, 5], 
                               y_range: Optional[List[float]] = None,
                               save_path: Optional[str] = None):
    """
    Visualize a symbolic function.
    
    Args:
        expr_str: String representation of the symbolic expression
        function_type: Type of function ('activation' or 'loss')
        x_range: Range of x values to plot
        y_range: Range of y values to limit the plot (optional)
        save_path: Path to save the plot (if None, plot is displayed)
    """
    plt.figure(figsize=(10, 6))
    
    # Parse the expression
    x = sp.symbols('x')
    y = sp.symbols('y')
    
    try:
        if function_type == 'activation':
            # For activation functions, we plot f(x)
            expr = sp.sympify(expr_str)
            f = sp.lambdify(x, expr, "numpy")
            
            # Generate x values
            x_vals = np.linspace(x_range[0], x_range[1], 1000)
            
            # Calculate y values
            y_vals = f(x_vals)
            
            # Plot the function
            plt.plot(x_vals, y_vals)
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.title(f'Activation Function: {expr_str}')
            
        elif function_type == 'loss':
            # For loss functions, we create a heatmap
            expr = sp.sympify(expr_str)
            f = sp.lambdify((y, x), expr, "numpy")  # y is true, x is predicted
            
            # Generate grid of values
            x_vals = np.linspace(x_range[0], x_range[1], 100)
            y_vals = np.linspace(x_range[0], x_range[1], 100)
            X, Y = np.meshgrid(x_vals, y_vals)
            Z = f(Y, X)  # Y is true, X is predicted
            
            # Create heatmap
            plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
            plt.colorbar(label='Loss Value')
            plt.xlabel('Predicted Value')
            plt.ylabel('True Value')
            plt.title(f'Loss Function: {expr_str}')
            
        # Set y-axis limits if provided
        if y_range:
            plt.ylim(y_range)
            
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    except Exception as e:
        print(f"Error visualizing function: {str(e)}")
        plt.close()


def plot_history(history: List[Dict]):
    """Legacy function for evolution history visualization."""
    return plot_evolution_progress(history)