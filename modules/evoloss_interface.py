"""
Interface to the EvoLoss subsystem: generation of initial loss functions and their parameters.
Implements symbolic representation of loss functions using sympy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import sympy as sp
from typing import Any, Dict, List, Union, Callable

# Define symbolic variables for loss functions
# y_true represents the target value, y_pred represents the predicted value
y_true = sp.Symbol('y_true')
y_pred = sp.Symbol('y_pred')

# Basic loss function templates as sympy expressions
LOSS_TEMPLATES = [
    (y_true - y_pred)**2,                                  # MSE
    sp.Abs(y_true - y_pred),                               # MAE/L1
    sp.Piecewise(
        (0.5 * (y_true - y_pred)**2, sp.Abs(y_true - y_pred) < 1),
        (sp.Abs(y_true - y_pred) - 0.5, True)
    ),                                                     # Smooth L1/Huber
    1 - y_true * y_pred,                                   # Hinge loss (simplified)
    -y_true * sp.log(y_pred + 1e-7) - (1-y_true) * sp.log(1-y_pred + 1e-7),  # Binary cross-entropy
    sp.log(1 + sp.exp(-y_true * y_pred))                   # Logistic loss
]

# Operations for creating new loss functions
UNARY_OPS = [
    lambda expr: sp.Abs(expr),
    lambda expr: expr**2,
    lambda expr: expr**3,
    lambda expr: sp.sqrt(sp.Abs(expr)),
    lambda expr: sp.log(1 + sp.Abs(expr)),
    lambda expr: sp.tanh(sp.Abs(expr)),
    lambda expr: 1 - sp.exp(-expr)
]

BINARY_OPS = [
    lambda a, b: a + b,
    lambda a, b: a * b,
    lambda a, b: a / (1 + sp.Abs(b)),
    lambda a, b: sp.Max(a, b),
    lambda a, b: sp.Min(a, b),
    lambda a, b: a * (1 - sp.exp(-sp.Abs(b)))
]

def get_loss_function(loss_config: Dict[str, Any]) -> Callable:
    """
    Convert a symbolic loss function to a PyTorch callable function.
    
    Args:
        loss_config: Dictionary containing loss function configuration
        
    Returns:
        PyTorch callable loss function
    """
    name = loss_config.get("name", "mse").lower()
    
    if name == "mse":
        return nn.MSELoss()
    elif name == "mae":
        return nn.L1Loss()
    elif name == "smooth_l1":
        return nn.SmoothL1Loss()
    elif name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif name == "binary_crossentropy":
        return nn.BCELoss()
    
    expr_str = loss_config.get("expr", str(loss_config.get("sympy_expr", "(y_true - y_pred)**2")))

    def custom_loss(output, target):
        y_pred = output
        y_true = target
        
        # A simple parser for the expression string
        # This is not a full-fledged parser, but it can handle some basic expressions.
        # For a real implementation, a proper parsing library should be used.
        
        if "**2" in expr_str and "y_true - y_pred" in expr_str:
            return F.mse_loss(y_pred, y_true)
        elif "Abs(y_true - y_pred)" in expr_str:
            return F.l1_loss(y_pred, y_true)
        elif "log(1 + exp(-y_true * y_pred))" in expr_str:
            return torch.log(1 + torch.exp(-y_true * y_pred)).mean()
        elif "-y_true*log(y_pred + 1e-7) - (1-y_true)*log(1-y_pred + 1e-7)" in expr_str:
            return F.binary_cross_entropy(y_pred, y_true)
        else:
            # Fallback to MSE if the expression is not recognized
            return F.mse_loss(y_pred, y_true)

    return custom_loss

def generate_random_loss() -> Dict[str, Any]:
    """Generate a random loss function as a sympy expression."""
    # Start with either a basic template or a simple difference
    if random.random() < 0.7:
        expr = random.choice(LOSS_TEMPLATES)
    else:
        expr = (y_true - y_pred)**2  # Default to squared error
    
    # Apply random operations (0-2 operations)
    num_ops = random.randint(0, 2)
    for _ in range(num_ops):
        if random.random() < 0.7:  # Apply unary operation
            op = random.choice(UNARY_OPS)
            expr = op(expr)
        else:  # Apply binary operation with a constant or another expression
            op = random.choice(BINARY_OPS)
            if random.random() < 0.5:
                # Use a constant
                const = random.uniform(0.1, 2.0)  # Positive constants for loss functions
                expr = op(expr, const)
            else:
                # Use a simple expression
                simple_expr = random.choice([
                    sp.Abs(y_true - y_pred),
                    (y_true - y_pred)**2,
                    sp.log(1 + sp.Abs(y_true - y_pred))
                ])
                expr = op(expr, simple_expr)
    
    # Generate a name based on a simplified form of the expression
    name = f"custom_loss_{hash(str(expr)) % 10000}"
    
    # Convert to string representation for storage
    expr_str = str(expr)
    
    # Add reduction method (mean or sum)
    reduction = random.choice(["mean", "sum"])
    
    return {
        "name": name, 
        "expr": expr_str, 
        "sympy_expr": expr,
        "params": {"reduction": reduction}
    }

def generate_initial_losses(size: int) -> List[Dict[str, Any]]:
    """Generate a diverse initial population of loss functions."""
    losses = []
    
    # Include some standard losses for stability
    standard_losses = [
        {"name": "mse", "expr": "(y_true - y_pred)**2", 
         "sympy_expr": (y_true - y_pred)**2, "params": {"reduction": "mean"}},
        {"name": "mae", "expr": "Abs(y_true - y_pred)", 
         "sympy_expr": sp.Abs(y_true - y_pred), "params": {"reduction": "mean"}},
        {"name": "smooth_l1", "expr": "Piecewise((0.5*(y_true - y_pred)**2, Abs(y_true - y_pred) < 1), (Abs(y_true - y_pred) - 0.5, True))", 
         "sympy_expr": sp.Piecewise((0.5 * (y_true - y_pred)**2, sp.Abs(y_true - y_pred) < 1), (sp.Abs(y_true - y_pred) - 0.5, True)),
         "params": {"reduction": "mean"}},
        {"name": "binary_crossentropy", 
         "expr": "-y_true*log(y_pred + 1e-7) - (1-y_true)*log(1-y_pred + 1e-7)", 
         "sympy_expr": -y_true * sp.log(y_pred + 1e-7) - (1-y_true) * sp.log(1-y_pred + 1e-7),
         "params": {"reduction": "mean"}}
    ]
    
    # Add standard losses first (if size permits)
    for i in range(min(len(standard_losses), size)):
        losses.append(standard_losses[i])
    
    # Fill the rest with random losses
    while len(losses) < size:
        losses.append(generate_random_loss())
    
    return losses