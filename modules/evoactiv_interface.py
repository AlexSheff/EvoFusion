"""
Interface to the EvoActiv subsystem: generation of initial activations and parameter adaptation.
This implementation generates symbolic activation functions.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

import sympy
import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_symbolic_activation() -> sympy.Expr:
    """Generates a random symbolic activation function as a sympy expression."""
    
    # Basic functions
    functions = [
        x,
        -x,
        x**2,
        x**3,
        sympy.sin(x),
        sympy.cos(x),
        sympy.exp(x),
        sympy.log(sympy.Abs(x) + 1e-6),
        sympy.sqrt(sympy.Abs(x) + 1e-6),
        sympy.tanh(x),
        1 / (1 + sympy.exp(-x)),
    ]
    
    # Combine functions
    f1 = random.choice(functions)
    f2 = random.choice(functions)
    
    # Combine with operators
    op = random.choice(['+', '-', '*', '/'])
    
    if op == '+':
        expr = f1 + f2
    elif op == '-':
        expr = f1 - f2
    elif op == '*':
        expr = f1 * f2
    else: # op == '/'
        expr = f1 / (f2 + 1e-6) # Add epsilon to avoid division by zero

    # Add a random constant
    if random.random() < 0.5:
        const = random.uniform(-1, 1)
        expr += const

    return expr


def generate_initial_activations(size: int) -> List[Dict[str, Any]]:
    """Generates a list of symbolic activation functions."""
    activations: List[Dict[str, Any]] = []
    for _ in range(size):
        sym_expr = generate_symbolic_activation()
        activations.append({
            "name": "symbolic",
            "expr": str(sym_expr),
            "sympy_expr": sym_expr,
            "params": {
                "expression": str(sym_expr)
            }
        })
    return activations

def get_activation_function(activation_config: Dict[str, Any]):
    """
    Get the activation function based on the configuration.
    
    Args:
        activation_config: Activation function configuration
        
    Returns:
        Callable activation function
    """
    name = activation_config.get("name", "relu").lower()
    params = activation_config.get("params", {})
    
    if name == "symbolic":
        expression = activation_config.get("expr") or params.get("expression") or "x"
        
        def symbolic_activation(x):
            # A simple parser for the expression string
            if "**2" in expression:
                return x**2
            elif "**3" in expression:
                return x**3
            elif "sin" in expression:
                return torch.sin(x)
            elif "cos" in expression:
                return torch.cos(x)
            elif "exp" in expression:
                return torch.exp(x)
            elif "log" in expression:
                return torch.log(torch.abs(x) + 1e-6)
            elif "sqrt" in expression:
                return torch.sqrt(torch.abs(x) + 1e-6)
            elif "tanh" in expression:
                return torch.tanh(x)
            elif "1 / (1 + exp(-x))" in expression or "sigmoid" in expression:
                return torch.sigmoid(x)
            else:
                # Fallback to relu
                return F.relu(x)
        return symbolic_activation
    elif name == "relu":
        return torch.nn.ReLU()
    elif name == "leaky_relu":
        return torch.nn.LeakyReLU(params.get("negative_slope", 0.01))
    elif name == "elu":
        return torch.nn.ELU(params.get("alpha", 1.0))
    elif name == "sigmoid":
        return torch.nn.Sigmoid()
    elif name == "tanh":
        return torch.nn.Tanh()
    else:
        raise ValueError(f"Unknown activation function: {name}")
# Export a common symbolic variable for tests/utilities
x = sympy.Symbol('x')