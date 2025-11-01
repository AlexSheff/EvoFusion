"""
Crossover and mutation operations for evolutionary algorithms.
Implements tree-based crossover and mutation for symbolic expressions.
"""

from __future__ import annotations

import random
import sympy as sp
from sympy.functions.elementary.piecewise import ExprCondPair
from typing import Any, Dict, List, Tuple, Union, Optional, Callable

# Define symbolic variables for activation and loss functions
activ_x = sp.Symbol('x')
y_true, y_pred = sp.symbols('y_true y_pred')


def get_expression_tree(expr_str: str, is_activation: bool = False) -> sp.Expr:
    """Convert string representation back to sympy expression."""
    try:
        # Parse the expression string based on the type
        if is_activation:
            return eval(expr_str, {"x": activ_x, "sin": sp.sin, "cos": sp.cos, "exp": sp.exp, "log": sp.log, "Abs": sp.Abs, "sqrt": sp.sqrt, "tanh": sp.tanh, "sigmoid": sp.sigmoid})
        else:
            return eval(expr_str, {"y_true": y_true, "y_pred": y_pred, "log": sp.log, "Abs": sp.Abs, "exp": sp.exp, "sqrt": sp.sqrt, "Max": sp.Max, "Min": sp.Min, "tanh": sp.tanh, "sigmoid": sp.sigmoid, "Piecewise": sp.Piecewise})
    except Exception as e:
        # Fallback to a simple expression if parsing fails
        if is_activation:
            return activ_x
        else:
            return (y_true - y_pred)**2


def get_random_subtree(expr: sp.Expr) -> sp.Expr:
    """Get a random subtree from a sympy expression."""
    if isinstance(expr, sp.Symbol) or isinstance(expr, sp.Number):
        return expr
    
    # Get all subexpressions
    subexprs = list(expr.args)
    if not subexprs:
        return expr
    
    # Choose a random subexpression or the whole expression
    choices = subexprs + [expr]
    return random.choice(choices)


def replace_subtree(expr: sp.Expr, old_subtree: sp.Expr, new_subtree: sp.Expr) -> sp.Expr:
    """Replace a subtree in a sympy expression, handling Piecewise safely."""
    if expr == old_subtree:
        return new_subtree

    # Base cases
    if isinstance(expr, (sp.Symbol, sp.Number)):
        return expr

    # Handle ExprCondPair: only replace in the expression part, keep condition intact
    if isinstance(expr, ExprCondPair):
        new_expr = replace_subtree(expr.expr, old_subtree, new_subtree)
        return ExprCondPair(new_expr, expr.cond)

    # Handle Piecewise specifically: rebuild pairs while preserving conditions
    if isinstance(expr, sp.Piecewise):
        new_pairs = []
        for pair in expr.args:
            # pair is ExprCondPair
            new_first = replace_subtree(pair.expr, old_subtree, new_subtree)
            new_pairs.append(ExprCondPair(new_first, pair.cond))
        return sp.Piecewise(*new_pairs)

    # Generic case: recursively replace in arguments
    new_args = [replace_subtree(arg, old_subtree, new_subtree) for arg in expr.args]
    return expr.func(*new_args)


def crossover_entities(entity1: Dict[str, Any], entity2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Perform crossover between two entities.
    Implements tree-based crossover for symbolic expressions.
    """
    # Determine if we're dealing with activation functions or loss functions
    is_activation = "expr" in entity1 and "x" in entity1["expr"]
    
    # Create copies to avoid modifying originals
    child1 = {**entity1}
    child2 = {**entity2}
    
    # Handle symbolic expressions if available
    if "expr" in entity1 and "expr" in entity2:
        # Get sympy expressions
        if "sympy_expr" in entity1 and "sympy_expr" in entity2:
            expr1 = entity1["sympy_expr"]
            expr2 = entity2["sympy_expr"]
        else:
            expr1 = get_expression_tree(entity1["expr"], is_activation)
            expr2 = get_expression_tree(entity2["expr"], is_activation)
        
        # Get random subtrees
        subtree1 = get_random_subtree(expr1)
        subtree2 = get_random_subtree(expr2)
        
        # Perform crossover by swapping subtrees
        new_expr1 = replace_subtree(expr1, subtree1, subtree2)
        new_expr2 = replace_subtree(expr2, subtree2, subtree1)
        
        # Update children with new expressions
        child1["sympy_expr"] = new_expr1
        child1["expr"] = str(new_expr1)
        child1["name"] = f"crossover_{hash(str(new_expr1)) % 10000}"
        
        child2["sympy_expr"] = new_expr2
        child2["expr"] = str(new_expr2)
        child2["name"] = f"crossover_{hash(str(new_expr2)) % 10000}"
    
    # Also crossover parameters if they exist
    if "params" in entity1 and "params" in entity2:
        # Create new parameter dictionaries
        child1["params"] = {**entity1["params"]}
        child2["params"] = {**entity2["params"]}
        
        # Find common parameters
        common_params = set(entity1["params"].keys()).intersection(set(entity2["params"].keys()))
        if common_params:
            # Select a random parameter to swap
            param_to_swap = random.choice(list(common_params))
            child1["params"][param_to_swap] = entity2["params"][param_to_swap]
            child2["params"][param_to_swap] = entity1["params"][param_to_swap]
    
    return child1, child2


def get_random_operator() -> Callable:
    """Get a random operator for mutation."""
    unary_ops = [
        lambda x: sp.Abs(x),
        lambda x: x**2,
        lambda x: sp.sqrt(sp.Abs(x)),
        lambda x: sp.log(1 + sp.Abs(x)),
        lambda x: sp.tanh(x),
        lambda x: sp.sin(x),
        lambda x: sp.exp(x)
    ]
    
    binary_ops = [
        lambda x, y: x + y,
        lambda x, y: x - y,
        lambda x, y: x * y,
        lambda x, y: x / (1 + sp.Abs(y)),
        lambda x, y: sp.Max(x, y),
        lambda x, y: sp.Min(x, y)
    ]
    
    if random.random() < 0.7:
        return random.choice(unary_ops)
    else:
        return random.choice(binary_ops)


def mutate_entity(entity: Dict[str, Any], mutation_rate: float = 0.3) -> Dict[str, Any]:
    """
    Mutate an entity with a given probability.
    Implements tree-based mutation for symbolic expressions.
    """
    # Skip mutation if random value exceeds mutation rate
    if random.random() > mutation_rate:
        return {**entity}
    
    # Create a copy to avoid modifying the original
    mutated = {**entity}
    
    # Determine if we're dealing with activation functions or loss functions
    is_activation = "expr" in entity and "x" in entity["expr"]
    
    # Handle symbolic expressions if available
    if "expr" in entity and random.random() < 0.7:  # 70% chance to mutate expression
        # Get sympy expression
        if "sympy_expr" in entity:
            expr = entity["sympy_expr"]
        else:
            expr = get_expression_tree(entity["expr"], is_activation)
        
        # Choose mutation type
        mutation_type = random.choice(["replace_subtree", "change_operator", "add_operation"])
        
        if mutation_type == "replace_subtree" and not isinstance(expr, sp.Symbol):
            # Replace a random subtree with a constant or simple expression
            subtree = get_random_subtree(expr)
            
            if random.random() < 0.5:
                # Replace with a constant
                new_value = random.uniform(0.1, 2.0)
                new_subtree = sp.Float(new_value)
            else:
                # Replace with a simple expression
                if is_activation:
                    new_subtree = random.choice([
                        activ_x,
                        activ_x**2,
                        sp.Abs(activ_x),
                        sp.tanh(activ_x)
                    ])
                else:
                    new_subtree = random.choice([
                        y_true - y_pred,
                        (y_true - y_pred)**2,
                        sp.Abs(y_true - y_pred)
                    ])
            
            expr = replace_subtree(expr, subtree, new_subtree)
            
        elif mutation_type == "change_operator" and not isinstance(expr, sp.Symbol):
            # Change an operator in the expression
            if len(expr.args) > 0:
                # Get a random operator
                op = get_random_operator()
                
                if len(expr.args) == 1 or (len(expr.args) > 1 and len(op.__code__.co_varnames) == 1):
                    # Unary operator
                    arg = expr.args[0]
                    expr = op(arg)
                elif len(expr.args) > 1 and len(op.__code__.co_varnames) > 1:
                    # Binary operator
                    arg1, arg2 = expr.args[0], expr.args[1]
                    expr = op(arg1, arg2)
        
        elif mutation_type == "add_operation":
            # Add a new operation
            op = get_random_operator()
            
            if len(op.__code__.co_varnames) == 1:
                # Unary operator
                expr = op(expr)
            else:
                # Binary operator with a constant
                const = sp.Float(random.uniform(0.1, 2.0))
                if random.random() < 0.5:
                    expr = op(expr, const)
                else:
                    expr = op(const, expr)
        
        # Update mutated entity with new expression
        mutated["sympy_expr"] = expr
        mutated["expr"] = str(expr)
        mutated["name"] = f"mutated_{hash(str(expr)) % 10000}"
    
    # Also mutate parameters if they exist
    if "params" in entity and random.random() < 0.5:  # 50% chance to mutate parameters
        mutated["params"] = {**entity["params"]}
        
        for param_name, param_value in entity["params"].items():
            # Only mutate numeric parameters with 30% chance
            if isinstance(param_value, (int, float)) and random.random() < 0.3:
                # Add random noise
                if isinstance(param_value, int):
                    noise = random.randint(-1, 1)
                else:  # float
                    noise = random.uniform(-0.1, 0.1)
                
                mutated["params"][param_name] = max(0.0001, param_value + noise)  # Ensure positive value
    
    return mutated