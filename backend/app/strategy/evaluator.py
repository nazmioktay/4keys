import operator

import pandas as pd

from .schemas import ConditionNode, Operand

_OPS = {
    "gt": operator.gt,
    "lt": operator.lt,
    "gte": operator.ge,
    "lte": operator.le,
    "eq": operator.eq,
}


def _value(operand: Operand | None, row: pd.Series) -> float | None:
    if operand is None:
        return None
    if operand.indicator is not None:
        if operand.indicator not in row:
            raise ValueError(f"Bilinmeyen gösterge: {operand.indicator}")
        return row[operand.indicator]
    return operand.value


def evaluate(node: ConditionNode, row: pd.Series, prev_row: pd.Series | None) -> bool:
    """Bir kural düğümünü tek bir mum (ve bir önceki mum) için değerlendirir."""
    if node.type == "and":
        return all(evaluate(child, row, prev_row) for child in (node.conditions or []))

    if node.type == "or":
        return any(evaluate(child, row, prev_row) for child in (node.conditions or []))

    if node.type == "compare":
        left = _value(node.left, row)
        right = _value(node.right, row)
        if left is None or right is None or pd.isna(left) or pd.isna(right):
            return False
        return _OPS[node.op](left, right)

    if node.type == "cross":
        if prev_row is None:
            return False
        left_prev, right_prev = _value(node.left, prev_row), _value(node.right, prev_row)
        left_now, right_now = _value(node.left, row), _value(node.right, row)
        values = [left_prev, right_prev, left_now, right_now]
        if any(v is None or pd.isna(v) for v in values):
            return False
        if node.direction == "above":
            return left_prev <= right_prev and left_now > right_now
        return left_prev >= right_prev and left_now < right_now

    raise ValueError(f"Bilinmeyen kural tipi: {node.type}")
