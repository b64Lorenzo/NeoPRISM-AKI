"""Compatibility imports for binary clinical metrics shared by all models."""
from src.models.evaluation.metrics import auc
from src.models.evaluation.metrics import evaluate_binary as evaluate

__all__ = ['auc', 'evaluate']
