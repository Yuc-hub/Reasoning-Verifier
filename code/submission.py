"""Frozen MP2 submission: role-aware process verifier."""

from pathlib import Path

from methods.role_aware_process import predict as _predict


DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / 'checkpoints' / 'role_aware_process' / 'model.npz'


def predict(bank, arrays, checkpoint):
    """Return one Python integer in 0..3 per question, in bank order.

    bank: list of question dictionaries with four candidate texts each.
    arrays['embeddings']: float32 NumPy array of shape [N, 4, 1024].
    checkpoint: pathlib.Path or None, supplied through --checkpoint.
    """
    return _predict(bank, arrays, DEFAULT_CHECKPOINT if checkpoint is None else checkpoint)
