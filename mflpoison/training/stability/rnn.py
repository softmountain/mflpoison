from contextlib import contextmanager

import torch


@contextmanager
def second_order_rnn_context(enabled: bool):
    """Use native RNN kernels when a loss requires double backward."""

    with torch.backends.cudnn.flags(enabled=not bool(enabled)):
        yield
