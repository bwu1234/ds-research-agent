"""A local, read-only web viewer for the run ledger: batches, per-run traces,
tasks across batches, and paired comparisons. See ``server`` for how it is
served and ``trace`` for how a run's steps are arranged."""

from eval.kramabench.viewer.pages import Context
from eval.kramabench.viewer.server import ViewerServer, serve

__all__ = ["Context", "ViewerServer", "serve"]
