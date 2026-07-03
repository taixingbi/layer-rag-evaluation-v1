"""Layer RAG evaluation: batch gold eval, retrieval metrics, LLM-as-judge."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("layer-rag-evaluation")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
