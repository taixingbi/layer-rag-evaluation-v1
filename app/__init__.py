"""Layer RAG evaluation: gold JSONL generation and batch RAG eval with optional LLM judge."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("layer-rag-evaluation")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
