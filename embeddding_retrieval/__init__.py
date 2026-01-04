"""Embedding Retrieval Pipeline Package.

A modular and extensible pipeline for document embedding, retrieval, and ranking
designed for forecasting question-answering tasks.

Main Components:
- Config: Centralized configuration management
- DataLoader: Document and question loading with caching
- EmbeddingManager: Embedding computation and caching  
- RetrievalEngine: KNN search and document filtering
- EmbeddingRetrievalPipeline: Main orchestrator

Example Usage:
    from embeddding_retrieval import Config, EmbeddingRetrievalPipeline
    
    # Create configuration
    config = Config(base_data_dir='/path/to/data')
    
    # Run pipeline
    pipeline = EmbeddingRetrievalPipeline(config)
    pipeline.run()
"""

from .config import Config, ModelConfig, ProcessingConfig, PathConfig
from .data_loader import DataLoader, load_all_data
from .embedding_manager import EmbeddingManager  
from .retrieval import RetrievalEngine
from .pipeline import EmbeddingRetrievalPipeline

__version__ = "1.0.0"
__author__ = "Forecasting Team"

__all__ = [
    "Config",
    "ModelConfig", 
    "ProcessingConfig",
    "PathConfig",
    "DataLoader",
    "load_all_data",
    "EmbeddingManager",
    "RetrievalEngine", 
    "EmbeddingRetrievalPipeline",
] 