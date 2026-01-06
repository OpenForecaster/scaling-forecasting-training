"""Configuration module for the embedding retrieval pipeline.

This module centralizes all configuration settings including paths, model names,
and processing parameters to make the pipeline more flexible and maintainable.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelConfig:
    """Configuration for model names and settings."""
    summary_model: str = "/fast/nchandak/models/Qwen3-32B"
    embedding_model: str = 'Qwen/Qwen3-Embedding-8B'
    reranker_model: str = 'Qwen/Qwen3-Reranker-8B'


@dataclass
class ProcessingConfig:
    """Configuration for processing parameters."""
    max_tokens_per_passage: int = 512
    passage_stride: int = 64
    knn_k: int = 500
    max_relevant_docs: int = 10
    delta_days: int = 30  # Time window for filtering documents before question resolution
    

@dataclass
class PathConfig:
    """Configuration for all file paths."""
    # Base data directory - can be overridden via environment variable
    base_data_dir: Path
    
    # Input directories
    documents_dir: Path
    questions_dir: Path
    
    # Output/cache directories  
    precompiled_dir: Path
    precompiled_documents_dir: Path
    precompiled_questions_dir: Path
    
    def __post_init__(self):
        """Ensure all directories exist."""
        self.precompiled_dir.mkdir(parents=True, exist_ok=True)
        self.precompiled_documents_dir.mkdir(parents=True, exist_ok=True)
        self.precompiled_questions_dir.mkdir(parents=True, exist_ok=True)


class Config:
    """Main configuration class that combines all settings."""
    
    def __init__(self, base_data_dir: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            base_data_dir: Override the base data directory. If None, uses environment
                          variable DATA_DIR or defaults to /fast/nchandak/forecasting/newsdata/retrieval/data/
        """
        if base_data_dir is None:
            base_data_dir = os.getenv(
                'DATA_DIR', 
                '/fast/nchandak/forecasting/newsdata/retrieval/metaculus/'
            )
        
        base_path = Path(base_data_dir)
        
        self.paths = PathConfig(
            base_data_dir=base_path,
            documents_dir=base_path / 'documents',
            questions_dir=base_path / 'questions', 
            precompiled_dir=base_path / 'precompiled',
            precompiled_documents_dir=base_path / 'precompiled' / 'documents',
            precompiled_questions_dir=base_path / 'precompiled' / 'questions',
        )
        
        self.models = ModelConfig()
        self.processing = ProcessingConfig()
    
    def get_document_files(self) -> List[Path]:
        """Get all JSONL files in the documents directory."""
        if not self.paths.documents_dir.exists():
            return []
        return list(self.paths.documents_dir.glob('*.jsonl'))
    
    def get_question_files(self) -> List[Path]:
        """Get all JSONL files in the questions directory.""" 
        if not self.paths.questions_dir.exists():
            return []
        return list(self.paths.questions_dir.glob('*.jsonl'))
    
    def get_precompiled_doc_path(self, original_file: Path) -> Path:
        """Get the precompiled path for a document file."""
        return self.paths.precompiled_documents_dir / f"{original_file.stem}.pkl"
    
    def get_precompiled_question_path(self, original_file: Path) -> Path:
        """Get the precompiled path for a question file."""
        return self.paths.precompiled_questions_dir / f"{original_file.stem}.pkl"
    
    def get_embeddings_path(self, name: str, data_type: str = 'embeddings') -> Path:
        """Get path for embeddings file."""
        return self.paths.precompiled_dir / f"{name}_{data_type}.npy"
    
    def get_passages_path(self) -> Path:
        """Get path for chunked passages."""
        return self.paths.precompiled_dir / 'passages_chunked.pkl'
    
    def get_pairs_path(self, question_set: str, doc_set: str) -> Path:
        """Get path for query-document pairs."""
        return self.paths.precompiled_dir / f"pairs_{question_set}_{doc_set}.pkl"
    
    def get_results_path(self, question_set: str, days: int, format: str = 'jsonl') -> Path:
        """Get path for final results."""
        return self.paths.precompiled_dir / f"ranked_queries_{question_set}_{days}.{format}" 