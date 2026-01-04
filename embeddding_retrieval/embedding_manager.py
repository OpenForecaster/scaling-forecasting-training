"""Embedding management module.

This module handles text construction, embedding computation, and caching
for both documents and questions in a modular way.
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple

from config import Config
from embed import construct_text, get_embeddings, chunk_texts


class EmbeddingManager:
    """Manages embedding computation and caching for documents and questions."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def prepare_document_embeddings(self, documents: Dict[str, Any], force_recompute: bool = False) -> Tuple[List[str], np.ndarray, List[str], List[int]]:
        """Prepare document embeddings with passage-level chunking.
        
        Args:
            documents: Dictionary of document data
            force_recompute: If True, recompute embeddings even if cached
            
        Returns:
            Tuple of (document_keys, embeddings, passage_texts, passage_to_doc_idx)
        """
        print("Preparing document embeddings...")
        
        # Construct document texts
        document_keys, document_texts = construct_text(documents, field="maintext")
        
        # Handle passage chunking
        passages_path = self.config.get_passages_path()
        if not force_recompute and passages_path.exists():
            print(f"Loading cached passages from {passages_path}")
            with open(passages_path, 'rb') as f:
                passage_texts, passage_to_doc_idx = pickle.load(f)
        else:
            print("Chunking documents into passages...")
            passage_texts, passage_to_doc_idx = chunk_texts(
                document_texts,
                model_name=self.config.models.embedding_model,
                max_tokens=self.config.processing.max_tokens_per_passage,
                stride=self.config.processing.passage_stride
            )
            print(f"Caching passages to {passages_path}")
            with open(passages_path, 'wb') as f:
                pickle.dump((passage_texts, passage_to_doc_idx), f)
        
        # Handle embeddings
        embeddings_path = self.config.get_embeddings_path("doc_passage")
        if not force_recompute and embeddings_path.exists():
            print(f"Loading cached document embeddings from {embeddings_path}")
            embeddings = np.load(embeddings_path)
        else:
            print(f"Computing embeddings for {len(passage_texts)} passages...")
            embeddings = get_embeddings(
                passage_texts,
                is_query=False,
                model_name=self.config.models.embedding_model
            )
            print(f"Caching embeddings to {embeddings_path}")
            np.save(embeddings_path, embeddings)
        
        print(f"Document embeddings ready: {embeddings.shape}")
        return document_keys, embeddings, passage_texts, passage_to_doc_idx
    
    def prepare_question_embeddings(self, question_datasets: Dict[str, Dict[str, Any]], force_recompute: bool = False) -> Dict[str, Tuple[List[str], List[str], np.ndarray]]:
        """Prepare embeddings for all question datasets.
        
        Args:
            question_datasets: Dictionary mapping dataset names to question data
            force_recompute: If True, recompute embeddings even if cached
            
        Returns:
            Dictionary mapping dataset names to (keys, texts, embeddings) tuples
        """
        print("Preparing question embeddings...")
        
        question_embeddings = {}
        
        for dataset_name, questions in question_datasets.items():
            print(f"Processing question dataset: {dataset_name}")
            
            # Construct question texts
            question_keys, question_texts = construct_text(questions, field="question_title")
            
            # Handle embeddings
            embeddings_path = self.config.get_embeddings_path(f"{dataset_name}_questions")
            if not force_recompute and embeddings_path.exists():
                print(f"  Loading cached embeddings from {embeddings_path}")
                embeddings = np.load(embeddings_path)
            else:
                print(f"  Computing embeddings for {len(question_texts)} questions...")
                embeddings = get_embeddings(
                    question_texts,
                    is_query=True,
                    model_name=self.config.models.embedding_model
                )
                print(f"  Caching embeddings to {embeddings_path}")
                np.save(embeddings_path, embeddings)
            
            question_embeddings[dataset_name] = (question_keys, question_texts, embeddings)
            print(f"  Question embeddings ready: {embeddings.shape}")
        
        return question_embeddings
    
    def prepare_summary_embeddings(self, documents: Dict[str, Any], force_recompute: bool = False) -> np.ndarray:
        """Prepare summary embeddings if summaries are available.
        
        Args:
            documents: Dictionary of document data
            force_recompute: If True, recompute embeddings even if cached
            
        Returns:
            Summary embeddings array or None if no summaries available
        """
        print("Checking for document summaries...")
        
        # Check if documents have summaries
        has_summaries = any('summary' in doc and doc['summary'] for doc in documents.values())
        if not has_summaries:
            print("No summaries found in documents")
            return None
        
        # Construct summary texts
        _, summary_texts = construct_text(documents, field="summary")
        if not summary_texts:
            print("No valid summary texts found")
            return None
        
        # Handle embeddings
        embeddings_path = self.config.get_embeddings_path("summary")
        if not force_recompute and embeddings_path.exists():
            print(f"Loading cached summary embeddings from {embeddings_path}")
            embeddings = np.load(embeddings_path)
        else:
            print(f"Computing summary embeddings for {len(summary_texts)} summaries...")
            embeddings = get_embeddings(
                summary_texts,
                is_query=False,
                model_name=self.config.models.embedding_model
            )
            print(f"Caching summary embeddings to {embeddings_path}")
            np.save(embeddings_path, embeddings)
        
        print(f"Summary embeddings ready: {embeddings.shape}")
        return embeddings 