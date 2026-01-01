"""Retrieval module for KNN search and document filtering.

This module handles the first-stage retrieval using KNN search with 
time-based filtering to ensure documents are published before question resolution.
"""

import json
import pickle
import numpy as np
from typing import Dict, Any, List, Tuple

from config import Config
from tiny_knn import exact_search


class RetrievalEngine:
    """Handles KNN search and document filtering for question-document pairs."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def retrieve_documents(
        self,
        question_dataset_name: str,
        question_texts: List[str],
        question_data: Dict[str, Any],
        question_embeddings: np.ndarray,
        document_keys: List[str],
        document_data: Dict[str, Any],
        document_embeddings: np.ndarray,
        passage_texts: List[str],
        passage_to_doc_idx: List[int],
        force_recompute: bool = False
    ) -> None:
        """Perform KNN retrieval and save results.
        
        Args:
            question_dataset_name: Name of the question dataset
            question_texts: List of question text strings
            question_data: Dictionary of question metadata
            question_embeddings: Question embedding matrix
            document_keys: List of document keys
            document_data: Dictionary of document metadata  
            document_embeddings: Document embedding matrix
            passage_texts: List of passage text strings
            passage_to_doc_idx: Mapping from passage index to document index
            force_recompute: If True, recompute even if cached results exist
        """
        print(f"Performing retrieval for {question_dataset_name}")
        
        # Check for cached results
        pairs_path = self.config.get_pairs_path(question_dataset_name, "docs")
        results_path = self.config.get_results_path(
            question_dataset_name, 
            self.config.processing.delta_days
        )
        
        if not force_recompute and pairs_path.exists() and results_path.exists():
            print(f"  Cached results found at {results_path}")
            return
        
        print(f"  Performing KNN search: questions {question_embeddings.shape} vs documents {document_embeddings.shape}")
        
        # Perform KNN search
        indices, distances = exact_search(
            question_embeddings,
            document_embeddings, 
            k=self.config.processing.knn_k,
            metric='cosine'
        )
        
        print(f"  KNN search completed: {indices.shape}")
        
        # Process results with time filtering
        pairs = []
        delta_seconds = self.config.processing.delta_days * 86400
        
        for i in range(distances.shape[0]):
            count = 0
            relevant_docs = []
            query = question_texts[i]
            seen_docs = set()
            
            # Get question metadata
            q_meta = question_data.get(str(i + 1))
            if q_meta is None:
                continue
            
            for j in range(distances.shape[1]):
                if count >= self.config.processing.max_relevant_docs:
                    break
                    
                passage_idx = int(indices[i, j])
                score = float(distances[i, j])
                docid = passage_to_doc_idx[passage_idx]
                passage_text = passage_texts[passage_idx]
                key = document_keys[docid]
                doc_meta = document_data.get(key)
                
                if doc_meta is None:
                    continue
                
                # Apply time filtering
                if not self._is_document_valid(doc_meta, q_meta, delta_seconds):
                    continue
                
                # Avoid duplicate documents
                if docid in seen_docs:
                    continue
                
                seen_docs.add(docid)
                count += 1
                
                # Prepare document metadata (remove large text field)
                doc_meta_copy = doc_meta.copy()
                doc_meta_copy.pop('maintext', None)
                doc_meta_copy['relevant_passage'] = passage_text
                
                relevant_docs.append((str(score), key, doc_meta_copy))
                pairs.append(((i, passage_idx), (query, passage_text)))
            
            # Store results in question data
            question_data[str(i + 1)]['relevant_articles_sorted_by_docs'] = relevant_docs
        
        # Save pairs for potential reranking
        print(f"  Saving {len(pairs)} pairs to {pairs_path}")
        with open(pairs_path, 'wb') as f:
            pickle.dump(pairs, f)
        
        # Save results
        self._save_results(question_dataset_name, question_data)
        print(f"  Results saved to {results_path}")
    
    def _is_document_valid(self, doc_meta: Dict[str, Any], q_meta: Dict[str, Any], delta_seconds: int) -> bool:
        """Check if document is valid for the given question based on timing constraints.
        
        Args:
            doc_meta: Document metadata
            q_meta: Question metadata
            delta_seconds: Time window in seconds before question resolution
            
        Returns:
            True if document is valid (published before question resolution with buffer)
        """
        doc_date = doc_meta.get('max_date')
        resolution_date = q_meta.get('resolution_date')
        
        if doc_date is None or resolution_date is None:
            return False
        
        # Document must be published at least delta_seconds before resolution
        return doc_date < (resolution_date - delta_seconds)
    
    def _save_results(self, dataset_name: str, question_data: Dict[str, Any]) -> None:
        """Save results in JSONL format.
        
        Args:
            dataset_name: Name of the question dataset
            question_data: Dictionary of question data with results
        """
        results_path = self.config.get_results_path(
            dataset_name,
            self.config.processing.delta_days
        )
        
        with open(results_path, 'w', encoding='utf-8') as f:
            for qid, qdata in question_data.items():
                record = {'qid': qid}
                record.update(qdata)
                f.write(json.dumps(record, ensure_ascii=False) + '\n') 