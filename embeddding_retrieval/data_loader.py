"""Data loading module with caching support.

This module provides functionality to load documents and questions from JSONL files
with automatic caching to pickle files for faster subsequent loads.
"""

import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict

from config import Config
from parse import load_jsonl, _extract_data_fields, _extract_data_fields_questions


class DataLoader:
    """Handles loading and caching of documents and questions."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def load_documents(self, force_reload: bool = False) -> Dict[str, Any]:
        """Load all documents with caching.
        
        Args:
            force_reload: If True, ignore cached data and reload from source
            
        Returns:
            Dictionary mapping document IDs to document data
        """
        all_documents = {}
        document_files = self.config.get_document_files()
        
        print(f"Found {len(document_files)} document files")
        
        for doc_file in document_files:
            print(f"Loading documents from {doc_file.name}")
            cached_path = self.config.get_precompiled_doc_path(doc_file)
            
            # Check if cached version exists and is newer than source
            # if True or (not force_reload and 
            #     cached_path.exists() and 
            #     cached_path.stat().st_mtime > doc_file.stat().st_mtime):
            try:
                print(f"  Loading from cache: {cached_path}")
                with open(cached_path, 'rb') as f:
                    documents = pickle.load(f)
            # else:
            except Exception as e:
                print(f"Error loading from cache: {e}")
                print(f"  Loading from source: {doc_file}")
                documents = load_jsonl(doc_file, transform=_extract_data_fields)
                
                # Cache the loaded data
                print(f"  Caching to: {cached_path}")
                with open(cached_path, 'wb') as f:
                    pickle.dump(documents, f)
            
            print(f"  Loaded {len(documents)} documents")
            all_documents.update(documents)
        
        print(f"Total documents loaded: {len(all_documents)}")
        return all_documents
    
    def load_questions(self, force_reload: bool = False) -> Dict[str, Dict[str, Any]]:
        """Load all questions organized by dataset.
        
        Args:
            force_reload: If True, ignore cached data and reload from source
            
        Returns:
            Dictionary mapping dataset names to question dictionaries
        """
        question_datasets = {}
        question_files = self.config.get_question_files()
        
        print(f"Found {len(question_files)} question files")
        
        for question_file in question_files:
            dataset_name = self._get_dataset_name(question_file)
            # if 'updated_resolution' not in dataset_name:
            #     continue
            if 'Nov' not in dataset_name:
                continue
            
            print(f"Loading questions from {question_file.name} -> {dataset_name}")
            
            cached_path = self.config.get_precompiled_question_path(question_file)
            
            # Check if cached version exists and is newer than source
            if (not force_reload and 
                cached_path.exists() and 
                cached_path.stat().st_mtime > question_file.stat().st_mtime):
                
                print(f"  Loading from cache: {cached_path}")
                with open(cached_path, 'rb') as f:
                    questions = pickle.load(f)
            else:
                print(f"  Loading from source: {question_file}")
                questions = load_jsonl(question_file, transform=_extract_data_fields_questions)
                
                # Cache the loaded data
                print(f"  Caching to: {cached_path}")
                with open(cached_path, 'wb') as f:
                    pickle.dump(questions, f)
            
            print(f"  Loaded {len(questions)} questions")
            
            # Merge questions from the same dataset
            if dataset_name in question_datasets:
                question_datasets[dataset_name].update(questions)
            else:
                question_datasets[dataset_name] = questions
        
        # Print summary
        for dataset_name, questions in question_datasets.items():
            print(f"Dataset '{dataset_name}': {len(questions)} questions")
        
        return question_datasets
    
    def _get_dataset_name(self, question_file: Path) -> str:
        """Determine dataset name from filename.
        
        This method maps question filenames to logical dataset names
        for better organization.
        """
        filename = question_file.name.lower()
        
        return question_file.stem
    
        # Define mapping rules
        if "deepseek" in filename or "janmarch" in filename:
            return "deepseek"
        elif "o4mini" in filename:
            return "o4mini"  
        elif "binary_train" in filename:
            return "binary_train"
        elif "metaculus" in filename:
            return "metaculus"
        elif "combined_non_numeric_all_train" in filename:
            return "train"
        elif "combined_non_numeric_all_validation" in filename:
            return "validation"
        elif "combined" in filename:
            return "combined"
        else:
            # Use filename stem as fallback
            return question_file.stem
    
    def get_active_datasets(self, question_datasets: Dict[str, Dict[str, Any]]) -> List[str]:
        """Get list of dataset names that should be processed.
        
        This can be configured or filtered based on requirements.
        Currently returns all available datasets.
        """
        return list(question_datasets.keys())


def load_all_data(config: Config, force_reload: bool = False) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Convenience function to load all documents and questions.
    
    Args:
        config: Configuration object
        force_reload: If True, ignore cached data and reload from source
        
    Returns:
        Tuple of (documents, question_datasets)
    """
    loader = DataLoader(config)
    documents = loader.load_documents(force_reload=force_reload)
    questions = loader.load_questions(force_reload=force_reload)
    return documents, questions 