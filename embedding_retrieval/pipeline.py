"""Main pipeline orchestrator.

This module provides the main pipeline class that coordinates all components
of the embedding and retrieval system in a clean, modular way.
"""

from typing import Optional, List

from config import Config
from data_loader import DataLoader
from embedding_manager import EmbeddingManager
from retrieval import RetrievalEngine


class EmbeddingRetrievalPipeline:
    """Main pipeline that orchestrates the embedding and retrieval process."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the pipeline with configuration.
        
        Args:
            config: Configuration object. If None, uses default configuration.
        """
        self.config = config or Config()
        self.data_loader = DataLoader(self.config)
        self.embedding_manager = EmbeddingManager(self.config)
        self.retrieval_engine = RetrievalEngine(self.config)
    
    def run(
        self, 
        force_reload_data: bool = False,
        force_recompute_embeddings: bool = False,
        force_recompute_retrieval: bool = False,
        active_datasets: Optional[List[str]] = None
    ) -> None:
        """Run the complete pipeline.
        
        Args:
            force_reload_data: If True, reload data from source files
            force_recompute_embeddings: If True, recompute embeddings
            force_recompute_retrieval: If True, recompute retrieval results
            active_datasets: List of dataset names to process. If None, processes all.
        """
        print("=" * 80)
        print("EMBEDDING RETRIEVAL PIPELINE")
        print("=" * 80)
        print(f"Base data directory: {self.config.paths.base_data_dir}")
        print(f"Precompiled directory: {self.config.paths.precompiled_dir}")
        print()
        
        # Step 1: Load data
        print("STEP 1: Loading data...")
        documents = self.data_loader.load_documents(force_reload=force_reload_data)
        question_datasets = self.data_loader.load_questions(force_reload=force_reload_data)
        print()
        
        if not documents:
            print("ERROR: No documents loaded. Check your data directory.")
            return
        
        if not question_datasets:
            print("ERROR: No questions loaded. Check your data directory.")
            return
        
        # Filter datasets if specified
        if active_datasets:
            question_datasets = {
                name: data for name, data in question_datasets.items() 
                if name in active_datasets
            }
            print(f"Processing only specified datasets: {list(question_datasets.keys())}")
        
        # Step 2: Prepare document embeddings
        print("STEP 2: Preparing document embeddings...")
        (document_keys, document_embeddings, 
         passage_texts, passage_to_doc_idx) = self.embedding_manager.prepare_document_embeddings(
            documents, force_recompute=force_recompute_embeddings
        )
        print()
        
        # Step 3: Prepare question embeddings
        print("STEP 3: Preparing question embeddings...")
        question_embeddings = self.embedding_manager.prepare_question_embeddings(
            question_datasets, force_recompute=force_recompute_embeddings
        )
        print()
        
        # Step 4: Perform retrieval for each dataset
        print("STEP 4: Performing retrieval...")
        for dataset_name, (q_keys, q_texts, q_embeddings) in question_embeddings.items():
            print(f"Processing dataset: {dataset_name}")
            
            self.retrieval_engine.retrieve_documents(
                question_dataset_name=dataset_name,
                question_texts=q_texts,
                question_data=question_datasets[dataset_name],
                question_embeddings=q_embeddings,
                document_keys=document_keys,
                document_data=documents,
                document_embeddings=document_embeddings,
                passage_texts=passage_texts,
                passage_to_doc_idx=passage_to_doc_idx,
                force_recompute=force_recompute_retrieval
            )
            print()
        
        print("=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("Results saved in JSONL format to:")
        for dataset_name in question_embeddings.keys():
            results_path = self.config.get_results_path(
                dataset_name, 
                self.config.processing.delta_days
            )
            print(f"  {dataset_name}: {results_path}")
    
    def run_single_dataset(
        self,
        dataset_name: str,
        force_reload_data: bool = False,
        force_recompute_embeddings: bool = False,
        force_recompute_retrieval: bool = False
    ) -> None:
        """Run the pipeline for a single dataset.
        
        Args:
            dataset_name: Name of the dataset to process
            force_reload_data: If True, reload data from source files
            force_recompute_embeddings: If True, recompute embeddings
            force_recompute_retrieval: If True, recompute retrieval results
        """
        self.run(
            force_reload_data=force_reload_data,
            force_recompute_embeddings=force_recompute_embeddings,
            force_recompute_retrieval=force_recompute_retrieval,
            active_datasets=[dataset_name]
        )
    
    def list_available_datasets(self) -> List[str]:
        """List all available question datasets.
        
        Returns:
            List of dataset names
        """
        question_datasets = self.data_loader.load_questions()
        return list(question_datasets.keys())
    
    def get_config_summary(self) -> str:
        """Get a summary of the current configuration.
        
        Returns:
            String summary of configuration
        """
        summary = []
        summary.append("Configuration Summary:")
        summary.append(f"  Base directory: {self.config.paths.base_data_dir}")
        summary.append(f"  Documents dir: {self.config.paths.documents_dir}")
        summary.append(f"  Questions dir: {self.config.paths.questions_dir}")
        summary.append(f"  Precompiled dir: {self.config.paths.precompiled_dir}")
        summary.append(f"  Embedding model: {self.config.models.embedding_model}")
        summary.append(f"  KNN k: {self.config.processing.knn_k}")
        summary.append(f"  Max relevant docs: {self.config.processing.max_relevant_docs}")
        summary.append(f"  Time delta (days): {self.config.processing.delta_days}")
        return "\n".join(summary) 