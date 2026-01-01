#!/usr/bin/env python3
"""New modular main script for the embedding retrieval pipeline.

This script demonstrates how to use the new modular pipeline architecture.
It supports configuration via environment variables and command line arguments.
"""

import argparse
import os
from pathlib import Path

from config import Config
from pipeline import EmbeddingRetrievalPipeline


def main():
    """Main entry point for the embedding retrieval pipeline."""
    parser = argparse.ArgumentParser(
        description="Embedding Retrieval Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python main_new.py
  
  # Use custom data directory
  python main_new.py --data-dir /path/to/data
  
  # Force recomputation of all components
  python main_new.py --force-all
  
  # Process only specific datasets
  python main_new.py --datasets deepseek metaculus
  
  # List available datasets
  python main_new.py --list-datasets
  
Environment Variables:
  DATA_DIR: Base data directory (overrides default)
  
Directory Structure:
  DATA_DIR/
  ├── documents/          # Input JSONL files with documents
  ├── questions/          # Input JSONL files with questions  
  └── precompiled/        # Output directory for cached data
      ├── documents/      # Cached document pickles
      ├── questions/      # Cached question pickles
      ├── *_embeddings.npy # Cached embeddings
      ├── passages_chunked.pkl # Cached passages
      └── ranked_queries_*.jsonl # Final results
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        help='Base data directory (overrides DATA_DIR environment variable)'
    )
    
    parser.add_argument(
        '--force-reload-data',
        action='store_true',
        help='Force reload data from source files (ignore cached pickles)'
    )
    
    parser.add_argument(
        '--force-recompute-embeddings',
        action='store_true', 
        help='Force recomputation of embeddings (ignore cached embeddings)'
    )
    
    parser.add_argument(
        '--force-recompute-retrieval',
        action='store_true',
        help='Force recomputation of retrieval results'
    )
    
    parser.add_argument(
        '--force-all',
        action='store_true',
        help='Force recomputation of all components (equivalent to all --force-* flags)'
    )
    
    parser.add_argument(
        '--datasets',
        nargs='*',
        help='Specific datasets to process (if not specified, processes all)'
    )
    
    parser.add_argument(
        '--list-datasets',
        action='store_true',
        help='List available datasets and exit'
    )
    
    parser.add_argument(
        '--config-summary',
        action='store_true',
        help='Show configuration summary and exit'
    )
    
    parser.add_argument(
        '--delta-days',
        type=int,
        help='Number of days before question resolution to filter documents'
    )
    
    parser.add_argument(
        '--max-docs',
        type=int,
        help='Maximum number of relevant documents to retrieve per question'
    )
    
    parser.add_argument(
        '--knn-k',
        type=int,
        help='Number of nearest neighbors to retrieve in KNN search'
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config(base_data_dir=args.data_dir)
    
    # Override configuration with command line arguments
    if args.delta_days is not None:
        config.processing.delta_days = args.delta_days
    if args.max_docs is not None:
        config.processing.max_relevant_docs = args.max_docs
    if args.knn_k is not None:
        config.processing.knn_k = args.knn_k
    
    # Create pipeline
    pipeline = EmbeddingRetrievalPipeline(config)
    
    # Handle special commands
    if args.config_summary:
        print(pipeline.get_config_summary())
        return
    
    if args.list_datasets:
        try:
            datasets = pipeline.list_available_datasets()
            print("Available datasets:")
            for dataset in datasets:
                print(f"  - {dataset}")
        except Exception as e:
            print(f"Error listing datasets: {e}")
            print("Make sure your data directory contains question files.")
        return
    
    # Set force flags
    force_reload_data = args.force_reload_data or args.force_all
    force_recompute_embeddings = args.force_recompute_embeddings or args.force_all
    force_recompute_retrieval = args.force_recompute_retrieval or args.force_all
    
    # Run pipeline
    try:
        pipeline.run(
            force_reload_data=force_reload_data,
            force_recompute_embeddings=force_recompute_embeddings,
            force_recompute_retrieval=force_recompute_retrieval,
            active_datasets=args.datasets
        )
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    main() 