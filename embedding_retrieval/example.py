#!/usr/bin/env python3
"""Example usage of the modular embedding retrieval pipeline.

This script demonstrates various ways to use the new modular system
for different use cases and configurations.
"""

import os
from pathlib import Path

from config import Config, ModelConfig, ProcessingConfig
from pipeline import EmbeddingRetrievalPipeline


def example_basic_usage():
    """Example 1: Basic usage with default configuration."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 60)
    
    # Create pipeline with default configuration
    pipeline = EmbeddingRetrievalPipeline()
    
    # Show configuration summary
    print(pipeline.get_config_summary())
    print()
    
    # List available datasets
    try:
        datasets = pipeline.list_available_datasets()
        print(f"Available datasets: {datasets}")
    except Exception as e:
        print(f"Could not list datasets: {e}")
    
    # Run pipeline (commented out to avoid actual execution)
    # pipeline.run()


def example_custom_config():
    """Example 2: Custom configuration."""
    print("=" * 60)
    print("EXAMPLE 2: Custom Configuration")
    print("=" * 60)
    
    # Create custom configuration
    config = Config(base_data_dir='/path/to/your/data')
    
    # Modify processing parameters
    config.processing.delta_days = 60  # 60 days before resolution
    config.processing.max_relevant_docs = 15  # Return top 15 docs
    config.processing.knn_k = 1000  # Search top 1000 candidates
    
    # Create pipeline with custom config
    pipeline = EmbeddingRetrievalPipeline(config)
    
    print(pipeline.get_config_summary())
    
    # Run pipeline (commented out)
    # pipeline.run()


def example_selective_processing():
    """Example 3: Selective processing and force flags."""
    print("=" * 60)
    print("EXAMPLE 3: Selective Processing")
    print("=" * 60)
    
    pipeline = EmbeddingRetrievalPipeline()
    
    # Process only specific datasets
    print("Processing only 'deepseek' dataset...")
    # pipeline.run_single_dataset('deepseek')
    
    # Force recomputation of embeddings only
    print("Force recomputing embeddings...")
    # pipeline.run(force_recompute_embeddings=True)
    
    # Force everything
    print("Force recomputing everything...")
    # pipeline.run(force_reload_data=True, 
    #              force_recompute_embeddings=True,
    #              force_recompute_retrieval=True)


def example_environment_config():
    """Example 4: Using environment variables."""
    print("=" * 60)
    print("EXAMPLE 4: Environment Configuration")
    print("=" * 60)
    
    # Set environment variable
    os.environ['DATA_DIR'] = '/custom/data/path'
    
    # Configuration will automatically use the environment variable
    config = Config()
    print(f"Using data directory: {config.paths.base_data_dir}")
    
    pipeline = EmbeddingRetrievalPipeline(config)


def example_configuration_profiles():
    """Example 5: Configuration profiles for different use cases."""
    print("=" * 60)
    print("EXAMPLE 5: Configuration Profiles")
    print("=" * 60)
    
    def create_fast_profile():
        """Fast processing profile - less thorough but quicker."""
        config = Config()
        config.processing.knn_k = 100
        config.processing.max_relevant_docs = 5
        config.processing.max_tokens_per_passage = 256
        return config
    
    def create_thorough_profile():
        """Thorough processing profile - slower but more comprehensive."""
        config = Config()
        config.processing.knn_k = 2000
        config.processing.max_relevant_docs = 20
        config.processing.max_tokens_per_passage = 1024
        return config
    
    def create_development_profile():
        """Development profile - for testing with small datasets."""
        config = Config()
        config.processing.knn_k = 50
        config.processing.max_relevant_docs = 3
        return config
    
    # Use different profiles
    profiles = {
        'fast': create_fast_profile(),
        'thorough': create_thorough_profile(),
        'dev': create_development_profile()
    }
    
    for name, config in profiles.items():
        print(f"\n{name.upper()} Profile:")
        print(f"  KNN k: {config.processing.knn_k}")
        print(f"  Max docs: {config.processing.max_relevant_docs}")
        print(f"  Max tokens: {config.processing.max_tokens_per_passage}")
        
        # Create pipeline with profile
        pipeline = EmbeddingRetrievalPipeline(config)
        # pipeline.run()  # Uncomment to actually run


def example_error_handling():
    """Example 6: Error handling and debugging."""
    print("=" * 60)
    print("EXAMPLE 6: Error Handling")
    print("=" * 60)
    
    try:
        # Try with invalid data directory
        config = Config(base_data_dir='/nonexistent/path')
        pipeline = EmbeddingRetrievalPipeline(config)
        
        # This will create the directories but won't find any data
        datasets = pipeline.list_available_datasets()
        print(f"Found datasets: {datasets}")
        
        if not datasets:
            print("No datasets found. Check your data directory structure.")
            
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Make sure your data directory exists and contains the required files.")


def example_batch_processing():
    """Example 7: Batch processing strategies."""
    print("=" * 60)
    print("EXAMPLE 7: Batch Processing")
    print("=" * 60)
    
    pipeline = EmbeddingRetrievalPipeline()
    
    # Strategy 1: Process all datasets at once
    print("Strategy 1: Process all datasets together")
    # pipeline.run()
    
    # Strategy 2: Process datasets individually (memory-efficient)
    print("\nStrategy 2: Process datasets individually")
    try:
        datasets = pipeline.list_available_datasets()
        for dataset in datasets:
            print(f"Processing {dataset}...")
            # pipeline.run_single_dataset(dataset)
    except Exception as e:
        print(f"Could not get datasets: {e}")
    
    # Strategy 3: Process specific subset
    print("\nStrategy 3: Process specific subset")
    target_datasets = ['deepseek', 'metaculus']
    # pipeline.run(active_datasets=target_datasets)


def main():
    """Run all examples."""
    print("EMBEDDING RETRIEVAL PIPELINE - EXAMPLES")
    print("=" * 80)
    print("This script demonstrates various usage patterns for the pipeline.")
    print("Most actual pipeline runs are commented out to avoid execution.")
    print("Uncomment the pipeline.run() calls to actually execute.")
    print()
    
    # Run examples
    example_basic_usage()
    example_custom_config()
    example_selective_processing()
    example_environment_config()
    example_configuration_profiles()
    example_error_handling()
    example_batch_processing()
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("See the README.md for more detailed documentation.")


if __name__ == "__main__":
    main() 