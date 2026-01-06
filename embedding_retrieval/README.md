# Embedding Retrieval Pipeline

A modular and extensible pipeline for document embedding, retrieval, and ranking designed for forecasting question-answering tasks.

## Overview

This system provides a complete pipeline for:
1. Loading and caching documents and questions from JSONL files
2. Computing and caching embeddings for both documents and questions
3. Performing KNN retrieval with time-based filtering
4. Saving results in structured JSONL format

## Directory Structure

```
DATA_DIR/
├── documents/              # Input JSONL files with documents
│   ├── guardian_2025.jsonl
│   ├── www.cnbc.com_*.jsonl
│   └── ...
├── questions/              # Input JSONL files with questions
│   ├── deepseekv3_*.jsonl
│   ├── metaculus.jsonl
│   └── ...
└── precompiled/            # Output directory for cached data
    ├── documents/          # Cached document pickles
    ├── questions/          # Cached question pickles
    ├── *_embeddings.npy    # Cached embeddings
    ├── passages_chunked.pkl # Cached passages
    └── ranked_queries_*.jsonl # Final results
```

## Quick Start

### Basic Usage

```python
from embedding_retrieval import Config, EmbeddingRetrievalPipeline

# Create configuration (uses default paths)
config = Config()

# Run the complete pipeline
pipeline = EmbeddingRetrievalPipeline(config)
pipeline.run()
```

### Custom Configuration

```python
from embedding_retrieval import Config, EmbeddingRetrievalPipeline

# Custom data directory
config = Config(base_data_dir='/path/to/your/data')

# Modify processing parameters
config.processing.delta_days = 60  # 60 days before resolution
config.processing.max_relevant_docs = 15  # Return top 15 docs per question
config.processing.knn_k = 1000  # Search top 1000 candidates

# Run pipeline
pipeline = EmbeddingRetrievalPipeline(config)
pipeline.run()
```

### Command Line Usage

```bash
# Basic run
python main_new.py

# Custom data directory
python main_new.py --data-dir /path/to/data

# Force recomputation of all components
python main_new.py --force-all

# Process only specific datasets
python main_new.py --datasets deepseek metaculus

# List available datasets
python main_new.py --list-datasets

# Show configuration
python main_new.py --config-summary
```

## Configuration

### Environment Variables

- `DATA_DIR`: Base data directory (default: `/fast/nchandak/forecasting/newsdata/retrieval/data/`)

### Configuration Classes

#### ModelConfig
- `summary_model`: Model for text summarization
- `embedding_model`: Model for computing embeddings  
- `reranker_model`: Model for reranking results

#### ProcessingConfig
- `max_tokens_per_passage`: Maximum tokens per document passage (default: 512)
- `passage_stride`: Overlap between passages (default: 64)
- `knn_k`: Number of candidates in KNN search (default: 500)
- `max_relevant_docs`: Maximum documents per question (default: 10)
- `delta_days`: Days before resolution to filter documents (default: 30)

## Modules

### config.py
Centralized configuration management with support for:
- Path configuration and automatic directory creation
- Model configuration
- Processing parameters
- Helper methods for file paths

### data_loader.py
Handles loading and caching of data:
- Automatic detection of document and question files
- Intelligent caching with timestamp checking
- Dataset name mapping and organization
- Support for multiple question datasets

### embedding_manager.py
Manages embedding computation and caching:
- Document passage chunking for better retrieval
- Embedding computation with task-specific instructions
- Automatic caching to avoid recomputation
- Support for both document and question embeddings

### retrieval.py
Handles KNN search and filtering:
- Cosine similarity-based KNN search
- Time-based document filtering
- Deduplication of results
- Structured output generation

### pipeline.py
Main orchestrator that coordinates all components:
- Step-by-step pipeline execution
- Progress reporting and error handling
- Support for partial recomputation
- Dataset filtering capabilities

## Data Formats

### Document JSONL Format
```json
{
  "id": "doc_123",
  "title": "Document Title",
  "description": "Brief description",
  "maintext": "Full document text...",
  "max_date": 1640995200,
  "source_domain": "example.com",
  "authors": ["Author Name"]
}
```

### Question JSONL Format
```json
{
  "question_title": "Will X happen by Y?",
  "background": "Context information...",
  "resolution_criteria": "How the question will be resolved",
  "resolution_date": 1672531200,
  "question_start_date": 1640995200,
  "answer_type": "binary",
  "data_source": "source_name"
}
```

### Output JSONL Format
```json
{
  "qid": "1",
  "question_title": "Will X happen by Y?",
  "resolution_date": 1672531200,
  "relevant_articles_sorted_by_docs": [
    ["0.123", "doc_456", {
      "title": "Relevant Article",
      "max_date": 1650000000,
      "relevant_passage": "Most relevant text passage..."
    }]
  ]
}
```

## Extending the System

### Adding New Datasets
1. Place JSONL files in the `questions/` directory
2. Update `_get_dataset_name()` in `data_loader.py` if needed
3. The system will automatically detect and process new files

### Custom Processing
```python
from embedding_retrieval import EmbeddingRetrievalPipeline, Config

class CustomPipeline(EmbeddingRetrievalPipeline):
    def run(self, **kwargs):
        # Add custom preprocessing
        super().run(**kwargs)
        # Add custom postprocessing
```

### Configuration Profiles
```python
def create_fast_config():
    config = Config()
    config.processing.knn_k = 100  # Faster but less thorough
    config.processing.max_relevant_docs = 5
    return config

def create_thorough_config():
    config = Config()
    config.processing.knn_k = 2000  # Slower but more thorough
    config.processing.max_relevant_docs = 20
    return config
```

## Performance Tips

1. **Use Caching**: The system automatically caches processed data. Only use `--force-*` flags when necessary.

2. **Batch Processing**: Process multiple datasets in one run rather than separate runs.

3. **Memory Management**: For large datasets, consider processing one dataset at a time:
   ```bash
   python main_new.py --datasets deepseek
   python main_new.py --datasets metaculus
   ```

4. **Parallel Processing**: The embedding computation uses multiple GPUs automatically when available.

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `knn_k` or process datasets individually
2. **Missing Files**: Check that input files are in the correct directories
3. **Permission Errors**: Ensure write permissions for the precompiled directory
4. **Model Loading**: Verify model paths and availability

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

pipeline = EmbeddingRetrievalPipeline(config)
pipeline.run()
```

## Migration from Legacy Code

The new system maintains compatibility with the original data formats. To migrate:

1. Update your scripts to use the new imports:
   ```python
   from embedding_retrieval import EmbeddingRetrievalPipeline, Config
   ```

2. Replace hardcoded paths with configuration:
   ```python
   # Old
   mainpath = '/fast/nchandak/forecasting/newsdata/retrieval/data/'
   
   # New
   config = Config(base_data_dir='/fast/nchandak/forecasting/newsdata/retrieval/data/')
   ```

3. Use the pipeline instead of manual orchestration:
   ```python
   # Old
   # ... manual data loading, embedding, retrieval ...
   
   # New
   pipeline = EmbeddingRetrievalPipeline(config)
   pipeline.run()
   ``` 