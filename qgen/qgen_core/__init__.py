"""
Core business logic for forecasting question generation.

This package contains the main classes for:
- Generating forecasting questions from news articles (ForecastingQuestionGenerator)
- Processing and filtering articles (ArticleProcessor)

These are the primary interfaces for the question generation pipeline.
"""

from .question_generator import ForecastingQuestionGenerator
from .article_processor import ArticleProcessor

__all__ = [
    'ForecastingQuestionGenerator',
    'ArticleProcessor',
]

