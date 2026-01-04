from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseInference(ABC):
    """Base class for inference engines"""
    
    @abstractmethod
    async def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate text completions for the given prompts
        
        Args:
            prompts: List of prompts to generate completions for
            **kwargs: Additional arguments for generation
            
        Returns:
            List of generated completions
        """
        pass 