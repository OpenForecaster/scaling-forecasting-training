import logging
import torch
from typing import List, Dict, Any
from transformers import AutoTokenizer
from qgen.inference.base_inference import BaseInference

logger = logging.getLogger(__name__)

class VLLMInference(BaseInference):
    """vLLM-based inference engine"""
    
    def __init__(self, model_path: str, max_tokens: int = 2048, temperature: float = 0.6):
        """
        Initialize vLLM inference engine
        
        Args:
            model_path: Path to the HuggingFace model
            max_tokens: Maximum tokens for generation
            temperature: Temperature for generation
        """
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.engine = None
        self.tokenizer = None 
        
        self._init_engine()
        
    def _init_engine(self):
        """Initialize the vLLM engine."""
        try:
            from vllm import LLM, SamplingParams
            logger.info(f"Initializing vLLM engine with model {self.model_path}")
            self.engine = LLM(
                model=self.model_path,
                trust_remote_code=True,
                tensor_parallel_size=torch.cuda.device_count(),
            )
            # Store SamplingParams class for later use
            self.SamplingParams = SamplingParams
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            logger.info("vLLM engine initialized successfully")
        except ImportError:
            logger.error("vLLM not installed. Run: pip install vllm")
            raise
        except Exception as e:
            logger.error(f"Error initializing vLLM engine: {e}")
            raise
            
    async def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate completions for prompts using vLLM
        
        Args:
            prompts: List of prompts to generate completions for
            **kwargs: Additional arguments for generation
            
        Returns:
            List of generated completions
        """
        if not self.engine:
            raise ValueError("vLLM engine not initialized")
            
        results = []
        
        prompts_tokenized = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True
            ) for prompt in prompts
        ]
        
        # logger.info(f"Example prompt: {prompts_tokenized[0]}")
        top_p = kwargs.get('top_p', 0.95)
        if "llama" in self.model_path:
            top_p = 0.9
        
        # Create sampling parameters for vLLM
        sampling_params = self.SamplingParams(
            temperature=kwargs.get('temperature', self.temperature),
            max_tokens=kwargs.get('max_tokens', self.max_tokens),
            top_p=top_p,
            top_k=kwargs.get('top_k', 20),
            stop=["<|im_end|>"]
        )
        
        # Generate completions (vLLM is synchronous)
        outputs = self.engine.generate(
            prompts=prompts_tokenized,
            sampling_params=sampling_params
        )
            
        # Process outputs
        for output in outputs:
            results.append(output.outputs[0].text.strip())
        
        return results 