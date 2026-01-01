import os
import json
import logging
import asyncio
import argparse
from typing import List, Dict

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

from qgen.question_generator import ForecastingQuestionGenerator
# from qgen.inference.vllm_inference import VLLMInference
from qgen.inference.openrouter_inference import OpenRouterInference


class LeakageRemover:
    def __init__(self, inference_engine, use_freeq: bool = True):
        """
        Initialize the leakage remover.
        
        Args:
            inference_engine: Engine for text generation (must implement BaseInference)
            use_freeq: If True, process free-form questions, else MCQ questions
        """
        self.inference_engine = inference_engine
        self.use_freeq = use_freeq
        
        # Create a question generator instance to reuse its methods
        self.question_generator = ForecastingQuestionGenerator(
            inference_engine=inference_engine,
            use_freeq=use_freeq,
            check_leakage=True,
            leakage_engine=inference_engine
        )
    
    def load_jsonl(self, file_path: str) -> List[Dict]:
        """Load data from JSONL file."""
        data = []
        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            item = json.loads(line)
                            data.append(item)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Could not decode JSON on line {line_num}: {e}")
            logger.info(f"Loaded {len(data)} entries from {file_path}")
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return []
        return data
    
    def save_jsonl(self, data: List[Dict], file_path: str) -> None:
        """Save data to JSONL file."""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                for item in data:
                    f.write(json.dumps(item) + '\n')
            logger.info(f"Saved {len(data)} entries to {file_path}")
        except Exception as e:
            logger.error(f"Error saving file {file_path}: {e}")
    
    def extract_question_from_entry(self, entry: Dict) -> str:
        """Extract the question text from a data entry."""
        # Try different possible fields where the question might be stored
        question_fields = [
            'final_question',
            'choose_best_response', 
            'generated_questions',
            'question'
        ]
        
        for field in question_fields:
            if field in entry and entry[field]:
                question_text = entry[field]
                if isinstance(question_text, str) and len(question_text.strip()) > 10:
                    # If it's from choose_best_response, extract the final question
                    if field == 'choose_best_response':
                        return self.question_generator.extract_final_question(question_text)
                    return question_text
        
        return ""
    
    def needs_leakage_removal(self, entry: Dict) -> bool:
        """Check if entry needs leakage removal processing."""
        # Skip if already processed successfully
        
        if len(entry.get('leakage_removed_question', '')) < 10:
            return True
        
        if entry.get('leakage_removed', 0) == 1 :
            return False
        
        # Skip if no valid question found
        question_text = self.extract_question_from_entry(entry)
        if not question_text or len(question_text.strip()) <= 10:
            return False
            
        return True

    async def remove_leakage_from_entries(self, entries: List[Dict], input_file: str, batch_size: int = 5) -> List[Dict]:
        """
        Remove leakage from all entries in the list with incremental saving to input file.
        
        Args:
            entries: List of data entries
            input_file: Path to save results incrementally (overwrites input file)
            batch_size: Number of entries to process in parallel
            
        Returns:
            List of entries with leakage removed
        """
        # Filter entries that need processing
        pending_entries = []
        pending_indices = []
        
        for i, entry in enumerate(entries):
            if self.needs_leakage_removal(entry):
                pending_entries.append(entry)
                pending_indices.append(i)
        
        logger.info(f"Found {len(pending_entries)} entries that need leakage removal out of {len(entries)} total")
        
        if not pending_entries:
            logger.info("No entries need leakage removal processing")
            return entries
        
        # Process entries in batches
        for i in range(0, len(pending_entries), batch_size):
            batch_entries = pending_entries[i:i+batch_size]
            batch_indices = pending_indices[i:i+batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(pending_entries) + batch_size - 1)//batch_size}...")
            
            # Extract questions and prepare prompts
            questions = []
            valid_indices = []
            
            for j, entry in enumerate(batch_entries):
                question_text = self.extract_question_from_entry(entry)
                if question_text and len(question_text.strip()) > 10:
                    questions.append(question_text)
                    valid_indices.append(j)
                else:
                    logger.warning(f"Skipping entry: No valid question found")
            
            if not questions:
                logger.info("No valid questions in this batch, skipping...")
                continue
            
            # Prepare leakage check prompts using the question generator's method
            leakage_prompts = self.question_generator._prepare_leakage_check_prompts(
                questions, batch_entries[:len(questions)]
            )
            
            # Filter out SKIP prompts
            valid_prompts = []
            prompt_to_entry_map = []
            for idx, prompt in enumerate(leakage_prompts):
                if prompt != "SKIP":
                    valid_prompts.append(prompt)
                    prompt_to_entry_map.append(valid_indices[idx])
            
            if not valid_prompts:
                logger.info("No valid prompts in this batch, skipping...")
                continue
            
            try:
                # Generate corrected questions
                corrected_texts = await self.inference_engine.generate(
                    valid_prompts, 
                    batch_size=len(valid_prompts)
                )
                
                # Update entries with corrected questions
                for prompt_idx, corrected_text in enumerate(corrected_texts):
                    if prompt_idx < len(prompt_to_entry_map):
                        entry_idx = prompt_to_entry_map[prompt_idx]
                        entry = batch_entries[entry_idx]
                        original_idx = batch_indices[entry_idx]
                        
                        # Check if corrected_text is valid (not None or empty)
                        if corrected_text is not None and len(str(corrected_text).strip()) > 10:
                            # Store the corrected question
                            entry['leakage_removed_question'] = self.question_generator.extract_final_question(corrected_text)
                            # entry['leakage_removal_response'] = corrected_text
                            entry['leakage_removed'] = 1
                            
                            # Update the original entry in the full list
                            entries[original_idx] = entry
                            
                            # logger.info(f"Successfully processed entry {original_idx + 1}/{len(entries)}")
                        else:
                            # Handle None or empty responses
                            entry['leakage_removed'] = 0
                            entry['leakage_removal_error'] = "Received None or empty response from inference engine"
                            entries[original_idx] = entry
                            logger.warning(f"Failed to process entry {original_idx + 1}/{len(entries)}: Received None or empty response")
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                # Mark failed entries
                for j in valid_indices:
                    entry = batch_entries[j]
                    original_idx = batch_indices[j]
                    entry['leakage_removed'] = 0
                    entry['leakage_removal_error'] = str(e)
                    entries[original_idx] = entry
            
            # Save results incrementally after each batch
            logger.info(f"Saving results after batch {i//batch_size + 1}...")
            self.save_jsonl(entries, input_file)
        
        logger.info("Finished leakage removal processing")
        return entries

async def main():
    parser = argparse.ArgumentParser(description="Remove leakage from forecasting questions in JSONL file")
    parser.add_argument("--input_file", help="Input JSONL file path (will be modified in-place)")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for processing")
    parser.add_argument("--use_freeq", default=True, help="Process free-form questions")
    
    args = parser.parse_args()
    
    # Import and initialize the inference engine
    # You'll need to replace this with your actual inference engine initialization
    try:
        # from inference_engines import OpenRouterInference  # Replace with your actual import
        
        leakage_engine = OpenRouterInference(
            # model="deepseek/deepseek-r1-0528",
            # model="qwen/qwen3-32b",
            model="meta-llama/llama-4-maverick",
            max_tokens=10000,
            temperature=0.6
        )
        inference_engine = leakage_engine
    except ImportError:
        logger.error("Please implement your inference engine import and initialization")
        return
    
    # Initialize leakage remover
    remover = LeakageRemover(
        inference_engine=inference_engine,
        use_freeq=args.use_freeq
    )
    
    # Load input data
    entries = remover.load_jsonl(args.input_file)
    if not entries:
        logger.error("No data loaded, exiting")
        return
    
    # Remove leakage (with incremental saving to input file)
    processed_entries = await remover.remove_leakage_from_entries(
        entries, 
        input_file=args.input_file,
        batch_size=args.batch_size
    )
    
    # Final save to ensure all results are saved
    remover.save_jsonl(processed_entries, args.input_file)
    
    # Print summary
    total_processed = sum(1 for entry in processed_entries if entry.get('leakage_removed', 0) == 1)
    total_failed = sum(1 for entry in processed_entries if entry.get('leakage_removed', 0) == 0)
    
    logger.info(f"Summary: {total_processed} entries processed successfully, {total_failed} failed")

if __name__ == "__main__":
    asyncio.run(main())
