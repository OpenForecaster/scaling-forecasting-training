#!/usr/bin/env python3
"""
Script to reorganize checkpoint directories by creating actor subdirectories.

This script:
1. Finds all global_step_* directories in the input directory
2. For each directory that doesn't have an 'actor' subdirectory:
   - Creates an 'actor' directory
   - Moves all files except 'data.pt' into the 'actor' directory
   - Keeps 'data.pt' in the original location

Usage:
    python reorganize_checkpoints.py --input_dir /path/to/checkpoints
"""

import os
import shutil
import argparse
import logging
from pathlib import Path
from typing import List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_global_step_dirs(input_dir: str) -> List[str]:
    """
    Find all global_step_* directories in the input directory.
    
    Args:
        input_dir: Root directory to search for global_step directories
        
    Returns:
        List of paths to global_step directories
    """
    global_step_dirs = []
    
    for root, dirs, files in os.walk(input_dir):
        for dir_name in dirs:
            if dir_name.startswith('global_step_'):
                global_step_dirs.append(os.path.join(root, dir_name))
    
    return global_step_dirs

def needs_reorganization(global_step_dir: str) -> bool:
    """
    Check if a global_step directory needs reorganization.
    
    Args:
        global_step_dir: Path to the global_step directory
        
    Returns:
        True if the directory needs reorganization (no actor subdirectory)
    """
    actor_dir = os.path.join(global_step_dir, 'actor')
    return not os.path.exists(actor_dir)

def reorganize_global_step_dir(global_step_dir: str) -> bool:
    """
    Reorganize a single global_step directory by creating an actor subdirectory
    and moving all files except data.pt into it.
    
    Args:
        global_step_dir: Path to the global_step directory
        
    Returns:
        True if reorganization was successful, False otherwise
    """
    try:
        # Create actor directory
        actor_dir = os.path.join(global_step_dir, 'actor')
        os.makedirs(actor_dir, exist_ok=True)
        logger.info(f"Created actor directory: {actor_dir}")
        
        # Get all items in the global_step directory
        items = os.listdir(global_step_dir)
        
        # Move all items except 'data.pt' and the 'actor' directory itself
        moved_count = 0
        for item in items:
            if item == 'data.pt' or item == 'actor':
                continue
                
            source_path = os.path.join(global_step_dir, item)
            dest_path = os.path.join(actor_dir, item)
            
            if os.path.isfile(source_path):
                shutil.move(source_path, dest_path)
                logger.info(f"Moved file: {item}")
                moved_count += 1
            elif os.path.isdir(source_path):
                shutil.move(source_path, dest_path)
                logger.info(f"Moved directory: {item}")
                moved_count += 1
        
        logger.info(f"Successfully reorganized {global_step_dir}: moved {moved_count} items")
        return True
        
    except Exception as e:
        logger.error(f"Failed to reorganize {global_step_dir}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Reorganize checkpoint directories by creating actor subdirectories")
    parser.add_argument("--input_dir", type=str, default="/fast/nchandak/forecasting/training/verl/checkpoints/distill-grok-3-mini/",
                       help="Input directory containing checkpoints")
    parser.add_argument("--dry_run", action="store_true",
                       help="Show what would be done without actually doing it")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return
    
    logger.info(f"Processing checkpoints in: {args.input_dir}")
    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")
    
    # Find all global_step directories
    global_step_dirs = find_global_step_dirs(args.input_dir)
    logger.info(f"Found {len(global_step_dirs)} global_step directories")
    
    # Filter directories that need reorganization
    dirs_to_reorganize = [d for d in global_step_dirs if needs_reorganization(d)]
    logger.info(f"Found {len(dirs_to_reorganize)} directories that need reorganization")
    
    if not dirs_to_reorganize:
        logger.info("No directories need reorganization")
        return
    
    # Show what will be reorganized
    logger.info("Directories to be reorganized:")
    for dir_path in dirs_to_reorganize:
        logger.info(f"  - {dir_path}")
    
    if args.dry_run:
        logger.info("Dry run complete - no changes made")
        return
    
    # Reorganize directories
    success_count = 0
    failed_count = 0
    
    for global_step_dir in dirs_to_reorganize:
        if reorganize_global_step_dir(global_step_dir):
            success_count += 1
        else:
            failed_count += 1
    
    logger.info(f"Reorganization complete: {success_count} successful, {failed_count} failed")
    
    if success_count > 0:
        logger.info("Reorganized directories now have the following structure:")
        logger.info("  global_step_XXX/")
        logger.info("    ├── data.pt")
        logger.info("    └── actor/")
        logger.info("        ├── model_world_size_*_rank_*.pt")
        logger.info("        ├── optim_world_size_*_rank_*.pt")
        logger.info("        ├── extra_state_world_size_*_rank_*.pt")
        logger.info("        ├── fsdp_config.json")
        logger.info("        └── huggingface/")

if __name__ == "__main__":
    main()
