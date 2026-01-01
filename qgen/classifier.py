import pandas as pd
import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    AutoConfig
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import torch
import json
import os
# Set your W&B API key before importing wandb
# os.environ["WANDB_API_KEY"] = ""  # Replace with your actual API key
import wandb
from typing import Dict, List
from accelerate import Accelerator
import argparse

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return [{
        'date_resolve_at': item['date_resolve_at'],
        'text': item['prompt'],
        'label': item['answer_idx']
    } for item in data]

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model(
    train_dataset, 
    eval_dataset, 
    model_name: str,
    output_dir: str,
    freeze_base_model: bool = False,
    num_train_epochs: int = 10,
    random_init: bool = False,
    gradacc_steps: int = 2
) -> Dict[str, float]:
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if random_init:
        if accelerator.is_main_process:
            print(f"Initializing model {model_name} from random weights.")
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=4,  # For A, B, C, D options
            problem_type="single_label_classification"
        )
        model = AutoModelForSequenceClassification.from_config(config)
    else:
        if accelerator.is_main_process:
            print(f"Loading pretrained model {model_name}.")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=4,  # For A, B, C, D options
            problem_type="single_label_classification"
        )
    
    # Freeze base model parameters if requested
    print("freeze_base_model: ", freeze_base_model)
    if freeze_base_model:
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze classification head parameters
        for param in model.classifier.parameters():
            param.requires_grad = True
        
        if accelerator.is_main_process:
            print("Base model frozen. Only training classification head.")

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding="max_length",
            truncation=True,
            max_length=512
        )

    # Tokenize datasets
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

    # Calculate class weights
    labels = train_dataset['label']
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    class_weights = torch.FloatTensor([total_samples / (len(class_counts) * count) for count in class_counts])

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=gradacc_steps,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=50,
        metric_for_best_model="accuracy",
        save_total_limit=1,
        run_name="deepseek-mcq",
        warmup_ratio=0.1,
        logging_steps=10,
        report_to="wandb",
        save_strategy="no",
        lr_scheduler_type="cosine"
    )

    # Create custom trainer with weighted loss
    class WeightedTrainer(Trainer):
        def __init__(self, class_weights, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights.to(self.args.device)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Apply weighted cross entropy loss
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            
            return (loss, outputs) if return_outputs else loss

    # Initialize trainer with class weights
    trainer = Trainer(
        # class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
    )

    # Prepare everything with accelerator
    trainer = accelerator.prepare(trainer)

    # Train the model
    trainer.train()
    
    # Evaluate on validation set
    eval_results = trainer.evaluate()
    
    return eval_results

def load_and_split_data(file_path, train_ratio=0.6, test_ratio=0.3):
    """
    Load data and split it into train, validation, and test sets based on date_resolve_at.
    
    Args:
        file_path: Path to the JSON data file
        train_ratio: Proportion of data for training (default: 0.6)
        test_ratio: Proportion of data for testing (default: 0.3)
        
    Returns:
        train_data, val_data, test_data as lists
    """
    # Calculate validation ratio
    val_ratio = 1.0 - train_ratio - test_ratio
    
    # Ensure ratios are valid
    if train_ratio <= 0 or test_ratio <= 0 or val_ratio <= 0 or train_ratio + test_ratio + val_ratio != 1.0:
        raise ValueError("Invalid ratios: train_ratio + test_ratio + val_ratio must equal 1.0")
    
    # Load data
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert to list of dictionaries with required fields
    processed_data = [{
        'date_resolve_at': item['date_resolve_at'],
        'text': item['prompt'],
        'label': item['answer_idx']
    } for item in data]
    
    # Sort by date_resolve_at
    processed_data.sort(key=lambda x: x['date_resolve_at'])
    
    # Calculate split indices
    total_count = len(processed_data)
    train_end = int(total_count * train_ratio)
    val_end = train_end + int(total_count * val_ratio)
    
    # Split data
    train_data = processed_data[:train_end]
    val_data = processed_data[train_end:val_end]
    test_data = processed_data[val_end:]
    
    # Print first and last data points from each split
    print("\nFirst train data point:")
    print(f"Date: {train_data[0]['date_resolve_at']}")
    print(f"Text: {train_data[0]['text'][:100]}...")
    print(f"Label: {train_data[0]['label']}")
    
    print("\nLast train data point:")
    print(f"Date: {train_data[-1]['date_resolve_at']}")
    print(f"Text: {train_data[-1]['text'][:100]}...")
    print(f"Label: {train_data[-1]['label']}")
    
    print("\nFirst validation data point:")
    print(f"Date: {val_data[0]['date_resolve_at']}")
    print(f"Text: {val_data[0]['text'][:100]}...")
    print(f"Label: {val_data[0]['label']}")
    
    print("\nLast validation data point:")
    print(f"Date: {val_data[-1]['date_resolve_at']}")
    print(f"Text: {val_data[-1]['text'][:100]}...")
    print(f"Label: {val_data[-1]['label']}")
    
    print("\nFirst test data point:")
    print(f"Date: {test_data[0]['date_resolve_at']}")
    print(f"Text: {test_data[0]['text'][:100]}...")
    print(f"Label: {test_data[0]['label']}")
    
    print("\nLast test data point:")
    print(f"Date: {test_data[-1]['date_resolve_at']}")
    print(f"Text: {test_data[-1]['text'][:100]}...")
    print(f"Label: {test_data[-1]['label']}")
    
    return train_data, val_data, test_data

def main(args):
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Load and split dataset
    train_data, val_data, test_data = load_and_split_data(
        args.data_path, 
        args.train_ratio, 
        args.test_ratio
    )
    
    print("freeze_base_model: ", args.freeze_base_model)
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)
    
    # print length of each dataset
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Val dataset length: {len(val_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")
    
    # Initialize wandb only on the main process
    if accelerator.is_main_process:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    
    # Train on combined train and validation data, evaluate on test
    if accelerator.is_main_process:
        print("Training model on training data...")
    
    # First train on training data and evaluate on validation data
    test_results = train_model(
        train_dataset,
        test_dataset,
        args.model_name,
        output_dir=args.output_dir,
        freeze_base_model=args.freeze_base_model,
        num_train_epochs=args.num_train_epochs,
        random_init=args.random_init,
        gradacc_steps=args.gradacc_steps
    )
    
    # Log test results
    if accelerator.is_main_process:
        wandb.log({
            "test_accuracy": test_results['eval_accuracy'],
            "test_f1": test_results['eval_f1'],
            "test_precision": test_results['eval_precision'],
            "test_recall": test_results['eval_recall']
        })
        
        print("\nTest performance:")
        print(test_results)
        
        wandb.finish()
    
    return test_results

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train a classifier for multiple-choice questions")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, 
                        default="/is/cluster/fast/sgoel/forecasting/qgen/reuters/mcq/selected10_2021-2022_deepseekv3024",
                        help="Path to the JSON data file")
    parser.add_argument("--train_ratio", type=float, default=0.6,
                        help="Proportion of data for training")
    parser.add_argument("--test_ratio", type=float, default=0.3,
                        help="Proportion of data for testing")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-large",
                        help="Name of the pre-trained model to use")
    parser.add_argument("--output_dir", type=str, default="./results/dsv3-mcq-1k/test",
                        help="Directory to save model outputs")
    parser.add_argument("--freeze_base_model", action="store_true",
                        help="If set, freeze base model and train only the classification head")
    parser.add_argument("--num_train_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--random_init", action="store_true",
                        help="If set, initialize model with random weights instead of pretrained")
    parser.add_argument("--gradacc_steps", type=int, default=2,
                        help="Number of gradient accumulation steps")
    
    # Wandb parameters
    parser.add_argument("--wandb_project", type=str, default="mcq-classifier",
                        help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default="dsv3-mcq-1k",
                        help="Wandb run name")
    
    # Parse arguments
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    # Enable deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    test_results = main(args)
    print("\nTest Performance:")
    print(test_results)