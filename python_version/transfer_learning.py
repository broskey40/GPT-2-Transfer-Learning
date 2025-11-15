"""
Exercise 20.5: Transfer Learning for Domain Adaptation in Python
Using Transformers with Qwen2-0.5B model

This implementation demonstrates:
1. Loading a pre-trained LLM (Qwen2-0.5B)
2. Fine-tuning on domain-specific data (Rust programming)
3. Multiple transfer learning strategies (full fine-tuning vs layer freezing)
4. Performance evaluation before and after adaptation
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import numpy as np
from typing import Dict, List, Tuple
import math
from tqdm import tqdm

# ============================================================================
# DATASET PREPARATION
# ============================================================================

def prepare_domain_dataset() -> str:
    """Domain-specific dataset: Rust programming concepts"""
    return """
    In Rust, ownership is a set of rules that govern memory management.
    The Vec<T> type is a growable array stored on the heap.
    Result<T, E> is used for error handling in Rust programs.
    The match expression enables pattern matching on enums.
    Traits define shared behavior in an abstract way.
    The borrow checker ensures memory safety at compile time.
    Cargo is the Rust package manager and build system.
    impl blocks define methods on structs and enums.
    The Option<T> enum handles the presence or absence of values.
    Lifetimes ensure references are valid for their entire scope.
    The String type is a growable UTF-8 encoded string.
    Closures are anonymous functions that can capture their environment.
    The derive attribute automatically implements traits.
    Async/await syntax enables asynchronous programming.
    The candle crate provides tensor operations for machine learning.
    Error propagation uses the ? operator for Result types.
    Generic types enable code reuse with type parameters.
    The println! macro prints formatted text to stdout.
    References borrow values without taking ownership.
    Mutable references allow modification of borrowed data.
    """

def prepare_test_dataset() -> str:
    """Separate test set for evaluation"""
    return """
    Rust's ownership system prevents data races at compile time.
    The HashMap<K, V> type stores key-value pairs.
    Iterators provide a way to process sequences of elements.
    The std::fs module provides file system operations.
    Smart pointers like Box<T> enable heap allocation.
    """

def tokenize_dataset(
    text: str,
    tokenizer,
    block_size: int = 128
) -> Dataset:
    """Tokenize text and create dataset"""
    # Tokenize the text
    tokens = tokenizer(text, truncation=True, padding=False)

    # Create input-label pairs for language modeling
    input_ids = tokens['input_ids']

    # Create blocks of text
    examples = []
    for i in range(0, len(input_ids) - block_size, block_size):
        chunk = input_ids[i:i + block_size + 1]
        if len(chunk) == block_size + 1:
            examples.append({
                'input_ids': chunk[:-1],
                'labels': chunk[1:]
            })

    return Dataset.from_list(examples)

# ============================================================================
# TRANSFER LEARNING STRATEGIES
# ============================================================================

class TransferStrategy:
    """Enum-like class for transfer learning strategies"""
    FULL_FINE_TUNE = "Full Fine-Tuning"
    FREEZE_LOWER_LAYERS = "Freeze Lower Layers"
    FREEZE_EMBEDDINGS = "Freeze Embeddings"
    ADAPTER_LAYERS = "Adapter Layers"

def apply_transfer_strategy(model, strategy: str, n_layers: int = 24):
    """
    Apply transfer learning strategy by freezing/unfreezing parameters

    Args:
        model: The model to apply strategy to
        strategy: One of the TransferStrategy options
        n_layers: Total number of transformer layers
    """
    # First, unfreeze everything
    for param in model.parameters():
        param.requires_grad = True

    if strategy == TransferStrategy.FULL_FINE_TUNE:
        print(f"Strategy: {strategy} (all parameters)")
        # All parameters trainable
        pass

    elif strategy == TransferStrategy.FREEZE_LOWER_LAYERS:
        print(f"Strategy: {strategy} (freeze first {n_layers // 2} layers)")
        # Freeze first half of layers
        for name, param in model.named_parameters():
            # Freeze layers 0 to n_layers//2 - 1
            for i in range(n_layers // 2):
                if f"layers.{i}." in name:
                    param.requires_grad = False

    elif strategy == TransferStrategy.FREEZE_EMBEDDINGS:
        print(f"Strategy: {strategy}")
        # Freeze only embedding layers
        for name, param in model.named_parameters():
            if "embed" in name.lower():
                param.requires_grad = False

    elif strategy == TransferStrategy.ADAPTER_LAYERS:
        print(f"Strategy: {strategy} (train last 2 layers + head)")
        # Freeze all except last 2 layers and LM head
        for name, param in model.named_parameters():
            param.requires_grad = False
            # Unfreeze last 2 layers
            for i in range(n_layers - 2, n_layers):
                if f"layers.{i}." in name:
                    param.requires_grad = True
            # Unfreeze LM head and layer norm
            if "lm_head" in name or "ln_f" in name or "norm" in name.lower():
                param.requires_grad = True

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    return trainable_params

# ============================================================================
# EVALUATION & GENERATION
# ============================================================================

def calculate_perplexity(model, tokenizer, eval_dataset, device) -> float:
    """Calculate perplexity on evaluation dataset"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for example in eval_dataset:
            input_ids = torch.tensor([example['input_ids']]).to(device)
            labels = torch.tensor([example['labels']]).to(device)

            outputs = model(input_ids, labels=labels)
            loss = outputs.loss

            total_loss += loss.item() * len(example['input_ids'])
            total_tokens += len(example['input_ids'])

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return perplexity

def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 30,
    device: str = "cpu"
) -> str:
    """Generate text from a prompt using greedy sampling"""
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_on_prompts(model, tokenizer, prompts: List[str], device: str):
    """Evaluate model on multiple prompts"""
    print("\n=== Model Evaluation ===")
    for prompt in prompts:
        print(f"\nPrompt: \"{prompt}\"")
        try:
            generated = generate_text(model, tokenizer, prompt, device=device)
            print(f"Generated: \"{generated}\"")
        except Exception as e:
            print(f"Error: {e}")

# ============================================================================
# TRAINING
# ============================================================================

def train_model(
    model,
    train_dataset,
    eval_dataset,
    tokenizer,
    learning_rate: float = 1e-4,
    num_epochs: int = 20,
    device: str = "cpu"
) -> Tuple[List[float], float]:
    """Train the model and return training losses and final perplexity"""

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=learning_rate,
        warmup_steps=0,
        logging_steps=5,
        save_strategy="no",
        eval_strategy="no",  # Changed from evaluation_strategy
        report_to="none",
        disable_tqdm=False,
    )

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Train
    print(f"\nTraining with learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print(f"\n{'Epoch':<8} {'Train Loss':<12}")
    print("-" * 25)

    trainer.train()

    # Get training history
    losses = [log['loss'] for log in trainer.state.log_history if 'loss' in log]

    # Calculate final perplexity
    final_perplexity = calculate_perplexity(model, tokenizer, eval_dataset, device)

    return losses, final_perplexity

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    print("=" * 62)
    print("  Exercise 20.5: Transfer Learning for Domain Adaptation")
    print("  Using Transformers + Qwen2-0.5B (Python Version)")
    print("=" * 62 + "\n")

    # ========================================================================
    # Setup
    # ========================================================================
    print("[SETUP] Setting up environment...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")

    # ========================================================================
    # Load Pre-trained Model
    # ========================================================================
    print("\n[LOADING] Loading pre-trained Qwen2-0.5B model...")

    model_name = "Qwen/Qwen2-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("   [OK] Tokenizer loaded")

    # Load base model (we'll reload it for each strategy)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None
    )
    base_model = base_model.to(device)
    print("   [OK] Model loaded")

    # Get model configuration
    config = base_model.config
    n_layers = config.num_hidden_layers
    print(f"   [OK] Model has {n_layers} layers")

    # ========================================================================
    # Prepare Datasets
    # ========================================================================
    print("\n[DATA] Preparing domain-specific datasets...")
    train_text = prepare_domain_dataset()
    test_text = prepare_test_dataset()

    block_size = 32  # Sequence length
    train_dataset = tokenize_dataset(train_text, tokenizer, block_size)
    eval_dataset = tokenize_dataset(test_text, tokenizer, block_size)

    print(f"   Train examples: {len(train_dataset)}")
    print(f"   Test examples: {len(eval_dataset)}")

    # ========================================================================
    # Baseline Evaluation (Before Fine-tuning)
    # ========================================================================
    print("\n[EVAL] Baseline Evaluation (Pre-trained model)")
    baseline_perplexity = calculate_perplexity(
        base_model, tokenizer, eval_dataset, device
    )
    print(f"   Initial Perplexity: {baseline_perplexity:.2f}")

    eval_prompts = [
        "In Rust, Vec<T> is",
        "The Result type in Rust",
        "Ownership in Rust means",
    ]
    evaluate_on_prompts(base_model, tokenizer, eval_prompts, device)

    # ========================================================================
    # Experiment with Different Transfer Learning Strategies
    # ========================================================================
    print("\n\n" + "=" * 62)
    print("           TRANSFER LEARNING STRATEGY COMPARISON")
    print("=" * 62)

    strategies = [
        TransferStrategy.FULL_FINE_TUNE,
        TransferStrategy.FREEZE_LOWER_LAYERS,
        TransferStrategy.FREEZE_EMBEDDINGS,
        TransferStrategy.ADAPTER_LAYERS,
    ]

    results = {}

    for strategy in strategies:
        print("\n" + "=" * 60)
        print(f" {strategy}")
        print("=" * 60)

        # Reload model for each strategy
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None
        )
        model = model.to(device)

        # Apply strategy
        apply_transfer_strategy(model, strategy, n_layers)

        # Train
        losses, final_perplexity = train_model(
            model,
            train_dataset,
            eval_dataset,
            tokenizer,
            learning_rate=1e-4,
            num_epochs=20,
            device=device
        )

        print(f"\n[RESULT] Final Perplexity: {final_perplexity:.2f}")
        results[strategy] = (losses, final_perplexity)

        # Generate samples after training
        evaluate_on_prompts(model, tokenizer, eval_prompts, device)

        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ========================================================================
    # Final Results Summary
    # ========================================================================
    print("\n\n" + "=" * 62)
    print("                    RESULTS SUMMARY")
    print("=" * 62)
    print(f"\nBaseline (Pre-trained): {baseline_perplexity:.2f} perplexity\n")
    print(f"{'Strategy':<25} {'Final Perplexity':<15} {'Improvement':<15}")
    print("-" * 60)

    for strategy, (_, perplexity) in results.items():
        improvement = ((baseline_perplexity - perplexity) / baseline_perplexity) * 100.0
        print(f"{strategy:<25} {perplexity:<15.2f} {improvement:<15.1f}%")

    print("\n\nKey Findings:")
    print("   - Full fine-tuning typically achieves lowest perplexity")
    print("   - Freezing strategies reduce training time & overfitting risk")
    print("   - Adapter layers offer good trade-off for limited data")
    print("   - Domain adaptation successful with minimal training data\n")

if __name__ == "__main__":
    main()
