# Hybrid GPU/CPU Training Strategy

## Best of Both Worlds: Performance + Flexibility

This guide provides strategies to maximize GPU performance while gracefully handling memory limitations through CPU fallback and optimization techniques.

---

## Table of Contents
1. [Strategy Overview](#strategy-overview)
2. [Memory Optimization Techniques](#memory-optimization-techniques)
3. [Hybrid Approaches](#hybrid-approaches)
4. [Implementation Examples](#implementation-examples)
5. [Performance Comparison](#performance-comparison)

---

## Strategy Overview

### The Challenge
- **GPU Advantage**: 10-100x faster than CPU for parallel operations
- **GPU Limitation**: Limited VRAM (8-16GB consumer GPUs)
- **Large Models**: 500M+ parameters need 8-10GB+ for training

### The Solution: Multi-Tier Approach

```
┌─────────────────────────────────────────────────┐
│         Tier 1: Memory Optimization             │
│  • FP16/BF16 instead of FP32 (50% memory)      │
│  • Gradient Checkpointing (trade compute)       │
│  • Smaller batch sizes                          │
└─────────────────────────────────────────────────┘
                    ↓ If still OOM
┌─────────────────────────────────────────────────┐
│      Tier 2: Selective GPU/CPU Placement        │
│  • Model on GPU, Gradients on CPU              │
│  • Inference on GPU, Training on CPU           │
│  • Model parallelism (split across devices)    │
└─────────────────────────────────────────────────┘
                    ↓ If still issues
┌─────────────────────────────────────────────────┐
│         Tier 3: Full CPU Fallback               │
│  • Automatic detection and fallback             │
│  • Use GPU for inference only                   │
│  • Train on CPU with larger batch sizes        │
└─────────────────────────────────────────────────┘
```

---

## Memory Optimization Techniques

### 1. Mixed Precision Training (FP16/BF16)

**Memory Savings**: 50%
**Speed**: Often faster on modern GPUs

#### Rust (Candle)
```rust
// Use FP16 instead of FP32
let vb = VarBuilder::from_varmap(&varmap, DType::F16, &device);

// Convert inputs to FP16
let inputs_fp16 = inputs.to_dtype(DType::F16)?;
```

#### Python (PyTorch)
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    logits = model(inputs)
    loss = criterion(logits, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Trade-off**: Slight numerical precision loss (usually negligible)

---

### 2. Gradient Checkpointing

**Memory Savings**: 30-50%
**Speed**: 10-20% slower

**How it works**: Recompute activations during backward pass instead of storing them.

#### Python (PyTorch)
```python
from torch.utils.checkpoint import checkpoint

class ModelWithCheckpointing(nn.Module):
    def forward(self, x):
        # Checkpoint expensive layers
        x = checkpoint(self.transformer_block, x)
        return x
```

#### Rust (Candle)
Note: Candle doesn't have built-in gradient checkpointing yet. Manual implementation needed.

---

### 3. Gradient Accumulation

**Memory Savings**: Allows smaller batch sizes
**Speed**: Same throughput with smaller batches

#### Rust (Candle)
```rust
let accumulation_steps = 4;
let micro_batch_size = total_batch_size / accumulation_steps;

optimizer.zero_grad();
for step in 0..accumulation_steps {
    let micro_batch = get_micro_batch(step, micro_batch_size);
    let loss = forward_pass(model, micro_batch);

    // Scale loss by accumulation steps
    let scaled_loss = (loss / accumulation_steps as f64)?;
    scaled_loss.backward()?;
}
optimizer.step()?;
```

#### Python (PyTorch)
```python
accumulation_steps = 4

optimizer.zero_grad()
for step, batch in enumerate(dataloader):
    outputs = model(batch)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

### 4. Reduce Model Parameters

**Memory Savings**: 90-95% with adapters
**Speed**: Much faster

**Approaches**:
- **LoRA** (Low-Rank Adaptation): Train small adapter matrices
- **Adapter Layers**: Freeze most layers, train only last 2-3
- **BitFit**: Train only bias parameters

#### Example: Adapter Layers
```rust
// Train only 6% of parameters
let trainable_params = all_vars.into_iter()
    .filter(|var| {
        let name = var.as_tensor().to_string();
        // Only last 2 layers + head
        name.contains("h.22") ||
        name.contains("h.23") ||
        name.contains("lm_head")
    })
    .collect();
```

---

## Hybrid Approaches

### Approach 1: Automatic GPU/CPU Fallback

**Best for**: Development and testing

```rust
fn get_device() -> Result<Device> {
    match Device::new_cuda(0) {
        Ok(device) => {
            println!("✓ Using CUDA GPU");
            Ok(device)
        },
        Err(_) => {
            println!("⚠ CUDA not available, falling back to CPU");
            Ok(Device::Cpu)
        }
    }
}

fn train_with_fallback(model: &mut Model, device: &Device) -> Result<()> {
    match train_on_device(model, device) {
        Ok(result) => Ok(result),
        Err(e) if is_oom_error(&e) => {
            println!("⚠ GPU OOM, retrying on CPU");
            let cpu_device = Device::Cpu;
            model.to_device(&cpu_device)?;
            train_on_device(model, &cpu_device)
        },
        Err(e) => Err(e)
    }
}
```

---

### Approach 2: Model Parallelism

**Best for**: Very large models

Split model across devices:

```python
# Python example with device_map
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B",
    device_map="auto",  # Automatically split across available devices
    max_memory={
        0: "7GB",  # GPU 0
        "cpu": "20GB"  # CPU RAM
    }
)
```

**How it works**:
- Layers 0-16: GPU
- Layers 17-24: CPU
- Automatic data movement during forward/backward

---

### Approach 3: Inference on GPU, Training on CPU

**Best for**: When GPU has space for model but not gradients

```rust
fn hybrid_training_loop(
    model: &mut Model,
    gpu: &Device,
    cpu: &Device,
) -> Result<()> {
    // Load model on GPU for fast inference
    model.to_device(gpu)?;

    for batch in training_data {
        // Forward pass on GPU (fast)
        let logits = model.forward(&batch.to_device(gpu)?)?;

        // Move to CPU for backward pass (memory-safe)
        let logits_cpu = logits.to_device(cpu)?;
        let loss = compute_loss(&logits_cpu, &batch.labels)?;

        // Backward on CPU
        optimizer.backward_step(&loss)?;

        // Update model on GPU
        model.to_device(gpu)?;
    }

    Ok(())
}
```

---

### Approach 4: CPU Offloading

**Best for**: Maximum memory efficiency

```python
# PyTorch with CPU offloading
import torch
from torch.distributed.fsdp import CPUOffload

model = FSDP(
    model,
    cpu_offload=CPUOffload(offload_params=True)
)
```

**How it works**:
- Parameters on CPU when not in use
- Move to GPU only during computation
- Automatic management

---

## Implementation Examples

### Rust: Complete Hybrid Strategy

```rust
// src/main.rs additions

use candle_core::{Device, DType, Error};

// Configuration
struct TrainingConfig {
    prefer_gpu: bool,
    use_fp16: bool,
    batch_size: usize,
    gradient_accumulation_steps: usize,
    fallback_to_cpu: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            prefer_gpu: true,
            use_fp16: true,  // Enable FP16 for 50% memory savings
            batch_size: 4,    // Smaller batch size
            gradient_accumulation_steps: 2,  // Effective batch = 8
            fallback_to_cpu: true,
        }
    }
}

fn setup_device(config: &TrainingConfig) -> Result<Device> {
    if !config.prefer_gpu {
        println!("📍 CPU mode selected by configuration");
        return Ok(Device::Cpu);
    }

    match Device::new_cuda(0) {
        Ok(device) => {
            println!("✓ CUDA GPU detected and initialized");
            println!("   Using FP16: {}", config.use_fp16);
            Ok(device)
        },
        Err(_) => {
            println!("⚠ CUDA not available");
            if config.fallback_to_cpu {
                println!("✓ Falling back to CPU");
                Ok(Device::Cpu)
            } else {
                Err(Error::Msg("CUDA required but not available".into()))
            }
        }
    }
}

fn get_dtype(config: &TrainingConfig, device: &Device) -> DType {
    if config.use_fp16 && device.is_cuda() {
        println!("📊 Using FP16 for 50% memory savings");
        DType::F16
    } else {
        println!("📊 Using FP32");
        DType::F32
    }
}

fn train_with_memory_management(
    model: &mut Qwen2ModelWithHead,
    train_data: &(Tensor, Tensor),
    trainable_params: Vec<candle_core::Var>,
    config: &TrainingConfig,
) -> Result<Vec<f32>> {
    // Try training
    match train_model_internal(model, train_data, trainable_params.clone(), config) {
        Ok(losses) => Ok(losses),
        Err(e) if is_cuda_oom(&e) && config.fallback_to_cpu => {
            println!("\n⚠ CUDA Out of Memory!");
            println!("🔄 Attempting CPU fallback...\n");

            // Move everything to CPU
            let cpu_device = Device::Cpu;
            // Re-initialize model on CPU
            // ... (implementation details)

            train_model_internal(model, train_data, trainable_params, config)
        },
        Err(e) => Err(e)
    }
}

fn is_cuda_oom(error: &Error) -> bool {
    match error {
        Error::Cuda(e) => e.to_string().contains("out of memory"),
        _ => false
    }
}
```

---

### Python: Enhanced Hybrid Strategy

```python
# python_version/hybrid_training.py

import torch
import torch.nn as nn
from typing import Optional, Tuple
from contextlib import contextmanager

class HybridTrainingConfig:
    def __init__(
        self,
        prefer_gpu: bool = True,
        use_mixed_precision: bool = True,
        cpu_offload: bool = False,
        max_gpu_memory_gb: Optional[float] = None,
        gradient_checkpointing: bool = False,
    ):
        self.prefer_gpu = prefer_gpu
        self.use_mixed_precision = use_mixed_precision
        self.cpu_offload = cpu_offload
        self.max_gpu_memory_gb = max_gpu_memory_gb
        self.gradient_checkpointing = gradient_checkpointing

def setup_hybrid_device(config: HybridTrainingConfig) -> str:
    """Setup device with automatic fallback"""
    if not config.prefer_gpu or not torch.cuda.is_available():
        print("📍 Using CPU")
        return "cpu"

    # Check available GPU memory
    if config.max_gpu_memory_gb:
        available_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if available_memory < config.max_gpu_memory_gb:
            print(f"⚠ GPU memory ({available_memory:.1f}GB) < required ({config.max_gpu_memory_gb}GB)")
            print("✓ Falling back to CPU")
            return "cpu"

    print(f"✓ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    return "cuda"

@contextmanager
def hybrid_training_context(model, config: HybridTrainingConfig):
    """Context manager for hybrid training setup"""
    try:
        # Enable gradient checkpointing
        if config.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("✓ Gradient checkpointing enabled (saves 30-50% memory)")

        # Setup mixed precision
        if config.use_mixed_precision and torch.cuda.is_available():
            print("✓ Mixed precision training enabled (FP16)")

        yield model

    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def load_model_with_device_map(
    model_name: str,
    config: HybridTrainingConfig
) -> Tuple[nn.Module, str]:
    """Load model with automatic device placement"""
    from transformers import AutoModelForCausalLM

    if config.cpu_offload:
        print("🔄 Loading model with CPU offloading")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            max_memory={
                0: "7GB",  # Leave 1GB for activations
                "cpu": "20GB"
            },
            offload_folder="offload",
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = setup_hybrid_device(config)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if config.use_mixed_precision else torch.float32,
        )
        model = model.to(device)

    return model, device

# Example usage
def train_with_hybrid_strategy():
    config = HybridTrainingConfig(
        prefer_gpu=True,
        use_mixed_precision=True,
        cpu_offload=True,  # Enable CPU offloading
        gradient_checkpointing=True,
    )

    model, device = load_model_with_device_map("Qwen/Qwen2-0.5B", config)

    with hybrid_training_context(model, config):
        # Training loop here
        pass
```

---

## Performance Comparison

### Memory Usage by Strategy

| Strategy | Memory (GB) | Relative | Speed |
|----------|------------|----------|-------|
| FP32 Full Model | 10.0 | 100% | 1.0x |
| **FP16 Full Model** | **5.0** | **50%** | **1.2x** |
| **FP16 + Grad Checkpoint** | **3.5** | **35%** | **1.0x** |
| FP16 + Adapter (6%) | 2.0 | 20% | 1.5x |
| **CPU Offload** | **1.5 (GPU)** | **15%** | **0.3x** |
| CPU Only | 4.0 (RAM) | N/A | 0.1x |

### Speed Comparison

| Configuration | GPU Time | CPU Time | Speedup |
|--------------|----------|----------|---------|
| Full FP32 (if fits) | 4 min | 40 min | 10x |
| FP16 Optimized | 3.5 min | 40 min | 11x |
| Adapter Only | 0.5 min | 4 min | 8x |
| Hybrid (CPU offload) | 12 min | 40 min | 3.3x |

---

## Recommended Configurations

### Configuration 1: Maximum Performance (16GB+ GPU)
```
- Device: CUDA
- Precision: FP16
- Strategy: Full fine-tuning
- Batch size: 8
```

### Configuration 2: Balanced (8GB GPU)
```
- Device: CUDA
- Precision: FP16
- Strategy: Adapter layers (last 2)
- Gradient checkpointing: Enabled
- Batch size: 4
```

### Configuration 3: Memory Constrained (<8GB GPU)
```
- Device: Hybrid (model on GPU, training on CPU)
- OR: CPU with larger batch sizes
- Strategy: Adapter layers
```

### Configuration 4: CPU Only
```
- Device: CPU
- Precision: FP32
- Strategy: Adapter layers for speed
- Batch size: 16 (more RAM available)
```

---

## Quick Start Guide

### Rust Implementation

1. **Update `Cargo.toml`** (already done)
2. **Modify `src/main.rs`**:
   ```rust
   let config = TrainingConfig::default();
   let device = setup_device(&config)?;
   let dtype = get_dtype(&config, &device);
   let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
   ```

3. **Run with automatic fallback**:
   ```bash
   cargo run --release
   ```

### Python Implementation

1. **Install dependencies**:
   ```bash
   pip install accelerate  # For device_map
   ```

2. **Update training script**:
   ```python
   from hybrid_training import HybridTrainingConfig, load_model_with_device_map

   config = HybridTrainingConfig(
       prefer_gpu=True,
       use_mixed_precision=True,
       cpu_offload=True
   )

   model, device = load_model_with_device_map("Qwen/Qwen2-0.5B", config)
   ```

3. **Run**:
   ```bash
   python transfer_learning.py
   ```

---

## Troubleshooting

### Issue: Still Running Out of Memory on GPU

**Solutions**:
1. Enable FP16: `use_fp16: true`
2. Reduce batch size: `batch_size: 2` or `1`
3. Use adapter strategy: Train only last 2 layers
4. Enable gradient checkpointing (Python)
5. Enable CPU offloading

### Issue: Training Too Slow on CPU

**Solutions**:
1. Use adapter strategy (90% fewer parameters)
2. Reduce epochs
3. Use larger batch sizes (CPU has more RAM)
4. Consider using a smaller model

### Issue: CUDA Out of Memory Errors Are Random

**Solutions**:
1. Add `torch.cuda.empty_cache()` between strategies
2. Restart Python kernel between runs
3. Use `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`

---

## Conclusion

### Best Practices

1. **Start with GPU + FP16 + Adapters**: 80% success rate
2. **Monitor memory usage**: Use `nvidia-smi` or `torch.cuda.memory_summary()`
3. **Implement fallback**: Always have CPU as backup
4. **Profile first**: Understand bottlenecks before optimizing

### Future Improvements

1. **LoRA Support**: Most memory-efficient fine-tuning
2. **8-bit/4-bit Quantization**: Run larger models
3. **Flash Attention**: Faster and more memory-efficient
4. **DeepSpeed/FSDP**: Advanced distributed training

---

**End of Guide**

*For questions or issues, refer to:*
- Candle: https://github.com/huggingface/candle
- PyTorch: https://pytorch.org/docs/stable/notes/cuda.html
- Transformers: https://huggingface.co/docs/transformers/perf_train_gpu_one
