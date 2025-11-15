# Transfer Learning for Domain Adaptation - Python Version

This is a Python implementation of the Rust transfer learning project, using the HuggingFace Transformers library with the Qwen2-0.5B model.

## Overview

This implementation demonstrates:
1. Loading a pre-trained LLM (Qwen2-0.5B)
2. Fine-tuning on domain-specific data (Rust programming concepts)
3. Multiple transfer learning strategies
4. Performance evaluation before and after adaptation

## Setup

### Prerequisites
- Python 3.8 or higher
- pipenv (recommended) or pip

### Installation

#### Option 1: Using Pipenv (Recommended)

1. **Install pipenv** (if not already installed):
   ```bash
   pip install pipenv
   ```

2. **Navigate to the python_version directory**:
   ```bash
   cd python_version
   ```

3. **Install dependencies and create virtual environment**:
   ```bash
   pipenv install
   ```

4. **Activate the virtual environment**:
   ```bash
   pipenv shell
   ```

   Alternatively, run commands without activating:
   ```bash
   pipenv run python transfer_learning.py
   ```

#### Option 2: Using pip and venv

1. **Create a virtual environment**:
   ```bash
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On Linux/Mac
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   For CUDA support (if you have an NVIDIA GPU):
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```

## Usage

Run the transfer learning experiment:

**With pipenv:**
```bash
cd python_version
pipenv run python transfer_learning.py
```

**With activated virtual environment (pipenv or venv):**
```bash
cd python_version
python transfer_learning.py
```

The script will:
1. Load the pre-trained Qwen2-0.5B model
2. Evaluate baseline performance
3. Test 4 different transfer learning strategies:
   - Full Fine-Tuning
   - Freeze Lower Layers
   - Freeze Embeddings
   - Adapter Layers
4. Display results and perplexity improvements

## Transfer Learning Strategies

### 1. Full Fine-Tuning
- **Description**: Train all model parameters
- **Use case**: When you have sufficient data and computational resources
- **Expected**: Best performance but highest computational cost

### 2. Freeze Lower Layers
- **Description**: Freeze the first 50% of transformer layers
- **Use case**: When lower layers capture general features
- **Expected**: Faster training, reduced overfitting risk

### 3. Freeze Embeddings
- **Description**: Only freeze embedding layers
- **Use case**: When vocabulary is similar to pre-training data
- **Expected**: Balanced approach

### 4. Adapter Layers
- **Description**: Freeze all except last 2 layers and LM head
- **Use case**: Limited data scenarios
- **Expected**: Efficient training, good generalization

## Output

The script outputs:
- Initial baseline perplexity
- Training progress for each strategy
- Final perplexity for each strategy
- Text generation samples
- Comparative results summary

## Project Structure

```
python_version/
├── README.md              # This file
├── requirements.txt       # Python dependencies
└── transfer_learning.py   # Main implementation
```

## Differences from Rust Version

| Aspect | Rust Version | Python Version |
|--------|-------------|----------------|
| Framework | Candle | HuggingFace Transformers |
| Device | CPU/CUDA | CPU/CUDA (auto-detected) |
| Training | Manual loop | Trainer API |
| Block Size | 8 tokens | 32 tokens |
| Dependencies | Cargo | pip |

## Performance Notes

- **CPU Mode**: Works but slower, good for testing
- **CUDA Mode**: Significantly faster, recommended for training
- **Memory**: ~2-3GB for Qwen2-0.5B model
- **Training Time**: ~2-5 minutes per strategy (CPU), ~30-60 seconds (GPU)

## Troubleshooting

### Out of Memory
If you encounter OOM errors:
- Reduce `block_size` in the script (line 269)
- Use smaller batch size in `TrainingArguments`
- Enable gradient checkpointing

### Slow Training
- Make sure you're using GPU if available
- Check device with: `python -c "import torch; print(torch.cuda.is_available())"`
- Install CUDA-enabled PyTorch if needed

### Model Download Issues
- Ensure you have internet connection
- The first run downloads ~1GB model from HuggingFace
- Models are cached in `~/.cache/huggingface/`

## License

This project is part of a transfer learning educational exercise.
