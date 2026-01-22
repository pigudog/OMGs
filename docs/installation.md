# Installation Guide

This guide provides detailed installation instructions for OMGs, including system requirements, dependency installation, and LLM provider configuration.

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Python** | 3.10 | 3.10+ |
| **RAM** | 8 GB | 16+ GB |
| **Storage** | 5 GB | 20+ GB (with RAG index) |
| **GPU** | Not required | CUDA-compatible (for faster embeddings) |
| **OS** | Linux, macOS, Windows | Linux/macOS |

## Step 1: Clone Repository

```bash
git clone https://github.com/your-org/OMGs.git
cd OMGs
```

## Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Or using conda
conda create -n omgs python=3.10
conda activate omgs
```

## Step 3: Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install with specific versions
pip install openai>=1.0.0 chromadb>=0.4.0 torch>=2.0.0 tiktoken>=0.5.0
```

### Dependencies Overview

| Package | Version | Purpose |
|---------|---------|---------|
| `openai` | ≥1.0.0 | Multi-provider LLM API client (supports Azure, OpenAI, OpenRouter) |
| `chromadb` | ≥0.4.0 | Vector database for RAG |
| `langchain-huggingface` | ≥0.0.1 | Embedding model integration |
| `torch` | ≥2.0.0 | Deep learning framework |
| `tiktoken` | ≥0.5.0 | Token counting for budget management |
| `tqdm` | ≥4.65.0 | Progress bars |
| `prettytable` | ≥3.8.0 | Table rendering |
| `requests` | ≥2.28.0 | HTTP requests for PubMed API |

## Step 4: Configure LLM Provider

Set up environment variables for your chosen LLM provider(s). You can configure multiple providers and switch between them using the `--provider` argument.

### Azure OpenAI

**Linux/macOS:**
```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key-here"

# Add to ~/.bashrc or ~/.zshrc for persistence
echo 'export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"' >> ~/.bashrc
echo 'export AZURE_OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc
```

**Windows (PowerShell):**
```powershell
$env:AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com/"
$env:AZURE_OPENAI_API_KEY = "your-api-key-here"

# For persistence, add to system environment variables
[System.Environment]::SetEnvironmentVariable("AZURE_OPENAI_ENDPOINT", "https://your-resource.openai.azure.com/", "User")
[System.Environment]::SetEnvironmentVariable("AZURE_OPENAI_API_KEY", "your-api-key-here", "User")
```

**Windows (CMD):**
```cmd
set AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
set AZURE_OPENAI_API_KEY=your-api-key-here
```

### OpenAI Official API

**Linux/macOS:**
```bash
export OPENAI_API_KEY="your-openai-api-key"

# Add to ~/.bashrc or ~/.zshrc for persistence
echo 'export OPENAI_API_KEY="your-openai-api-key"' >> ~/.bashrc
```

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY = "your-openai-api-key"

# For persistence
[System.Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "your-openai-api-key", "User")
```

**Windows (CMD):**
```cmd
set OPENAI_API_KEY=your-openai-api-key
```

### OpenRouter

**Linux/macOS:**
```bash
export OPENROUTER_API_KEY="your-openrouter-api-key"

# Add to ~/.bashrc or ~/.zshrc for persistence
echo 'export OPENROUTER_API_KEY="your-openrouter-api-key"' >> ~/.bashrc
```

**Windows (PowerShell):**
```powershell
$env:OPENROUTER_API_KEY = "your-openrouter-api-key"

# For persistence
[System.Environment]::SetEnvironmentVariable("OPENROUTER_API_KEY", "your-openrouter-api-key", "User")
```

**Windows (CMD):**
```cmd
set OPENROUTER_API_KEY=your-openrouter-api-key
```

## Step 5: Verify Installation

```bash
# Check Python version
python --version

# Verify key imports
python -c "from utils import Color; print('Utils: OK')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import chromadb; print(f'ChromaDB: OK')"
```

## Step 6: Test LLM Provider Connections

Test your configured LLM providers:

```bash
# Test all configured providers
python -m clients.test_connection

# Test specific provider
python -m clients.test_connection --provider azure
python -m clients.test_connection --provider openai
python -m clients.test_connection --provider openrouter
```

## Next Steps

- See [Usage Guide](usage.md) for how to run the pipeline
- See [Configuration Guide](../config/README.md) for prompt and configuration customization
- See [Multi-Provider Guide](../clients/PROVIDERS.md) for detailed provider usage
