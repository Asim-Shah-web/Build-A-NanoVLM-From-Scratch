from pathlib import Path
import sys
import torch

# ============================================================
# Project Root / Python Path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# Tokenizer Configuration
# ============================================================

from nano_vlm.data.tokenizer import (
    VOCAB_SIZE,
    PAD_TOKEN_ID,
)

# ============================================================
# Project Paths
# ============================================================

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Main project directories
SRC_DIR = PROJECT_ROOT / "src"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_ARTIFACTS_DIR = ARTIFACTS_DIR / "models"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
ASSETS_DIR = PROJECT_ROOT / "assets"


# Create output directories if they do not exist
MODEL_ARTIFACTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Image Configuration
# ============================================================

IMAGE_CHANNELS = 3

IMAGE_SIZE = 32


# ============================================================
# Text Encoder Configuration
# ============================================================

CONTEXT_WINDOW = 16

EMBED_DIM = 64

ATTENTION_HEADS = 4


# ============================================================
# Shared Embedding Space
# ============================================================

PROJECTION_DIM = 64


# ============================================================
# Tokenizer Configuration
# ============================================================

# Imported from the centralized tokenizer.
# Do not redefine the vocabulary here.

TOKENIZER_VOCAB_SIZE = VOCAB_SIZE

TOKENIZER_PAD_TOKEN_ID = PAD_TOKEN_ID


# ============================================================
# Contrastive Learning Configuration
# ============================================================

TEMPERATURE = 0.07


# ============================================================
# Training Configuration
# ============================================================

BATCH_SIZE = 32

NUM_EPOCHS = 20

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4


# ============================================================
# Reproducibility
# ============================================================

SEED = 42


# ============================================================
# Checkpoint Configuration
# ============================================================

CHECKPOINT_NAME = "nanovlm_checkpoint.pt"

CHECKPOINT_PATH = (
    CHECKPOINTS_DIR / CHECKPOINT_NAME
)


# ============================================================
# Configuration Summary
# ============================================================

def print_config():
    """
    Print the main NanoVLM configuration.
    """

    print("=" * 60)
    print("NanoVLM Configuration")
    print("=" * 60)

    print("\nDevice:")
    print(DEVICE)

    print("\nImage:")
    print("Channels:", IMAGE_CHANNELS)
    print("Image size:", IMAGE_SIZE)

    print("\nText:")
    print("Vocabulary size:", TOKENIZER_VOCAB_SIZE)
    print("Context window:", CONTEXT_WINDOW)
    print("Embedding dimension:", EMBED_DIM)
    print("Attention heads:", ATTENTION_HEADS)

    print("\nShared embedding:")
    print("Projection dimension:", PROJECTION_DIM)

    print("\nContrastive learning:")
    print("Temperature:", TEMPERATURE)

    print("\nTraining:")
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", NUM_EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Weight decay:", WEIGHT_DECAY)

    print("\nPaths:")
    print("Project root:", PROJECT_ROOT)
    print("Artifacts:", ARTIFACTS_DIR)
    print("Models:", MODEL_ARTIFACTS_DIR)
    print("Figures:", FIGURES_DIR)
    print("Checkpoints:", CHECKPOINTS_DIR)

    print("=" * 60)
