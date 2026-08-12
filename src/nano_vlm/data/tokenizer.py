# ============================================================
# NanoVLM Tokenizer
# ============================================================

import torch


# ------------------------------------------------------------
# Special tokens
# ------------------------------------------------------------

SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
]


# ------------------------------------------------------------
# Dataset vocabulary
# ------------------------------------------------------------

DATASET_WORDS = [
    "red",
    "green",
    "blue",
    "yellow",
    "purple",
    "orange",
    "pink",
    "brown",
    "gray",

    "square",
    "circle",
    "triangle",

    "top-left",
    "top-center",
    "top-right",

    "middle-left",
    "center",
    "middle-right",

    "bottom-left",
    "bottom-center",
    "bottom-right",
]


# ------------------------------------------------------------
# Vocabulary
# ------------------------------------------------------------

VOCAB = SPECIAL_TOKENS + DATASET_WORDS

stoi = {
    token: index
    for index, token in enumerate(VOCAB)
}

itos = {
    index: token
    for token, index in stoi.items()
}


# ------------------------------------------------------------
# Vocabulary configuration
# ------------------------------------------------------------

VOCAB_SIZE = len(VOCAB)

PAD_TOKEN_ID = stoi["<PAD>"]
UNK_TOKEN_ID = stoi["<UNK>"]
BOS_TOKEN_ID = stoi["<BOS>"]
EOS_TOKEN_ID = stoi["<EOS>"]


# ------------------------------------------------------------
# Tokenize one caption
# ------------------------------------------------------------

def tokenize_caption(
    caption,
    max_length,
):
    """
    Convert one caption into a fixed-length
    tensor of token IDs.

    Output shape:
        [max_length]
    """

    words = caption.lower().split()

    # Start with BOS token.
    token_ids = [BOS_TOKEN_ID]

    # Convert words to token IDs.
    for word in words:

        token_ids.append(
            stoi.get(
                word,
                UNK_TOKEN_ID,
            )
        )

    # End with EOS token.
    token_ids.append(EOS_TOKEN_ID)

    # Truncate if necessary.
    token_ids = token_ids[:max_length]

    # Pad to the fixed context window.
    token_ids += [
        PAD_TOKEN_ID
    ] * (
        max_length - len(token_ids)
    )

    return torch.tensor(
        token_ids,
        dtype=torch.long,
    )


# ------------------------------------------------------------
# Tokenize a batch of captions
# ------------------------------------------------------------

def tokenize_captions(
    captions,
    max_length,
):
    """
    Tokenize a batch of caption strings.

    Input:
        list[str]

    Output:
        [B, L]
    """

    return torch.stack(
        [
            tokenize_caption(
                caption,
                max_length=max_length,
            )
            for caption in captions
        ]
    )