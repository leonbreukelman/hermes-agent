"""Lightweight Bedrock model metadata helpers.

This module intentionally has no boto3/lazy-dependency side effects. It is safe
for generic model metadata code to import when it only needs Bedrock context
windows, while the heavier ``agent.bedrock_adapter`` owns actual Converse API
runtime integration.
"""

from __future__ import annotations

from typing import Dict

# Bedrock's ListFoundationModels API doesn't expose context window sizes.
# Static fallback table for models where the Bedrock API doesn't expose
# context window sizes. AWS-imposed limits can differ from native providers
# (for example Claude models are 200K on Bedrock vs larger native windows).
BEDROCK_CONTEXT_LENGTHS: Dict[str, int] = {
    # Anthropic Claude models on Bedrock
    "anthropic.claude-opus-4-6": 200_000,
    "anthropic.claude-sonnet-4-6": 200_000,
    "anthropic.claude-sonnet-4-5": 200_000,
    "anthropic.claude-haiku-4-5": 200_000,
    "anthropic.claude-opus-4": 200_000,
    "anthropic.claude-sonnet-4": 200_000,
    "anthropic.claude-3-5-sonnet": 200_000,
    "anthropic.claude-3-5-haiku": 200_000,
    "anthropic.claude-3-opus": 200_000,
    "anthropic.claude-3-sonnet": 200_000,
    "anthropic.claude-3-haiku": 200_000,
    # Amazon Nova
    "amazon.nova-pro": 300_000,
    "amazon.nova-lite": 300_000,
    "amazon.nova-micro": 128_000,
    # Meta Llama
    "meta.llama4-maverick": 128_000,
    "meta.llama4-scout": 128_000,
    "meta.llama3-3-70b-instruct": 128_000,
    # Mistral
    "mistral.mistral-large": 128_000,
    # DeepSeek
    "deepseek.v3": 128_000,
}

# Default for unknown Bedrock models.
BEDROCK_DEFAULT_CONTEXT_LENGTH = 128_000


def get_bedrock_context_length(model_id: str) -> int:
    """Look up the context window size for a Bedrock model.

    Uses substring matching so versioned IDs like
    ``anthropic.claude-sonnet-4-6-20250514-v1:0`` resolve correctly.
    """
    model_lower = (model_id or "").lower()
    best_key = ""
    best_val = BEDROCK_DEFAULT_CONTEXT_LENGTH
    for key, val in BEDROCK_CONTEXT_LENGTHS.items():
        if key in model_lower and len(key) > len(best_key):
            best_key = key
            best_val = val
    return best_val
