"""Yukon Systems unified consensus engine package."""

from .agent import ConsensusEngine, build_engine, build_model_registry, run_cli
from .backends import (
    LLMBackend,
    LlamaCppBackend,
    OllamaBackend,
    OpenAIBackend,
    AnthropicBackend,
    LlamaAPIBackend,
    ReasoningBackend,
)
from .debate import DebateManager
from .templates import (
    ReasoningTemplate,
    detect_template,
    get_template_by_name,
    list_available_templates,
    get_template_details,
)

__all__ = [
    "ConsensusEngine",
    "build_engine",
    "build_model_registry",
    "run_cli",
    "LLMBackend",
    "LlamaCppBackend",
    "OllamaBackend",
    "OpenAIBackend",
    "AnthropicBackend",
    "LlamaAPIBackend",
    "ReasoningBackend",
    "DebateManager",
    "ReasoningTemplate",
    "detect_template",
    "get_template_by_name",
    "list_available_templates",
    "get_template_details",
]

