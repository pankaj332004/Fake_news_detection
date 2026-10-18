"""
LLM Loader: Manages loading of open-weight language models for verification.
Supports:
1. Local HuggingFace Transformers pipeline (Flan-T5-small/base by default)
2. HuggingFace Inference API (free tier, no GPU required)
3. Mock mode for testing without model weights

All models run on CPU and are free to use with no API keys required for local mode.
"""

import json
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Default model choices (all free, open-weight, CPU-friendly)
DEFAULT_LOCAL_MODEL = "google/flan-t5-small"
RECOMMENDED_LOCAL_MODEL = "google/flan-t5-base"
HF_API_URL = "https://api-inference.huggingface.co/models/"

# Generation parameters for verification prompts
DEFAULT_GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.1,        # Low temperature for factual consistency
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.2,
}


class LLMBackend:
    """Enumeration of supported LLM backend types."""
    LOCAL = "local"           # HuggingFace Transformers pipeline (CPU)
    HF_API = "hf_api"        # HuggingFace Inference API (free tier)
    MOCK = "mock"             # Deterministic mock for testing


class ModelInfo:
    """Metadata about a loaded model."""

    def __init__(
        self,
        model_name: str,
        backend: str,
        loaded: bool = False,
        load_time_seconds: float = 0.0,
        error: Optional[str] = None,
    ):
        self.model_name = model_name
        self.backend = backend
        self.loaded = loaded
        self.load_time_seconds = load_time_seconds
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "backend": self.backend,
            "loaded": self.loaded,
            "load_time_seconds": round(self.load_time_seconds, 2),
            "error": self.error,
        }


class LLMLoader:
    """
    Loads and manages an open-weight LLM for claim verification.
    Designed for CPU-only environments using small instruct-tuned models.
    """

    _pipeline = None
    _current_model: Optional[str] = None

    def __init__(
        self,
        model_name: str = DEFAULT_LOCAL_MODEL,
        backend: str = LLMBackend.LOCAL,
        hf_api_token: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            model_name: HuggingFace model ID (e.g. 'google/flan-t5-small').
            backend: One of 'local', 'hf_api', or 'mock'.
            hf_api_token: Optional HuggingFace API token for Inference API.
            generation_config: Override default generation parameters.
        """
        self.model_name = model_name
        self.backend = backend
        self.hf_api_token = hf_api_token or os.getenv("HUGGINGFACE_API_TOKEN", "")
        self.generation_config = {**DEFAULT_GENERATION_CONFIG, **(generation_config or {})}
        self.model_info: Optional[ModelInfo] = None

    def load(self) -> ModelInfo:
        """
        Loads the model based on the selected backend.
        Returns ModelInfo with loading status.
        """
        start = time.perf_counter()

        try:
            if self.backend == LLMBackend.MOCK:
                self.model_info = ModelInfo(
                    model_name="mock-model",
                    backend=LLMBackend.MOCK,
                    loaded=True,
                    load_time_seconds=0.0,
                )
                return self.model_info

            elif self.backend == LLMBackend.LOCAL:
                return self._load_local(start)

            elif self.backend == LLMBackend.HF_API:
                return self._load_hf_api(start)

            else:
                raise ValueError(f"Unknown backend: {self.backend}")

        except Exception as e:
            elapsed = time.perf_counter() - start
            self.model_info = ModelInfo(
                model_name=self.model_name,
                backend=self.backend,
                loaded=False,
                load_time_seconds=elapsed,
                error=str(e),
            )
            logger.error(f"Failed to load model '{self.model_name}': {e}")
            return self.model_info

    def _load_local(self, start: float) -> ModelInfo:
        """Loads a local HuggingFace Transformers text2text-generation pipeline."""
        # Skip reloading if same model already loaded
        if LLMLoader._pipeline is not None and LLMLoader._current_model == self.model_name:
            logger.info(f"Model '{self.model_name}' already loaded, reusing pipeline.")
            elapsed = time.perf_counter() - start
            self.model_info = ModelInfo(
                model_name=self.model_name,
                backend=LLMBackend.LOCAL,
                loaded=True,
                load_time_seconds=elapsed,
            )
            return self.model_info

        logger.info(f"Loading local model '{self.model_name}' on CPU...")
        from transformers import pipeline as hf_pipeline

        # Use text2text-generation for seq2seq models (Flan-T5)
        # Use text-generation for causal models (TinyLlama, Phi)
        if "t5" in self.model_name.lower() or "flan" in self.model_name.lower():
            task = "text2text-generation"
        else:
            task = "text-generation"

        LLMLoader._pipeline = hf_pipeline(
            task,
            model=self.model_name,
            device=-1,  # CPU
        )
        LLMLoader._current_model = self.model_name

        elapsed = time.perf_counter() - start
        logger.info(f"Model '{self.model_name}' loaded in {elapsed:.1f}s")

        self.model_info = ModelInfo(
            model_name=self.model_name,
            backend=LLMBackend.LOCAL,
            loaded=True,
            load_time_seconds=elapsed,
        )
        return self.model_info

    def _load_hf_api(self, start: float) -> ModelInfo:
        """Validates HuggingFace Inference API connectivity."""
        if not self.hf_api_token:
            logger.warning(
                "No HUGGINGFACE_API_TOKEN set. The free Inference API may have rate limits. "
                "Set HUGGINGFACE_API_TOKEN in .env for better reliability."
            )

        elapsed = time.perf_counter() - start
        self.model_info = ModelInfo(
            model_name=self.model_name,
            backend=LLMBackend.HF_API,
            loaded=True,
            load_time_seconds=elapsed,
        )
        return self.model_info

    def generate(self, prompt: str) -> str:
        """
        Generates text from the loaded model given a prompt.

        Args:
            prompt: Input text prompt.

        Returns:
            Generated text response string.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self.model_info is None or not self.model_info.loaded:
            raise RuntimeError(
                "Model not loaded. Call load() before generate()."
            )

        if self.backend == LLMBackend.MOCK:
            return self._mock_generate(prompt)
        elif self.backend == LLMBackend.LOCAL:
            return self._local_generate(prompt)
        elif self.backend == LLMBackend.HF_API:
            return self._hf_api_generate(prompt)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _local_generate(self, prompt: str) -> str:
        """Generates using local Transformers pipeline."""
        if LLMLoader._pipeline is None:
            raise RuntimeError("Local pipeline not initialized.")

        gen_kwargs = {
            "max_new_tokens": self.generation_config.get("max_new_tokens", 512),
            "temperature": self.generation_config.get("temperature", 0.1),
            "top_p": self.generation_config.get("top_p", 0.9),
            "do_sample": self.generation_config.get("do_sample", True),
            "repetition_penalty": self.generation_config.get("repetition_penalty", 1.2),
        }

        result = LLMLoader._pipeline(prompt, **gen_kwargs)

        if isinstance(result, list) and len(result) > 0:
            output = result[0]
            if "generated_text" in output:
                text = output["generated_text"]
                # For causal models, strip the prompt prefix
                if text.startswith(prompt):
                    text = text[len(prompt):].strip()
                return text
        return ""

    def _hf_api_generate(self, prompt: str) -> str:
        """Generates using HuggingFace Inference API (free tier)."""
        import requests

        url = f"{HF_API_URL}{self.model_name}"
        headers = {}
        if self.hf_api_token:
            headers["Authorization"] = f"Bearer {self.hf_api_token}"

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.generation_config.get("max_new_tokens", 512),
                "temperature": self.generation_config.get("temperature", 0.1),
                "top_p": self.generation_config.get("top_p", 0.9),
                "repetition_penalty": self.generation_config.get("repetition_penalty", 1.2),
            },
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            result = response.json()

            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "")
            elif isinstance(result, dict):
                return result.get("generated_text", "")
            return str(result)

        except requests.exceptions.RequestException as e:
            logger.error(f"HF Inference API call failed: {e}")
            raise RuntimeError(f"HuggingFace API request failed: {e}") from e

    def _mock_generate(self, prompt: str) -> str:
        """
        Returns a deterministic mock response for testing.
        Checks if the prompt mentions contradicting evidence to vary the output.
        """
        prompt_lower = prompt.lower()

        has_evidence = "no evidence passages" not in prompt_lower and "no relevant evidence" not in prompt_lower
        has_contradiction = "contradicting" in prompt_lower or "contradicts" in prompt_lower

        if not has_evidence:
            return json.dumps({
                "verdict": "UNVERIFIABLE",
                "confidence": 0.3,
                "reasoning": "No relevant evidence passages were found in the knowledge base. "
                             "Without evidence, the claim cannot be verified or refuted.",
                "evidence_citations": [],
                "contradictions_found": False,
                "limitations": "No evidence available in the reference collection. "
                               "This does NOT mean the claim is false.",
            })

        if has_contradiction:
            return json.dumps({
                "verdict": "FALSE",
                "confidence": 0.85,
                "reasoning": "Retrieved evidence directly contradicts the core assertions "
                             "of this claim. Multiple authoritative sources refute the "
                             "key premise.",
                "evidence_citations": ["DOC-MOCK-01#p0"],
                "contradictions_found": True,
                "limitations": "Assessment based on retrieved evidence only. Additional "
                               "context may exist outside the current knowledge base.",
            })

        return json.dumps({
            "verdict": "TRUE",
            "confidence": 0.75,
            "reasoning": "Retrieved evidence provides supporting context consistent "
                         "with the claim. Multiple passages corroborate the core "
                         "assertions.",
            "evidence_citations": ["DOC-MOCK-01#p0"],
            "contradictions_found": False,
            "limitations": "Verification limited to the available evidence collection. "
                           "Broader fact-checking may yield additional insights.",
        })
