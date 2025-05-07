# llm_models.py

import re
import ast
import torch
from typing import Optional, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)


class AbstractLLM:
    """
    Abstract base class for all LLM evaluators.
    Any subclass must implement evaluate_equation().
    """
    def evaluate_equation(self, equation_str: str, context: Optional[str] = None) -> float:
        raise NotImplementedError


def parse_llm_response(text_out: str) -> float:
    """
    Parses a model output and extracts a score from a vector of the form:
    [dim_corr, simp, sim, "feedback"]

    Returns:
        float: average score of the first three elements (0.0–1.0 scale).
    Defaults to 0.5 on failure.
    """
    pattern = r"\[.*?\]"
    matches = re.findall(pattern, text_out, flags=re.DOTALL)
    if not matches:
        return 0.5
    try:
        parsed = ast.literal_eval(matches[-1])
        if len(parsed) < 3:
            return 0.5
        dim_corr = float(parsed[0])
        simp = float(parsed[1])
        sim = float(parsed[2])
        return sum(map(lambda x: max(0, min(1, x)), [dim_corr, simp, sim])) / 3.0
    except Exception:
        return 0.5


class LlamaLLM(AbstractLLM):
    """
    Llama-based LLM evaluator (e.g. meta-llama/Llama-2-7b-chat-hf).
    Utilizes 4-bit quantized weights for efficiency.
    """
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf", device="cuda", cache_dir=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=False, trust_remote_code=True, cache_dir=cache_dir
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        print(f"Loading Llama model from {model_name} with 4-bit quantization...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory={0: "35GiB", "cpu": "80GiB"},
            torch_dtype=torch.float16,
            trust_remote_code=True,
            cache_dir=cache_dir
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=96,
            temperature=0.1,
            top_p=0.7,
            top_k=40,
            do_sample=True
        )

    def evaluate_equation(self, equation_str: str, context: Optional[str] = None) -> float:
        prompt = f"""Evaluate the following mathematical equation for dimensional correctness, 
simplicity, and scientific plausibility. Return a list like: [dim_corr, simp, sim, "feedback"].

Equation:
{equation_str}
"""
        out = self.pipe(prompt, num_return_sequences=1, max_new_tokens=128)
        return parse_llm_response(out[0]["generated_text"])


class FalconLLM(AbstractLLM):
    """
    Falcon-based LLM evaluator (e.g. tiiuae/falcon-7b-instruct).
    4-bit quantized version for efficient inference.
    """
    def __init__(self, model_name="tiiuae/falcon-7b-instruct", device="cuda", cache_dir=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, cache_dir=cache_dir
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        print(f"Loading Falcon model from {model_name} with 4-bit quantization...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory={0: "35GiB", "cpu": "80GiB"},
            torch_dtype=torch.float16,
            trust_remote_code=True,
            cache_dir=cache_dir
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=96,
            temperature=0.1,
            top_p=0.7,
            top_k=40,
            do_sample=True
        )

    def evaluate_equation(self, equation_str: str, context: Optional[str] = None) -> float:
        prompt = f"""Evaluate the following mathematical equation for dimensional correctness, 
simplicity, and scientific plausibility. Return a list like: [dim_corr, simp, sim, "feedback"].

Equation:
{equation_str}
"""
        out = self.pipe(prompt, num_return_sequences=1, max_new_tokens=128)
        return parse_llm_response(out[0]["generated_text"])


class MistralLLM(AbstractLLM):
    """
    Mistral-based LLM (e.g. mistralai/Mistral-7B-Instruct-v0.3) with caching.
    Avoids repeated evaluations for identical input pairs.
    """
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.3", device="cuda", cache_dir=None):
        self.cache = {}
        self.device = device

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=False, cache_dir=cache_dir, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print(f"[MistralLLM] Loading {model_name} with 4-bit quantization...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_cfg,
            device_map="auto",
            max_memory={0: "35GiB", "cpu": "80GiB"},
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
            trust_remote_code=True
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=96,
            temperature=0.1,
            top_p=0.7,
            top_k=40,
            do_sample=True
        )

    def evaluate_equation(self, equation_str: str, context: Optional[str] = None) -> float:
        key = (equation_str, context or "")
        if key in self.cache:
            return self.cache[key]

        prompt = f"""You are a scientific assistant.

Context:
{context or ""}

Task:
Evaluate the symbolic equation below for:
(1) dimensional correctness, (2) simplicity, and (3) scientific plausibility.
Respond ONLY with a list: [dim_corr, simp, sim, "short feedback"].

Equation:
{equation_str}
"""
        out = self.pipe(prompt, num_return_sequences=1, max_new_tokens=128)
        score = parse_llm_response(out[0]["generated_text"])
        self.cache[key] = score
        return score


class HuggingFaceLLM(AbstractLLM):
    """
    Generic wrapper for any HuggingFace LLM loaded with 4-bit quantization.
    """
    def __init__(self, model_name="your-hf-model", device="cuda", cache_dir=None):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=False, trust_remote_code=True, cache_dir=cache_dir
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print(f"Loading HF model {model_name} with 4-bit quantization...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory={0: "35GiB", "cpu": "80GiB"},
            torch_dtype=torch.float16,
            trust_remote_code=True,
            cache_dir=cache_dir
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=96,
            temperature=0.1,
            top_p=0.7,
            top_k=40,
            do_sample=True
        )

    def evaluate_equation(self, equation_str: str, context: Optional[str] = None) -> float:
        prompt = f"""Evaluate the following mathematical equation for dimensional correctness, 
simplicity, and scientific plausibility. Return a list like: [dim_corr, simp, sim, "feedback"].

Equation:
{equation_str}
"""
        out = self.pipe(prompt, num_return_sequences=1, max_new_tokens=128)
        return parse_llm_response(out[0]["generated_text"])


class DummyLLM(AbstractLLM):
    """
    Minimal evaluator that always returns a fixed score.
    Useful for debugging or unit tests.
    """
    def evaluate_equation(self, equation_str: str, context: Optional[str] = None) -> float:
        return 0.5
