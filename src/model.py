"""Model loading utilities – supports QLoRA quantisation & optional LoRA adapters."""
from __future__ import annotations

from typing import List, Tuple, Union

import torch
from omegaconf import DictConfig
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig  # only available with BnB installed
except ImportError:  # pragma: no cover – CPU fallback, will not be used on A100
    BitsAndBytesConfig = None  # type: ignore


# -----------------------------------------------------------------------------
#                         QUANTISATION CONFIG
# -----------------------------------------------------------------------------

def _quant_cfg(cfg: DictConfig):
    if BitsAndBytesConfig is None or cfg.model.quantization.scheme.lower() != "qlora":  # type: ignore[operator]
        return None
    return BitsAndBytesConfig(
        load_in_4bit=int(cfg.model.quantization.bits) == 4,
        load_in_8bit=int(cfg.model.quantization.bits) == 8,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=bool(cfg.model.quantization.double_quant),
    )


# -----------------------------------------------------------------------------
#                       BACKBONE + ADAPTER LOADING
# -----------------------------------------------------------------------------

def load_lm_with_adapters(cfg: DictConfig):
    model_name = cfg.model.name
    quant_cfg = _quant_cfg(cfg)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=cfg.model.device_map,
        quantization_config=quant_cfg,
        cache_dir=".cache/",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=".cache/", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --------------------------- LoRA ------------------------------------- #
    if cfg.model.lora.enabled:
        l_cfg = LoraConfig(
            r=int(cfg.model.lora.rank),
            lora_alpha=int(cfg.model.lora.alpha),
            lora_dropout=float(cfg.model.lora.dropout),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=list(cfg.model.lora.target_modules),
        )
        model = get_peft_model(model, l_cfg)

    if cfg.model.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model, tokenizer


# -----------------------------------------------------------------------------
#                            ZEST HYPER-NETWORK
# -----------------------------------------------------------------------------
class ZestPredictor(torch.nn.Module):
    """2-layer MLP g_φ : ℝ⁸²⁰⁵ → ℝ⁴ predicting (η₀, λ₀, α, β)."""

    def __init__(self, d_emb: int = 8205, hidden_layers: List[int] | None = None):
        super().__init__()
        hidden_layers = hidden_layers or [1024]
        layers = []
        inp = d_emb
        for h in hidden_layers:
            layers.extend([torch.nn.Linear(inp, h), torch.nn.ReLU()])
            inp = h
        layers.append(torch.nn.Linear(inp, 4))
        self.net = torch.nn.Sequential(*layers)
        self.softplus = torch.nn.Softplus()

    def forward(self, z: torch.Tensor):  # type: ignore[override]
        if z.dim() == 1:
            z = z.unsqueeze(0)
        raw = self.net(z).squeeze(0)
        eta_log, lam_log, a_raw, b_raw = raw
        eta0 = torch.exp(eta_log)
        lam0 = torch.exp(lam_log)
        alpha = self.softplus(a_raw) + 1.0
        beta = self.softplus(b_raw) + 1.0
        return eta0, lam0, alpha, beta