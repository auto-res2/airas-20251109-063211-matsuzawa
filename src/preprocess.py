"""Dataset loading & tokenisation pipeline (GSM8K-style QA)."""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling


class Preprocessor:
    """Builds PyTorch *DataLoader* for training, the *raw* validation dataset and
    the list of texts required for ZEST fingerprint embedding.
    """

    def __init__(self, cfg: DictConfig, tokenizer: AutoTokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------
    def _tokenise(self, ex: Dict[str, List[str]], *, include_answer: bool) -> Dict[str, List[List[int]]]:
        prompt_tmpl: str = self.cfg.dataset.prompt_template
        outputs = {"input_ids": [], "attention_mask": [], "labels": []}
        for q, a in zip(ex["question"], ex["answer"]):
            txt = prompt_tmpl.format(question=q)
            if include_answer:
                txt = f"{txt} {a}"
            enc = self.tokenizer(
                txt,
                truncation=True,
                max_length=self.cfg.dataset.max_seq_length,
                padding="max_length",
            )
            outputs["input_ids"].append(enc["input_ids"])
            outputs["attention_mask"].append(enc["attention_mask"])
            outputs["labels"].append(enc["input_ids"].copy())
        return outputs

    # ------------------------------------------------------------------
    def _load_raw(self) -> Tuple[Dataset, Dataset]:
        ds_train = load_dataset(
            self.cfg.dataset.hf_id,
            "main",
            split=self.cfg.dataset.train_split,
            cache_dir=".cache/",
        )
        ds_val = load_dataset(
            self.cfg.dataset.hf_id,
            "main",
            split=self.cfg.dataset.val_split,
            cache_dir=".cache/",
        )
        return ds_train, ds_val

    # ------------------------------------------------------------------
    def build(self) -> Tuple[DataLoader, Dataset, List[str]]:
        ds_train_raw, ds_val_raw = self._load_raw()
        # Tokenise training set (include answer so LM learns full sequence)
        ds_train_tok = ds_train_raw.map(
            lambda ex: self._tokenise(ex, include_answer=True),
            batched=True,
            remove_columns=ds_train_raw.column_names,
            desc="Tokenising train set",
        )
        collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        dl = DataLoader(
            ds_train_tok,
            batch_size=self.cfg.training.global_batch_size,
            shuffle=True,
            collate_fn=collator,
        )

        # Fingerprint sample (ZEST) – *without answer* to avoid leakage
        k = min(self.cfg.dataset.sample_for_fingerprint, len(ds_train_raw))
        idxs = random.sample(range(len(ds_train_raw)), k)
        fp_texts = [
            self.cfg.dataset.prompt_template.format(question=ds_train_raw[i]["question"])
            for i in idxs
        ]
        return dl, ds_val_raw, fp_texts


# -----------------------------------------------------------------------------
#                     GSM8K ACCURACY (exact number match)
# -----------------------------------------------------------------------------

def _extract_number(text: str) -> Optional[str]:
    import re

    m = re.search(r"####\s*(-?\d+)", text)
    if m:
        return m.group(1).lstrip("0") or "0"
    nums = re.findall(r"[-+]?[0-9]+", text)
    return nums[-1].lstrip("0") if nums else None


def compute_gsm8k_accuracy(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    val_ds: Dataset,
    *,
    max_samples: Optional[int] = None,
    device: str | torch.device = "cuda",
) -> float:
    model.eval()
    correct = 0
    total = 0
    for sample in val_ds:
        prompt = f"Q: {sample['question']}\nA:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=32)
        gen = tokenizer.decode(ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred = _extract_number(gen)
        truth = _extract_number(sample["answer"])
        if pred is not None and truth is not None and pred == truth:
            correct += 1
        total += 1
        if max_samples and total >= max_samples:
            break
    return correct / total if total else 0.0


# -----------------------------------------------------------------------------
#                         EMBEDDING FOR ZEST FINGERPRINT
# -----------------------------------------------------------------------------

def _nll_stats(nll: torch.Tensor) -> torch.Tensor:
    mean = nll.mean()
    std = nll.std(unbiased=False)
    mn = nll.min()
    mx = nll.max()
    median = nll.median()
    q25 = torch.quantile(nll, 0.25)
    q75 = torch.quantile(nll, 0.75)
    skew = ((nll - mean) ** 3).mean() / (std ** 3 + 1e-8)
    return torch.stack([mean, std, mn, mx, median, q25, q75, skew])


def embed_dataset(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    texts: List[str],
    *,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Compute the 8a0205-D fingerprint described in the paper."""
    model.eval()
    hidden_stats = []
    nlls = []
    for txt in texts:
        tok = tokenizer(txt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**tok, labels=tok["input_ids"], output_hidden_states=True, use_cache=False)
        h = out.hidden_states[-1].squeeze(0)
        hidden_stats.append(torch.stack([h.mean(0), h.var(0, unbiased=False)]))
        nlls.append(out.loss.detach())

    hs = torch.stack(hidden_stats)  # (N,2,dim)
    agg_h = torch.cat([hs[:, 0, :].mean(0), hs[:, 1, :].mean(0)])  # 4096*2

    nll_tensor = torch.stack(nlls)
    nll_summary = _nll_stats(nll_tensor)

    total_tokens = sum(len(t.split()) for t in texts)
    log_tokens = math.log(max(total_tokens, 1))
    sigma0 = nll_tensor.var(unbiased=False).item()
    model_width = math.log(getattr(model.config, "hidden_size", 4096))

    lora_r = 1
    if hasattr(model, "peft_config") and model.peft_config:
        lora_r = next(iter(model.peft_config.values())).r  # type: ignore[attr-defined]
    lora_r = math.log(lora_r)

    bits = 16
    if hasattr(model, "quantization_config") and model.quantization_config:
        bits = 4 if getattr(model.quantization_config, "load_in_4bit", False) else 8
    bits = float(bits)

    extras = torch.tensor([log_tokens, sigma0, model_width, lora_r, bits], device=device)
    fingerprint = torch.cat([agg_h.to(device), nll_summary.to(device), extras])
    assert fingerprint.numel() == 8205, f"Fingerprint dimension mismatch: {fingerprint.numel()}"
    return fingerprint