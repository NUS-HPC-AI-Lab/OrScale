"""
Downstream LM evaluation adapter for OrScale checkpoints.

Wraps an OrScale ``GPT`` model as an ``lm_eval.api.model.LM`` instance so it
can be scored on the EleutherAI lm-evaluation-harness benchmark suite used
in the Moonlight and Muon-post papers:

    HellaSwag, MMLU, MMLU-pro, BBH, TriviaQA, GSM8K, MATH, HumanEval, MBPP.

We intentionally keep the adapter small and only implement the two hooks
the harness needs: ``loglikelihood`` (for multiple-choice tasks) and
``generate_until`` (for free-form tasks like GSM8K or HumanEval). Both use
the ``tiktoken`` GPT-2 tokenizer that matches the training-time tokenizer.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from orscale.model.gpt import GPT, GPTConfig, PRESET_CONFIGS


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenizer helper (matches the training-time tokenizer)
# ---------------------------------------------------------------------------

def _get_tiktoken_encoder():
    try:
        import tiktoken
    except ImportError as err:
        raise ImportError(
            "tiktoken is required for downstream eval. pip install tiktoken"
        ) from err
    return tiktoken.get_encoding("gpt2")


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_gpt_from_checkpoint(path: str, device: str | torch.device = "cpu") -> GPT:
    """Rebuild a ``GPT`` from a ``Trainer.save_checkpoint`` file.

    The checkpoint stores the full training config under the ``config`` key,
    so we use it to reconstruct the model architecture, then load the state
    dict. If the config has a ``model.preset`` we use ``GPT.from_preset``;
    otherwise we feed the ``model`` subdict to ``GPTConfig`` directly.
    """
    state = torch.load(path, map_location=device, weights_only=False)
    model_cfg = state.get("config", {}).get("model", {})

    preset = model_cfg.get("preset")
    if preset and preset in PRESET_CONFIGS:
        overrides = {k: v for k, v in model_cfg.items() if k != "preset"}
        model = GPT.from_preset(preset, **overrides)
    else:
        valid_fields = GPTConfig.__dataclass_fields__.keys()
        filtered = {k: v for k, v in model_cfg.items() if k in valid_fields}
        model = GPT(GPTConfig(**filtered))

    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# lm-eval adapter
# ---------------------------------------------------------------------------

def _build_lm_adapter_class():
    """Build the adapter class lazily so ``lm_eval`` is only imported when used."""
    try:
        from lm_eval.api.model import LM
        from lm_eval.api.registry import register_model
    except ImportError as err:
        raise ImportError(
            "lm-eval is required for downstream evaluation. "
            "pip install 'lm-eval>=0.4.0'"
        ) from err

    class OrScaleLMAdapter(LM):
        """Adapter exposing an OrScale ``GPT`` to lm-evaluation-harness."""

        def __init__(
            self,
            model: GPT,
            batch_size: int = 8,
            device: str | torch.device = "cpu",
            max_length: int | None = None,
            dtype: torch.dtype = torch.float32,
        ):
            super().__init__()
            self._model = model.to(device)
            self._model.eval()
            self._device = torch.device(device)
            self._enc = _get_tiktoken_encoder()
            self._batch_size = int(batch_size)
            self._max_length = int(max_length or model.config.max_seq_len)
            self._dtype = dtype
            self._eos_token_id = self._enc.eot_token

        # -- Required properties --

        @property
        def eot_token_id(self) -> int:
            return self._eos_token_id

        @property
        def max_length(self) -> int:
            return self._max_length

        @property
        def max_gen_toks(self) -> int:
            return min(256, self._max_length)

        @property
        def batch_size(self) -> int:
            return self._batch_size

        @property
        def device(self):
            return self._device

        def tok_encode(self, string: str) -> list[int]:
            return self._enc.encode_ordinary(string)

        def tok_decode(self, tokens: list[int]) -> str:
            return self._enc.decode(tokens)

        # -- Loglikelihood (multiple-choice tasks) --

        def loglikelihood(self, requests):
            """Return list of (log_prob, is_greedy) for (context, continuation) pairs."""
            results = []
            for req in requests:
                context, continuation = req.args
                ctx_ids = self.tok_encode(context)
                cont_ids = self.tok_encode(continuation)
                # Prepend BOS (use eot_token as BOS, matching tiktoken GPT-2 usage)
                if not ctx_ids:
                    ctx_ids = [self._eos_token_id]

                input_ids = torch.tensor(
                    ctx_ids + cont_ids, dtype=torch.long, device=self._device
                ).unsqueeze(0)
                # Truncate from the left to fit context
                if input_ids.size(1) > self._max_length:
                    input_ids = input_ids[:, -self._max_length :]

                cont_len = len(cont_ids)
                if cont_len == 0:
                    results.append((0.0, True))
                    continue

                with torch.no_grad():
                    logits = self._model(input_ids)["logits"][0]

                # Compare continuation tokens vs argmax at their positions.
                # Position of continuation token t (0-indexed in cont) is at
                # index (len(ctx_ids) + t - 1) of input_ids' prefix, so the
                # predicted logits at index -cont_len-1 : -1.
                cont_logits = logits[-cont_len - 1 : -1].float()  # [cont_len, V]
                cont_targets = torch.tensor(cont_ids, device=self._device)
                log_probs = F.log_softmax(cont_logits, dim=-1)
                ll = log_probs.gather(-1, cont_targets.unsqueeze(-1)).squeeze(-1).sum().item()
                greedy = bool((cont_logits.argmax(dim=-1) == cont_targets).all().item())
                results.append((ll, greedy))
            return results

        def loglikelihood_rolling(self, requests):
            """Rolling loglikelihood over a full document (used by perplexity tasks)."""
            results = []
            for req in requests:
                (text,) = req.args
                token_ids = self.tok_encode(text)
                if not token_ids:
                    results.append(0.0)
                    continue

                total_ll = 0.0
                window = self._max_length
                for start in range(0, len(token_ids), window - 1):
                    chunk = token_ids[start : start + window]
                    if len(chunk) < 2:
                        break
                    input_ids = torch.tensor(
                        chunk, dtype=torch.long, device=self._device
                    ).unsqueeze(0)
                    with torch.no_grad():
                        logits = self._model(input_ids)["logits"][0, :-1].float()
                    targets = torch.tensor(chunk[1:], device=self._device)
                    log_probs = F.log_softmax(logits, dim=-1)
                    total_ll += log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).sum().item()
                results.append(total_ll)
            return results

        # -- Free-form generation (GSM8K, HumanEval, ...) --

        def generate_until(self, requests):
            results = []
            for req in requests:
                context, gen_kwargs = req.args
                until = gen_kwargs.get("until", []) if isinstance(gen_kwargs, dict) else []
                max_new = int(
                    gen_kwargs.get("max_gen_toks", self.max_gen_toks)
                    if isinstance(gen_kwargs, dict) else self.max_gen_toks
                )

                input_ids = torch.tensor(
                    self.tok_encode(context), dtype=torch.long, device=self._device
                ).unsqueeze(0)
                if input_ids.size(1) > self._max_length - max_new:
                    input_ids = input_ids[:, -(self._max_length - max_new) :]

                generated = input_ids
                with torch.no_grad():
                    for _ in range(max_new):
                        if generated.size(1) >= self._max_length:
                            break
                        logits = self._model(generated)["logits"][0, -1].float()
                        next_id = int(logits.argmax().item())
                        generated = torch.cat(
                            [generated, torch.tensor([[next_id]], device=self._device)], dim=1,
                        )
                        # Check stop strings
                        decoded = self.tok_decode(generated[0, input_ids.size(1):].tolist())
                        if any(stop and stop in decoded for stop in until):
                            break

                output = self.tok_decode(generated[0, input_ids.size(1):].tolist())
                for stop in until:
                    if stop and stop in output:
                        output = output.split(stop)[0]
                        break
                results.append(output)
            return results

    # Attempt to register, but don't fail if already registered
    try:
        register_model("orscale_gpt")(OrScaleLMAdapter)
    except Exception:  # noqa: BLE001
        pass
    return OrScaleLMAdapter


# Public factory (lazy import of lm_eval)

def OrScaleLMAdapter(*args, **kwargs):  # noqa: N802 (keep public name)
    """Factory returning a bound ``lm_eval.api.model.LM`` adapter instance."""
    cls = _build_lm_adapter_class()
    return cls(*args, **kwargs)


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def run_downstream(
    model_or_ckpt: GPT | str,
    tasks: Iterable[str] = ("hellaswag",),
    batch_size: int = 8,
    limit: int | None = None,
    device: str | torch.device = "cpu",
    num_fewshot: int = 0,
) -> dict[str, Any]:
    """Evaluate a GPT model on the given lm-eval tasks.

    Args:
        model_or_ckpt: Either an OrScale ``GPT`` instance or a path to a
            ``Trainer.save_checkpoint`` file.
        tasks: Iterable of lm-eval task names (e.g. ``"hellaswag"``,
            ``"mmlu"``, ``"gsm8k"``, ``"humaneval"``, ``"mbpp"``).
        batch_size: Eval batch size.
        limit: Optional cap on the number of test examples per task.
        device: Torch device for the forward passes.
        num_fewshot: Number of few-shot examples per task (default 0).

    Returns:
        The ``lm_eval.evaluator.simple_evaluate`` results dict.
    """
    try:
        from lm_eval import simple_evaluate
    except ImportError as err:
        raise ImportError(
            "lm-eval is required. pip install 'lm-eval>=0.4.0'"
        ) from err

    if isinstance(model_or_ckpt, str):
        model = load_gpt_from_checkpoint(model_or_ckpt, device=device)
    else:
        model = model_or_ckpt
    adapter = OrScaleLMAdapter(model, batch_size=batch_size, device=device)

    task_list = list(tasks)
    LOGGER.info("Running lm-eval on tasks=%s, limit=%s, fewshot=%d",
                task_list, limit, num_fewshot)
    results = simple_evaluate(
        model=adapter,
        tasks=task_list,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        limit=limit,
        device=str(device),
    )
    return results
