"""Hugging Face Transformers agent for symbolic macro-action evaluation."""

from __future__ import annotations

import json
import time
from typing import Any, Mapping, Optional, Sequence

from ..models import AgentDecision


def extract_first_json_object(text: str) -> Optional[Mapping[str, Any]]:
    """Return the first decoded JSON object found in model output."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            nested = value.get("action")
            return nested if isinstance(nested, Mapping) else value
    return None


class TransformersJSONAgent:
    """Load one instruction model and emit one benchmark macro action per call."""

    def __init__(
        self,
        model_id: str,
        *,
        max_new_tokens: int = 96,
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        token: Optional[str] = None,
        trust_remote_code: bool = True,
    ):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        self.model_id = model_id
        self.name = model_id.replace("/", "--")
        self.max_new_tokens = max_new_tokens
        self._torch = torch
        dtype = getattr(torch, torch_dtype)
        common = {
            "device_map": device_map,
            "torch_dtype": dtype,
            "token": token,
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        config = AutoConfig.from_pretrained(model_id, token=token, trust_remote_code=trust_remote_code)
        model_type = str(getattr(config, "model_type", ""))
        self.is_multimodal = model_type in {"gemma4", "qwen3_5", "mistral3"}
        if self.is_multimodal:
            try:
                from transformers import AutoModelForMultimodalLM
                self.model = AutoModelForMultimodalLM.from_pretrained(model_id, **common)
            except (ImportError, ValueError):
                self.model = AutoModelForCausalLM.from_pretrained(model_id, **common)
            self.processor = AutoProcessor.from_pretrained(model_id, token=token, trust_remote_code=trust_remote_code)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **common)
            self.processor = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=trust_remote_code)
        self.model.eval()

    @staticmethod
    def _prompt(observation: Mapping[str, Any], legal_actions: Sequence[str], context: Mapping[str, Any]) -> str:
        payload = {
            "task_id": context.get("task_id"),
            "step": context.get("step"),
            "objective": observation.get("objective"),
            "state": observation.get("player"),
            "legal_action_types": list(legal_actions),
            "action_candidates": observation.get("action_candidates", []),
        }
        return (
            "You control a civilization in a real-time strategy benchmark. "
            "Choose the single candidate action that best satisfies the objective from the current state. "
            "Return exactly one JSON object copied from action_candidates. Do not add prose or markdown.\n"
            + json.dumps(payload, sort_keys=True, separators=(",", ":"))
        )

    def _prepare_inputs(self, prompt: str):
        text_messages = [
            {"role": "system", "content": "Select one legal RTS macro action and output strict JSON."},
            {"role": "user", "content": prompt},
        ]
        multimodal_messages = [
            {"role": "system", "content": [{"type": "text", "text": "Select one legal RTS macro action and output strict JSON."}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        messages = multimodal_messages if self.is_multimodal else text_messages
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except (TypeError, ValueError):
            inputs = self.processor.apply_chat_template(
                text_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        return {key: value.to(self.model.device) for key, value in inputs.items()}

    def act(self, observation, legal_actions, context) -> AgentDecision:
        prompt = self._prompt(observation, legal_actions, context)
        inputs = self._prepare_inputs(prompt)
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        started = time.perf_counter()
        with self._torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=getattr(self.processor, "pad_token_id", None),
            )
        latency_ms = (time.perf_counter() - started) * 1000
        generated = output[0, prompt_tokens:]
        completion_tokens = int(generated.shape[-1])
        text = self.processor.decode(generated, skip_special_tokens=True)
        action = extract_first_json_object(text) or {"type": "invalid_model_output"}
        return AgentDecision(
            action=dict(action),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            rationale="Parsed a JSON action from the final model response." if action.get("type") != "invalid_model_output" else "The final model response did not contain a JSON action.",
        )

    def close(self) -> None:
        del self.model
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

