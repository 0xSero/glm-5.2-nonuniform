"""Overthinking-marker logit penalty (arXiv:2606.00206) as a vLLM v1 logits processor.

Quantized/pruned reasoning models reach a correct answer mid-CoT, then hedge
("wait", "alternatively", ...) without improving accuracy. The paper's fix:
subtract a fixed lambda from the logits of hesitation-marker tokens.
Per-request `logit_bias` is rejected by vLLM under speculative decoding, but a
launch-time logits processor applies to the target model's sampling just fine.

Enabled via:  --logits-processors overthink_logits_processor.OverthinkPenalty
Tune with:    OVERTHINK_PENALTY  (float lambda, default 2.0; set 0 to disable)
              OVERTHINK_MARKERS  (comma-separated words; leading-space variants
                                  are handled automatically)
"""

import os

import torch

from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor

DEFAULT_MARKERS = (
    "wait,Wait,WAIT,hmm,Hmm,alternatively,Alternatively,reconsider,Reconsider,"
    "recheck,double-check,second-guess,rethink"
)


class OverthinkPenalty(LogitsProcessor):
    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool):
        self.penalty = float(os.environ.get("OVERTHINK_PENALTY", "2.0"))
        ids: set[int] = set()
        if self.penalty > 0:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(
                vllm_config.model_config.tokenizer, trust_remote_code=True
            )
            markers = os.environ.get("OVERTHINK_MARKERS", DEFAULT_MARKERS)
            for word in markers.split(","):
                word = word.strip()
                if not word:
                    continue
                for variant in (word, " " + word):
                    toks = tok.encode(variant, add_special_tokens=False)
                    if toks:
                        ids.add(toks[0])
        self.ids = (
            torch.tensor(sorted(ids), dtype=torch.long, device=device) if ids else None
        )

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        pass

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.ids is not None:
            logits[:, self.ids] -= self.penalty
        return logits
