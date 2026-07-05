# glm-5.2-nonuniform — serve a non-uniformly expert-pruned GLM-5.2 with a 12-line vLLM fork

Serves **[0xSero/GLM-5.2-REAP-NU176-526B](https://huggingface.co/0xSero/GLM-5.2-REAP-NU176-526B)** —
GLM-5.2 with a **different number of routed experts per layer** (108–216, avg 176 of 256),
allocated by REAP saliency over 16.08B calibration tokens — on 4× RTX PRO 6000 Blackwell
(SM120, 4×96GB). 262k context @ concurrency 2, FP8 KV, MTP speculative decoding,
~77 tok/s single stream.

## Quick start

```bash
git clone https://github.com/0xSero/glm-5.2-nonuniform && cd glm-5.2-nonuniform
./run.sh /mnt/llm_models/GLM-5.2-REAP-NU176   # downloads 337GB, builds fork image, serves :8000
```

OpenAI-compatible API on `:8000` (`served-model-name: glm-5.2-nu176`).

Prebuilt fork image: `ghcr.io/0xsero/vllm-b12x-nonuniform:20260705` (compose uses it by
default; `docker compose build` rebuilds locally from the patch). Also mirrored as a
docker tar in the HF repo under `docker/` (`zstd -d | docker load`).

## Why a fork

Every MoE serving stack assumes one expert count for all layers. Non-uniform pruning is
strictly better in principle (prune where the router doesn't care, keep capacity where it
does), and it turns out vLLM only needs **12 lines** to support it:

- checkpoint config carries `num_routed_experts_per_layer: list[int]` (len 79 incl. MTP),
  with scalar `n_routed_experts = max(list)` so every uniform-assuming code path
  (expert-params mapping, MTP, quant configs) stays correct;
- `DeepseekV2MoE.__init__` resolves its own layer's count from the list (layer index
  parsed from the module prefix — covers the MTP drafter for free);
- each layer's experts are renumbered `0..k-1` in the checkpoint, `gate.weight` /
  `e_score_correction_bias` rows sliced to match.

See [`deepseek_v2_nonuniform.patch`](deepseek_v2_nonuniform.patch) (the diff) /
[`deepseek_v2_nu.py`](deepseek_v2_nu.py) (full file, applied by the Dockerfile on top of
the `voipmonitor/vllm` eldritch b12x image, which also contains the GLM-5.2 SM120 fixes).
Same pattern as vLLM's NemotronH heterogeneous-MoE support. GLM-5.2 has
`n_group=1`, so any per-layer count ≥ 8 is legal — no group-divisibility constraint.

## How the cut was made

1. **Observe**: per-(layer, expert) REAP saliency (router gate-weight × expert output
   norm), merged across 5 calibration runs / 16.08B tokens
   (`0xSero/glm-5.2-reap-observations`, `0xSero/glm52-reap-traces`).
2. **Allocate**: water-filling on layer-normalized saliency curves at a 75×176 budget →
   per-layer keep counts ([`keep_counts_avg176.json`](keep_counts_avg176.json)). GLM-5.2's
   router is heavily load-balanced, so equal-budget reallocation alone is ~neutral; the
   win comes from spending the budget where uniform cuts bite (layers 7–17) — mean dropped
   saliency 17.6% vs 22.3% for the prior uniform cut.
3. **Cut**: [`reap_apply_plan_nu.py`](reap_apply_plan_nu.py) — whole-expert copy from
   `nvidia/GLM-5.2-NVFP4` with per-layer renumbering ([`plan_nu176.json`](plan_nu176.json)).
4. **MTP fix**: NVIDIA ships the MTP layer (78) with **BF16** experts; b12x-class MoE
   kernels reject unquantized MoE. [`transplant_mtp.py`](transplant_mtp.py) swaps in NVFP4
   MTP experts from the uniform-168 cut (kept 0..167 identity-order, so tensors carry over).

## Hard-won launch facts (read before changing anything)

- `VLLM_USE_B12X_SPARSE_INDEXER=1` is **required** — without it the MTP drafter
  auto-selects `FlashInferMLASparseSM120`, which cannot return decode LSE → DCP assert.
- config.json must NOT contain `kv_cache_scheme` / `hf_quant_config.json` (ModelOpt
  export artifacts) — they derail backend selection. This repo's checkpoint is already clean.
- `NCCL_P2P_DISABLE=1` on Blackwell PCIe boxes (allreduce deadlock).
- The `index_topk_pattern` hf-override (79-char F/S map) is required for long-context
  coherence — vLLM ignores `config.indexer_types`.
- FP8 KV (`fp8_ds_mla`) is required on SM120; BF16 KV produces garbage.
- First boot JIT-compiles B12X kernels for ~30 distinct expert-count shapes: 30–40 min.
  The JIT cache persists in the `nu176-jit` volume; warm boots take ~5 min.

## Anti-overthinking (arXiv:2606.00206)

Quantized/pruned reasoning models reach the right answer mid-CoT, then hedge
("wait", "alternatively", …) without improving accuracy. The paper's fix is a logit
penalty on hesitation markers; vLLM rejects `logit_bias` under speculative decoding, so
apply it as a system directive (or run without MTP and use `logit_bias`). See
`overthinking_penalty_proxy.py` in this repo for a transparent sidecar implementation.

## Files

| file | purpose |
|---|---|
| `Dockerfile` + `docker-compose.yml` + `run.sh` | one-command serving |
| `deepseek_v2_nonuniform.patch` / `deepseek_v2_nu.py` | the vLLM fork |
| `reap_apply_plan_nu.py` | apply any non-uniform plan to any GLM/DeepSeek-style MoE checkpoint |
| `plan_nu176.json` / `keep_counts_avg176.json` | this model's exact prune plan |
| `transplant_mtp.py` | MTP NVFP4 expert transplant |
| `overthinking_penalty_proxy.py` | arXiv:2606.00206 serving sidecar |
