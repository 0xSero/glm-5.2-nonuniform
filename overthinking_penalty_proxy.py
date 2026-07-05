#!/usr/bin/env python3
"""
Overthinking-penalty sidecar for GLM-5.2 (arXiv:2606.00206).

Why this exists
---------------
The paper "Quantized Reasoning Models Think They Need to Think Longer, but They
Do Not" (Lotfi et al., FAIR, 2026) shows that quantized/pruned reasoning models
(REAP is a pruning perturbation, which the paper treats like quantization for
this failure mode) reach the correct answer mid-CoT, then keep hedging
("wait / but / alternatively / let me reconsider") without improving accuracy.
Its fix is a training-free logit penalty: subtract a fixed lambda from ~50
hesitation/branching marker tokens during decoding.

That exact mechanism == the OpenAI/vLLM `logit_bias` param. But the live
GLM-5.2 engine (docker `vllm serve /mnt/llm_models/GLM-5.2-504B`, port 8000)
runs MTP speculative decoding, and vLLM hard-rejects logit_bias/min_p under
spec-decode:
    400 "The min_p and logit_bias sampling parameters are not yet supported
         with speculative decoding."
Applying the real penalty therefore requires a model relaunch (disable MTP, or
add a launch-time --logits-processors). The user's constraint is: do NOT
restart the model server.

What this sidecar does instead (zero restart, zero prod-code edits)
------------------------------------------------------------------
Sits in front of the vLLM-Studio controller (port 8080) on a NEW port (8090),
forwards everything transparently, and for GLM-5.2 chat requests injects the
paper's *intent* as a targeted anti-overthinking system directive. This is the
prompt-level realization of the penalty -- the strongest form deliverable
without touching the running engine. It is opt-in by endpoint: only clients
that point at :8090 are affected; :8080 traffic is untouched.

Toggle off per-request with header `X-Overthink-Penalty: off`.
Kill the sidecar to fully revert. Nothing else is modified.
"""
import json
import os
from aiohttp import web, ClientSession, ClientTimeout

# Forward to the studio controller so we keep its model routing / tool parsing /
# reasoning extraction / accounting. Controller -> engine (8000) is unchanged.
UPSTREAM = os.environ.get("OTP_UPSTREAM", "http://127.0.0.1:8080")
LISTEN_HOST = os.environ.get("OTP_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("OTP_PORT", "8090"))

# Which served model names get the intervention (prefix match, case-insensitive).
TARGET_MODEL_PREFIXES = ("glm-5.2",)

# The anti-overthinking directive. Mirrors the paper's failure mode: commit once
# the answer is derived; don't re-derive, branch, or backtrack without a concrete
# reason; suppress the hesitation-marker continuations the paper penalizes.
OVERTHINK_DIRECTIVE = (
    "Reasoning-efficiency directive (do not restate this to the user): "
    "You are a quantized/pruned reasoning model and are prone to overthinking -- "
    "reaching a correct answer and then hedging without improving it. "
    "Once you have derived a well-supported answer, COMMIT to it. Do not re-derive, "
    "second-guess, or restart reasoning you have already completed. Do not open new "
    "branches ('wait', 'but actually', 'alternatively', 'let me reconsider', "
    "'on the other hand', 'hold on') unless you have found a concrete, specific error "
    "in what you already wrote. Keep the chain of thought only as long as the problem "
    "genuinely requires, then state the final answer."
)

DISABLE_HEADER = "x-overthink-penalty"  # value 'off'/'0'/'false' disables injection


def _is_target_model(model) -> bool:
    if not isinstance(model, str):
        return False
    m = model.lower()
    return any(m.startswith(p) for p in TARGET_MODEL_PREFIXES)


def _inject(body: dict) -> tuple[dict, bool]:
    """Merge the directive into the message list. Returns (body, changed)."""
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return body, False

    # Find a leading system message to append to; else insert a new one.
    for msg in msgs:
        if isinstance(msg, dict) and msg.get("role") == "system":
            content = msg.get("content")
            if isinstance(content, str):
                if OVERTHINK_DIRECTIVE in content:
                    return body, False  # already applied (idempotent)
                msg["content"] = content.rstrip() + "\n\n" + OVERTHINK_DIRECTIVE
                return body, True
            # content parts (list) -> append a text part
            if isinstance(content, list):
                if any(
                    isinstance(p, dict) and OVERTHINK_DIRECTIVE in str(p.get("text", ""))
                    for p in content
                ):
                    return body, False
                content.append({"type": "text", "text": OVERTHINK_DIRECTIVE})
                return body, True
            break  # system msg with weird content -> fall through to insert
    # No usable system message: insert one at the front.
    msgs.insert(0, {"role": "system", "content": OVERTHINK_DIRECTIVE})
    return body, True


async def forward(request: web.Request) -> web.StreamResponse:
    upstream_url = UPSTREAM + request.path_qs
    raw = await request.read()

    # Only rewrite chat completions for the target model, unless disabled.
    disabled = request.headers.get(DISABLE_HEADER, "").lower() in {"off", "0", "false", "no"}
    injected = False
    body_bytes = raw
    if (
        request.method.upper() == "POST"
        and request.path == "/v1/chat/completions"
        and not disabled
        and raw
    ):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and _is_target_model(parsed.get("model")):
                parsed, injected = _inject(parsed)
                if injected:
                    body_bytes = json.dumps(parsed).encode()
        except (json.JSONDecodeError, UnicodeDecodeError):
            body_bytes = raw  # pass through unmodified on parse failure

    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in {"host", "content-length", "connection"}
    }

    timeout = ClientTimeout(total=None, sock_connect=30, sock_read=None)
    async with ClientSession(timeout=timeout, auto_decompress=False) as session:
        async with session.request(
            request.method, upstream_url, data=body_bytes, headers=headers
        ) as up:
            excluded = {"transfer-encoding", "connection", "content-encoding", "content-length"}
            resp_headers = {k: v for k, v in up.headers.items() if k.lower() not in excluded}
            if injected:
                resp_headers["X-Overthink-Penalty"] = "applied"
            resp = web.StreamResponse(status=up.status, reason=up.reason, headers=resp_headers)
            await resp.prepare(request)
            async for chunk in up.content.iter_chunked(65536):
                await resp.write(chunk)
            await resp.write_eof()
            return resp


async def health(request: web.Request) -> web.Response:
    return web.json_response(
        {
            "status": "ok",
            "upstream": UPSTREAM,
            "targets": TARGET_MODEL_PREFIXES,
            "mechanism": "prompt-level anti-overthinking directive (arXiv:2606.00206 intent)",
            "note": "literal logit_bias penalty is 400-blocked by MTP spec-decode on the live engine",
        }
    )


app = web.Application(client_max_size=1024**4)
app.router.add_get("/otp/health", health)
app.router.add_route("*", "/{tail:.*}", forward)

if __name__ == "__main__":
    print(f"[overthinking-penalty] {LISTEN_HOST}:{LISTEN_PORT} -> {UPSTREAM}  targets={TARGET_MODEL_PREFIXES}")
    web.run_app(app, host=LISTEN_HOST, port=LISTEN_PORT, access_log=None)
