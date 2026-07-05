#!/usr/bin/env bash
# One command: download the model (if needed) and serve it.
#   ./run.sh [MODEL_DIR]
set -euo pipefail
export MODEL_DIR="${1:-${MODEL_DIR:-/mnt/llm_models/GLM-5.2-REAP-NU176}}"

if [ ! -f "$MODEL_DIR/model.safetensors.index.json" ]; then
  echo "== downloading 0xSero/GLM-5.2-REAP-NU176-526B (337GB) to $MODEL_DIR"
  hf download 0xSero/GLM-5.2-REAP-NU176-526B --local-dir "$MODEL_DIR"
fi

docker compose up -d --build
echo "== serving on :${PORT:-8000} (first boot JIT-compiles B12X kernels: 30-40 min; warm boots ~5 min)"
echo "   follow: docker logs -f glm52-nu176"
