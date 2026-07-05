#!/usr/bin/env python
"""Replace NU176's BF16 layer-78 (MTP) routed experts with the 504B's NVFP4 ones.
504B layer-78 kept = original experts 0..167 in identity order, so tensors carry over
verbatim (renumbering is identity). Rewrites only the NU176 shards that contain
layer-78 mlp expert/gate tensors, appends one new shard with the 504B tensors,
updates index + config (list[78]=168, n_routed_experts=max(list)).
"""
import json, os, re, glob
from safetensors import safe_open
from safetensors.torch import save_file

NU = "/mnt/llm_models/GLM-5.2-REAP-NU176"
SRC504 = "/mnt/llm_models/GLM-5.2-504B"
DROP = re.compile(r"\.layers\.78\.mlp\.(experts\.\d+\.|gate\.weight$|gate\.e_score_correction_bias$)")

idx = json.load(open(NU + "/model.safetensors.index.json"))
wm = idx["weight_map"]
drop_names = [n for n in wm if DROP.search(n)]
shards_to_rewrite = sorted({wm[n] for n in drop_names})
dropset = set(drop_names)
print("dropping", len(drop_names), "tensors from", len(shards_to_rewrite), "shards:", shards_to_rewrite)

freed = 0
for sh in shards_to_rewrite:
    path = os.path.join(NU, sh)
    keep = {}
    with safe_open(path, framework="pt") as f:
        for name in f.keys():
            t = f.get_tensor(name)
            if name in dropset:
                freed += t.numel() * t.element_size()
            else:
                keep[name] = t.clone()
    tmp = path + ".tmp"
    save_file(keep, tmp, metadata={"format": "pt"})
    os.replace(tmp, path)
    print("rewrote", sh, "kept", len(keep))
for n in drop_names:
    del wm[n]
print("freed %.1f GB" % (freed / 1e9))

# pull 504B layer-78 mlp experts + gate + bias
GRAB = re.compile(r"\.layers\.78\.mlp\.(experts\.\d+\.|gate\.weight$|gate\.e_score_correction_bias$)")
idx504 = json.load(open(SRC504 + "/model.safetensors.index.json"))["weight_map"]
grab = {}
for n, sh in idx504.items():
    if GRAB.search(n):
        grab.setdefault(sh, []).append(n)
new = {}
added = 0
for sh, names in sorted(grab.items()):
    with safe_open(os.path.join(SRC504, sh), framework="pt") as f:
        for n in names:
            t = f.get_tensor(n)
            new[n] = t.clone()
            added += t.numel() * t.element_size()
newshard = "model-mtp168.safetensors"
save_file(new, os.path.join(NU, newshard), metadata={"format": "pt"})
for n in new:
    wm[n] = newshard
print("added", len(new), "tensors (%.2f GB) in" % (added / 1e9), newshard)

exp_ids = sorted({int(m.group(1)) for n in new for m in [re.search(r"experts\.(\d+)\.", n)] if m})
assert exp_ids == list(range(168)), (len(exp_ids), exp_ids[:5], exp_ids[-5:])
gw = [n for n in new if n.endswith("gate.weight")]
print("gate tensors:", gw)

idx["metadata"]["total_size"] = idx["metadata"].get("total_size", 0) - freed + added
json.dump(idx, open(NU + "/model.safetensors.index.json", "w"))

cfg = json.load(open(NU + "/config.json"))
cfg["num_routed_experts_per_layer"][78] = 168
cfg["n_routed_experts"] = max(cfg["num_routed_experts_per_layer"])
json.dump(cfg, open(NU + "/config.json", "w"), indent=2)
print("config: n_routed_experts=%d list[78]=%d" % (cfg["n_routed_experts"], cfg["num_routed_experts_per_layer"][78]))
print("DONE")
