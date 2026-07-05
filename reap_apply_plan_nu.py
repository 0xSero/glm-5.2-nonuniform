#!/usr/bin/env python
"""NON-UNIFORM prune-apply: per-layer keep counts. Fork of reap_apply_plan.py.
Plan: {layer: {"kept": [ids], "keep": k}} — layers absent from the plan (incl. MTP 78)
are copied untouched. Writes config.num_routed_experts_per_layer (len = num_hidden_layers
+ num_nextn_predict_layers) and n_routed_experts = max over all layers.
  python reap_apply_plan_nu.py --src DIR --dst DIR --plan plan.json [--dry-run]
"""
import json, os, glob, argparse, re, shutil
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--plan", required=True)
ap.add_argument("--dry-run", action="store_true")
A = ap.parse_args()
SHARD_BYTES = 5_000_000_000

plan = json.load(open(A.plan))
remap, kept_rows, keep_n = {}, {}, {}
for L, d in plan.items():
    kept = sorted(d["kept"])
    assert len(kept) == d["keep"] == len(set(kept)), (L, len(kept))
    remap[L] = {old: new for new, old in enumerate(kept)}
    kept_rows[L] = kept
    keep_n[L] = len(kept)

EXP = re.compile(r"\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")
GATEW = re.compile(r"\.layers\.(\d+)\.mlp\.gate\.weight$")
BIAS = re.compile(r"\.layers\.(\d+)\.mlp\.gate\.e_score_correction_bias$")

def out_name(name):
    m = EXP.search(name)
    if m:
        L, e = m.group(1), int(m.group(2))
        if L not in remap:
            return name, None
        if e not in remap[L]:
            return None
        new = remap[L][e]
        return name[:m.start()] + ".layers.%s.mlp.experts.%d." % (L, new) + name[m.end():], None
    m = GATEW.search(name) or BIAS.search(name)
    if m and m.group(1) in kept_rows:
        return name, ("rows", m.group(1))
    return name, None

shards = sorted(glob.glob(A.src + "/*.safetensors"))
print("source shards:", len(shards), "| plan layers:", len(plan),
      "keep min/max/total:", min(keep_n.values()), max(keep_n.values()), sum(keep_n.values()))
os.makedirs(A.dst, exist_ok=True)
out_index = {}; buf = {}; buf_bytes = 0; out_i = 0
n_out = n_drop = 0; tot_bytes = 0; per_layer = {}

def flush():
    global buf, buf_bytes, out_i
    if not buf: return
    fname = "model-%05d.safetensors" % out_i
    if not A.dry_run:
        save_file(buf, os.path.join(A.dst, fname), metadata={"format": "pt"})
    for k in buf: out_index[k] = fname
    buf = {}; buf_bytes = 0; out_i += 1

for si, sh in enumerate(shards):
    with safe_open(sh, framework="pt") as f:
        for name in f.keys():
            res = out_name(name)
            if res is None:
                n_drop += 1; continue
            oname, sl = res
            t = f.get_tensor(name)
            if sl is not None:
                t = t[kept_rows[sl[1]]]
            mm = EXP.search(oname)
            if mm:
                per_layer.setdefault(mm.group(1), set()).add(int(mm.group(2)))
            n_out += 1; tot_bytes += t.numel() * t.element_size()
            if not A.dry_run:
                buf[oname] = t.clone(); buf_bytes += t.numel() * t.element_size()
                if buf_bytes >= SHARD_BYTES: flush()
    print("shard %d/%d done, out %.1f GB" % (si + 1, len(shards), tot_bytes / 1e9), flush=True)
flush()

bad = {L: len(s) for L, s in per_layer.items()
       if L in keep_n and len(s) != keep_n[L]}
unplanned = {L: len(s) for L, s in per_layer.items() if L not in keep_n}
print("tensors:", n_out, "dropped:", n_drop, "~%.1f GB" % (tot_bytes/1e9),
      "| per-layer counts ok:", not bad, (bad if bad else ""),
      "| untouched layers:", unplanned)
if not A.dry_run:
    json.dump({"metadata": {"total_size": tot_bytes}, "weight_map": out_index},
              open(os.path.join(A.dst, "model.safetensors.index.json"), "w"))
    for fn in os.listdir(A.src):
        if fn.endswith(".safetensors") or fn in ("model.safetensors.index.json", "config.json"):
            continue
        s = os.path.join(A.src, fn)
        if os.path.isfile(s):
            shutil.copy(s, os.path.join(A.dst, fn))
    cfg = json.load(open(os.path.join(A.src, "config.json")))
    n_layers = cfg["num_hidden_layers"] + cfg.get("num_nextn_predict_layers", 0)
    orig = cfg["n_routed_experts"]
    dense = cfg.get("first_k_dense_replace", 0)
    per_layer_cfg = []
    for i in range(n_layers):
        if i < dense:
            per_layer_cfg.append(0)
        else:
            per_layer_cfg.append(keep_n.get(str(i), orig))
    cfg["num_routed_experts_per_layer"] = per_layer_cfg
    cfg["n_routed_experts"] = max(per_layer_cfg)
    json.dump(cfg, open(os.path.join(A.dst, "config.json"), "w"), indent=2)
    print("config: n_routed_experts=%d, per-layer list len %d" % (cfg["n_routed_experts"], n_layers))
    print("WROTE ->", A.dst, "shards:", out_i)
else:
    print("DRY-RUN ok")
