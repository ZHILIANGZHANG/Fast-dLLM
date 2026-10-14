# SPDX-License-Identifier: Apache-2.0
"""
Feasibility probe: can re-running only the last k layers recover the
conditional distribution of still-masked tokens after some tokens are committed?

The decoding trajectory follows Fast-dLLM's threshold rule (commit every masked
position in the current block whose confidence >= threshold, and always the
argmax one). Every forward pass is a full, cache-free pass so the "truth" is exact.

At each step, with P = positions committed this step and F = positions in the
current block that stay masked:

  old    : p_j(. | x_t)                     from the step's forward pass
  truth  : p_j(. | x_t, x_P = v*)           full 32-layer rerun with P filled in
  For k in --ks, start from the old hidden states at the input of layer L-k,
  modify some rows, rerun only layers L-k..L-1 over the full sequence:
    tail_none        : nothing modified (sanity check: must equal `old`)
    oracle_pivot     : rows P replaced by the truth run's hidden states at L-k
                       (best case for any method that only edits the pivots)
    oracle_followers : rows F replaced by the truth run's hidden states at L-k
                       (tells whether the information already sits in F by L-k)
    embed_impulse    : rows P += gamma * (||h|| / mean||E||) * (E[v*] - sum_topM p E[v])
                       (the ISCP-style heuristic, without the norm-rescaling bug)

For followers whose truth distribution actually moved (argmax flipped or
KL(truth||old) > --kl_changed), we report how much of the gap each variant closes:
  recovery = 1 - sum KL(truth||variant) / sum KL(truth||old)
plus top-1 agreement with truth and unlock precision/recall at the threshold.

Step types are reported separately:
  parallel : >= 2 masked positions above threshold
  single   : exactly 1 above threshold
  fallback : none above threshold, the argmax is forced (pivot conf < threshold)

Extra control on parallel steps: commit only the top-1 above-threshold position
and check whether the other above-threshold positions change their argmax.

Example:
  python probe_tail_conditioning.py --num_questions 100 --gen_length 256 \
      --block_length 32 --threshold 0.9 --ks 1,2,4,8,16 --out probe_out
"""

import argparse
import json
import math
import os
import re
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

MASK_ID = 126336


# ----------------------------------------------------------------------------
# Model helpers (LLaDA layout: model.model.transformer.{wte, blocks, ln_f, ff_out})
# ----------------------------------------------------------------------------

def core(model):
    return model.model


def run_full(model, x):
    """Full cache-free forward. Returns (logits [N, V] float32, hidden list).

    hidden[l] is the input to block l for l in [0, L); hidden[L] is post-ln_f.
    """
    out = model(x, output_hidden_states=True, use_cache=False)
    return out.logits[0].float(), out.hidden_states


def hidden_to_logits(model, h):
    m = core(model)
    h = m.transformer.ln_f(h)
    if m.config.weight_tying:
        logits = F.linear(h, m.transformer.wte.weight, None)
    else:
        logits = m.transformer.ff_out(h)
    if m.config.scale_logits:
        logits = logits * (1 / math.sqrt(m.config.d_model))
    return logits[0].float()


def run_tail(model, h_entry, k):
    """Rerun the last k blocks over the full sequence starting from h_entry."""
    blocks = core(model).transformer.blocks
    L = len(blocks)
    h = h_entry
    for b in blocks[L - k:]:
        h, _ = b(h, attention_bias=None, layer_past=None, use_cache=False, replace_position=None)
    return hidden_to_logits(model, h)


def kl(p_log, q_log):
    """KL(p || q) per row, inputs are log-probs [n, V]."""
    return (p_log.exp() * (p_log - q_log)).sum(-1)


# ----------------------------------------------------------------------------
# Probe
# ----------------------------------------------------------------------------

class Stats:
    def __init__(self):
        # key: (step_type, variant, k) -> accumulators
        self.acc = defaultdict(lambda: defaultdict(float))
        self.steps = defaultdict(int)
        self.commits = defaultdict(int)
        self.oracle_extra = defaultdict(float)   # followers that truth would unlock
        self.par_check = defaultdict(float)
        self.sanity_max_kl = 0.0

    def add(self, key, **kw):
        a = self.acc[key]
        for name, v in kw.items():
            a[name] += float(v)


def embed_impulse(model, h_rows, pivot_logp, chosen, gamma, top_m, emb_mean_norm):
    E = core(model).transformer.wte.weight
    vals, ids = torch.topk(pivot_logp.exp(), k=top_m, dim=-1)
    vals = vals / vals.sum(-1, keepdim=True)
    expected = (vals.unsqueeze(-1).to(E.dtype) * E[ids]).sum(1)
    d = (E[chosen] - expected).float()
    scale = h_rows.float().norm(dim=-1, keepdim=True) / emb_mean_norm
    return (h_rows.float() + gamma * scale * d).to(h_rows.dtype)


@torch.no_grad()
def probe_question(model, prompt_ids, args, stats, records):
    device = prompt_ids.device
    Lp = prompt_ids.shape[1]
    x = torch.full((1, Lp + args.gen_length), MASK_ID, dtype=torch.long, device=device)
    x[:, :Lp] = prompt_ids
    L = len(core(model).transformer.blocks)
    ks = [k for k in args.ks if 1 <= k <= L]
    E = core(model).transformer.wte.weight
    emb_mean_norm = E.float().norm(dim=-1).mean()
    nfe = 0

    for nb in range(args.gen_length // args.block_length):
        s = Lp + nb * args.block_length
        e = s + args.block_length
        while (x[0, s:e] == MASK_ID).any():
            logits, H = run_full(model, x)
            nfe += 1
            logp = F.log_softmax(logits, -1)
            conf, pred = logp.exp().max(-1)

            blk_mask = torch.zeros_like(x[0], dtype=torch.bool)
            blk_mask[s:e] = x[0, s:e] == MASK_ID
            masked = blk_mask.nonzero(as_tuple=True)[0]
            above = masked[conf[masked] >= args.threshold]
            top1 = masked[conf[masked].argmax()]

            if len(above) >= 2:
                stype = "parallel"
            elif len(above) == 1:
                stype = "single"
            else:
                stype = "fallback"
            pivots = above if len(above) > 0 else top1.view(1)
            followers = masked[~torch.isin(masked, pivots)]
            stats.steps[stype] += 1
            stats.commits[stype] += len(pivots)

            # Control: in parallel steps, does committing only the top-1 change the others?
            if stype == "parallel" and args.parallel_check:
                x_c = x.clone()
                x_c[0, top1] = pred[top1]
                lg_c, _ = run_full(model, x_c)
                others = above[above != top1]
                flips = (lg_c[others].argmax(-1) != pred[others]).float()
                stats.par_check["n"] += len(others)
                stats.par_check["flips"] += flips.sum().item()
                stats.par_check["kl_sum"] += kl(F.log_softmax(lg_c[others], -1), logp[others]).sum().item()

            x_next = x.clone()
            x_next[0, pivots] = pred[pivots]

            if len(followers) > 0:
                lg_t, H_t = run_full(model, x_next)
                lp_t = F.log_softmax(lg_t[followers], -1)
                lp_o = logp[followers]
                kl_to = kl(lp_t, lp_o)
                conf_t, pred_t = lp_t.exp().max(-1)
                changed = (pred_t != pred[followers]) | (kl_to > args.kl_changed)
                unlock_t = conf_t >= args.threshold
                stats.oracle_extra[stype] += unlock_t.sum().item()

                variants = {}
                for k in ks:
                    h0 = H[L - k]
                    variants[("tail_none", k)] = run_tail(model, h0, k)
                    h = h0.clone()
                    h[0, pivots] = H_t[L - k][0, pivots]
                    variants[("oracle_pivot", k)] = run_tail(model, h, k)
                    h = h0.clone()
                    h[0, followers] = H_t[L - k][0, followers]
                    variants[("oracle_followers", k)] = run_tail(model, h, k)
                    for g in args.gammas:
                        h = h0.clone()
                        h[0, pivots] = embed_impulse(
                            model, h0[0, pivots], logp[pivots], pred[pivots], g, args.top_m, emb_mean_norm
                        )
                        variants[(f"embed_impulse@{g}", k)] = run_tail(model, h, k)

                for (vname, k), lg_v in variants.items():
                    lp_v = F.log_softmax(lg_v[followers], -1)
                    if vname == "tail_none":
                        stats.sanity_max_kl = max(stats.sanity_max_kl, kl(lp_o, lp_v).max().item())
                    kl_tv = kl(lp_t, lp_v)
                    conf_v, pred_v = lp_v.exp().max(-1)
                    unlock_v = (conf_v >= args.threshold) & (pred_v == pred_t)
                    c = changed
                    stats.add(
                        (stype, vname, k),
                        n_follow=len(followers),
                        n_changed=c.sum(),
                        kl_old_changed=kl_to[c].sum(),
                        kl_var_changed=kl_tv[c].sum(),
                        top1_changed=(pred_v[c] == pred_t[c]).sum(),
                        kl_old_all=kl_to.sum(),
                        kl_var_all=kl_tv.sum(),
                        unlock_truth=unlock_t.sum(),
                        unlock_hit=(unlock_v & unlock_t).sum(),
                        unlock_pred=(conf_v >= args.threshold).sum(),
                        unlock_pred_ok=unlock_v.sum(),
                    )

                if args.dump_records:
                    pv_conf = conf[pivots].tolist()
                    for idx in range(len(followers)):
                        records.append({
                            "step_type": stype,
                            "pivot_conf": pv_conf,
                            "pos": int(followers[idx]) - Lp,
                            "conf_old": float(conf[followers[idx]]),
                            "conf_truth": float(conf_t[idx]),
                            "argmax_flip": bool(pred_t[idx] != pred[followers[idx]]),
                            "kl_truth_old": float(kl_to[idx]),
                        })

            x = x_next

    return x, nfe


def load_questions(args):
    if args.data:
        with open(args.data) as f:
            rows = [json.loads(l) for l in f if l.strip()]
    else:
        from datasets import load_dataset
        rows = list(load_dataset("openai/gsm8k", "main", split="test"))
    return rows[: args.num_questions]


def last_number(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def summarize(stats, args):
    out = {"steps": dict(stats.steps), "commits_per_step": {}, "oracle_extra_per_step": {},
           "tail_none_max_kl": stats.sanity_max_kl, "table": []}
    for t, n in stats.steps.items():
        out["commits_per_step"][t] = stats.commits[t] / n
        out["oracle_extra_per_step"][t] = stats.oracle_extra[t] / n
    if stats.par_check["n"]:
        out["parallel_check"] = {
            "n_others": stats.par_check["n"],
            "argmax_flip_rate": stats.par_check["flips"] / stats.par_check["n"],
            "mean_kl": stats.par_check["kl_sum"] / stats.par_check["n"],
        }
    for (t, v, k), a in sorted(stats.acc.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        nc = max(a["n_changed"], 1)
        out["table"].append({
            "step_type": t, "variant": v, "k": k,
            "followers": int(a["n_follow"]), "changed": int(a["n_changed"]),
            "recovery_changed": 1 - a["kl_var_changed"] / max(a["kl_old_changed"], 1e-12),
            "top1_agree_changed": a["top1_changed"] / nc,
            "recovery_all": 1 - a["kl_var_all"] / max(a["kl_old_all"], 1e-12),
            "unlock_recall": a["unlock_hit"] / max(a["unlock_truth"], 1),
            "unlock_precision": a["unlock_pred_ok"] / max(a["unlock_pred"], 1),
        })
    return out


def print_summary(summ):
    print("\n=== step types ===")
    for t, n in summ["steps"].items():
        print(f"{t:9s} steps={n:6d}  commits/step={summ['commits_per_step'][t]:.2f}  "
              f"truth-unlockable followers/step={summ['oracle_extra_per_step'][t]:.2f}")
    if "parallel_check" in summ:
        pc = summ["parallel_check"]
        print(f"parallel check: n={pc['n_others']} argmax flip rate={pc['argmax_flip_rate']:.4f} mean KL={pc['mean_kl']:.4f}")
    print(f"sanity: max KL(old || tail_none) = {summ['tail_none_max_kl']:.2e} (should be ~0)")
    print("\n=== recovery on followers whose truth distribution moved ===")
    hdr = f"{'type':9s} {'variant':22s} {'k':>3s} {'changed':>8s} {'recov':>7s} {'top1':>6s} {'unl_R':>6s} {'unl_P':>6s}"
    print(hdr)
    for r in summ["table"]:
        if r["variant"] == "tail_none":
            continue
        print(f"{r['step_type']:9s} {r['variant']:22s} {r['k']:3d} {r['changed']:8d} "
              f"{r['recovery_changed']:7.3f} {r['top1_agree_changed']:6.3f} "
              f"{r['unlock_recall']:6.3f} {r['unlock_precision']:6.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--data", default=None, help="jsonl with 'question' and 'answer'; default: GSM8K test")
    ap.add_argument("--num_questions", type=int, default=100)
    ap.add_argument("--gen_length", type=int, default=256)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--ks", type=lambda s: [int(v) for v in s.split(",")], default=[1, 2, 4, 8, 16])
    ap.add_argument("--gammas", type=lambda s: [float(v) for v in s.split(",") if v], default=[0.35])
    ap.add_argument("--top_m", type=int, default=5)
    ap.add_argument("--kl_changed", type=float, default=0.1)
    ap.add_argument("--parallel_check", type=int, default=1)
    ap.add_argument("--dump_records", type=int, default=1)
    ap.add_argument("--out", default="probe_out")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from model.modeling_llada import LLaDAModelLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LLaDAModelLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=getattr(torch, args.dtype)
    ).to(device).eval()
    assert core(model).config.block_group_size == 1
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    os.makedirs(args.out, exist_ok=True)
    stats, records = Stats(), []
    correct, total_nfe, t0 = 0, 0, time.time()
    rows = load_questions(args)
    for qi, row in enumerate(rows):
        msg = [{"role": "user", "content": row["question"]}]
        text = tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        ids = torch.tensor(tok(text)["input_ids"], device=device).unsqueeze(0)
        out, nfe = probe_question(model, ids, args, stats, records)
        total_nfe += nfe
        gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        gt = row["answer"].split("####")[-1].strip().replace(",", "")
        correct += int(last_number(gen) == gt)
        print(f"[{qi + 1}/{len(rows)}] nfe={nfe} acc={correct / (qi + 1):.3f} "
              f"elapsed={time.time() - t0:.0f}s", flush=True)

    summ = summarize(stats, args)
    summ.update({"accuracy": correct / len(rows), "mean_nfe": total_nfe / len(rows), "args": vars(args)})
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    if args.dump_records:
        with open(os.path.join(args.out, "followers.jsonl"), "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
    print_summary(summ)
    print(f"\naccuracy={summ['accuracy']:.3f} mean_nfe={summ['mean_nfe']:.1f} -> {args.out}/summary.json")


if __name__ == "__main__":
    main()
