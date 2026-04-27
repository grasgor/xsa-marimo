# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "torch",
#     "transformers",
#     "numpy",
#     "matplotlib",
#     "plotly",
#     "Pillow",
# ]
# ///

import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium", app_title="Exclusive Self-Attention")


# ==============================================================================
# Imports
# ==============================================================================

@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import math
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from transformers import AutoTokenizer, AutoModelForCausalLM
    try:
        from tqdm.auto import tqdm
    except ImportError:
        def tqdm(x, **kw): return x
    DEVICE = torch.device("cpu")
    return (AutoModelForCausalLM, AutoTokenizer, DEVICE, F, go,
            make_subplots, math, nn, np, optim, os, plt,
            time, torch, tqdm)


@app.cell
def _(go, plt):
    """Global visual theme for all matplotlib + plotly plots in this notebook."""
    import plotly.io as pio

    plt.style.use("default")
    plt.rcParams.update({
        "figure.facecolor": "#f4f7fb",
        "axes.facecolor": "#ffffff",
        "axes.edgecolor": "#cfd8e3",
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.color": "#e8eef7",
        "grid.linewidth": 0.9,
        "grid.alpha": 1.0,
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.labelcolor": "#243447",
        "xtick.color": "#334155",
        "ytick.color": "#334155",
        "axes.titleweight": "semibold",
        "axes.titlepad": 9.0,
        "legend.frameon": False,
        "mathtext.fontset": "stix",
    })

    _palette = ["#2F6DA5", "#F08A24", "#2EA46F", "#D64550", "#18A3AC", "#7A5CC2"]
    pio.templates["xsa_clean"] = go.layout.Template(
        layout=go.Layout(
            paper_bgcolor="#f4f7fb",
            plot_bgcolor="#ffffff",
            colorway=_palette,
            font=dict(family="DejaVu Sans, Arial", size=12, color="#1f2937"),
            hoverlabel=dict(bgcolor="#ffffff", font=dict(color="#111827")),
            xaxis=dict(showgrid=False, zeroline=False, linecolor="#cfd8e3"),
            yaxis=dict(showgrid=True, gridcolor="#e8eef7", zeroline=False, linecolor="#cfd8e3"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        ),
    )
    pio.templates.default = "xsa_clean+plotly_white"
    return


# ==============================================================================
# Training infrastructure (used in §6 learnable-α and §7 4-way training proof)
# ==============================================================================

@app.cell
def _(AutoTokenizer, DEVICE, F, math, nn, np, optim, plt, torch, tqdm):
    """Tiny LM (RoPE + RMSNorm + SwiGLU + α-XSA) + Muon optimizer + data helpers.
    All ported from xsa_competition.py: kept self-contained in one cell so
    downstream training cells can pull whatever they need.
    """

    # ── α-XSA projection ──────────────────────────────────────────────────────
    def project_exclusive(y, v, alpha=1.0):
        v_hat = F.normalize(v, dim=-1, eps=1e-8)
        return y - alpha * (y * v_hat).sum(-1, keepdim=True) * v_hat

    # ── RoPE helpers ──────────────────────────────────────────────────────────
    def _rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def _build_rope(seq_len, head_dim, base=10000.0, device="cpu", dtype=torch.float32):
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        pos = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(pos, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None].to(dtype), emb.sin()[None, None].to(dtype)

    def _apply_rope(q, k, cos, sin):
        return (q * cos + _rotate_half(q) * sin), (k * cos + _rotate_half(k) * sin)

    # ── RMSNorm ───────────────────────────────────────────────────────────────
    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps
        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

    # ── tiny attention with fixed α ───────────────────────────────────────────
    def _attn_metrics(attn_w, v, y):
        """Same metric as paper, on training-time (B,H,S,D) tensors."""
        B, H, S, D = v.shape
        vn = F.normalize(v, dim=-1)
        yn = F.normalize(y, dim=-1)
        pair = vn @ vn.transpose(-1, -2)
        up = torch.triu(torch.ones(S, S, dtype=torch.bool, device=v.device), diagonal=1)
        vc = pair[..., up].mean().item() if up.any() else float("nan")
        diag = attn_w.diagonal(dim1=-2, dim2=-1)
        if S > 1:
            diag = diag[..., 1:]
        da = diag.mean().item()
        oc = (yn * vn).sum(dim=-1).mean().item()
        return {"value_cos": vc, "diag_attn": da, "output_cos": oc}

    class TinyXSAAttention(nn.Module):
        def __init__(self, model_dim, num_heads, alpha):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = model_dim // num_heads
            self.alpha = alpha
            self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
            self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
            self.v_proj = nn.Linear(model_dim, model_dim, bias=False)
            self.out_proj = nn.Linear(model_dim, model_dim, bias=False)

        def forward(self, x, collect_metrics=False):
            b, s, w = x.shape
            q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
            cos, sin = _build_rope(s, self.head_dim, device=x.device, dtype=q.dtype)
            q, k = _apply_rope(q, k, cos, sin)
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            mask = torch.triu(torch.ones(s, s, device=x.device, dtype=torch.bool), diagonal=1)
            attn_w = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
            raw = attn_w @ v
            v_hat = F.normalize(v, dim=-1)
            out = raw - self.alpha * (raw * v_hat).sum(-1, keepdim=True) * v_hat
            merged = out.transpose(1, 2).contiguous().view(b, s, w)
            metrics = _attn_metrics(attn_w, v, out) if collect_metrics else None
            return self.out_proj(merged), metrics

    class SwiGLU(nn.Module):
        def __init__(self, dim):
            super().__init__()
            h = int(dim * 8 / 3)
            self.w1 = nn.Linear(dim, h, bias=False)
            self.w2 = nn.Linear(dim, h, bias=False)
            self.w3 = nn.Linear(h, dim, bias=False)
        def forward(self, x):
            return self.w3(F.silu(self.w1(x)) * self.w2(x))

    class TinyBlock(nn.Module):
        def __init__(self, dim, heads, alpha):
            super().__init__()
            self.norm1 = RMSNorm(dim); self.attn = TinyXSAAttention(dim, heads, alpha)
            self.norm2 = RMSNorm(dim); self.ff = SwiGLU(dim)
        def forward(self, x, collect_metrics=False):
            a, m = self.attn(self.norm1(x), collect_metrics=collect_metrics)
            x = x + a
            return x + self.ff(self.norm2(x)), m

    class TinyTokenLM(nn.Module):
        def __init__(self, vocab_size, model_dim, num_heads, num_layers, alpha):
            super().__init__()
            self.token_emb = nn.Embedding(vocab_size, model_dim)
            self.blocks = nn.ModuleList(
                [TinyBlock(model_dim, num_heads, alpha) for _ in range(num_layers)])
            self.norm_f = RMSNorm(model_dim)
            self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)
            self.lm_head.weight = self.token_emb.weight
        def forward(self, idx, targets=None, collect_metrics=False):
            x = self.token_emb(idx)
            metrics = []
            for blk in self.blocks:
                x, m = blk(x, collect_metrics=collect_metrics)
                if collect_metrics:
                    metrics.append(m)
            logits = self.lm_head(self.norm_f(x))
            loss = None
            if targets is not None:
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            return logits, loss, metrics

    # ── learnable-α attention ─────────────────────────────────────────────────
    class LearnableAlphaAttention(nn.Module):
        def __init__(self, model_dim, num_heads):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = model_dim // num_heads
            self._alpha_raw = nn.Parameter(torch.zeros(num_heads))   # σ → α∈(0,1)
            self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
            self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
            self.v_proj = nn.Linear(model_dim, model_dim, bias=False)
            self.out_proj = nn.Linear(model_dim, model_dim, bias=False)

        @property
        def alpha(self):
            return torch.sigmoid(self._alpha_raw)

        def forward(self, x):
            b, s, w = x.shape
            q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
            cos, sin = _build_rope(s, self.head_dim, device=x.device, dtype=q.dtype)
            q, k = _apply_rope(q, k, cos, sin)
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            mask = torch.triu(torch.ones(s, s, device=x.device, dtype=torch.bool), diagonal=1)
            attn_w = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
            raw = attn_w @ v
            a = self.alpha.view(1, self.num_heads, 1, 1)
            v_hat = F.normalize(v, dim=-1)
            out = raw - a * (raw * v_hat).sum(-1, keepdim=True) * v_hat
            merged = out.transpose(1, 2).contiguous().view(b, s, w)
            return self.out_proj(merged)

    class LearnableAlphaBlock(nn.Module):
        def __init__(self, dim, heads):
            super().__init__()
            self.norm1 = RMSNorm(dim); self.attn = LearnableAlphaAttention(dim, heads)
            self.norm2 = RMSNorm(dim); self.ff = SwiGLU(dim)
        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            return x + self.ff(self.norm2(x))

    class LearnableAlphaLM(nn.Module):
        def __init__(self, vocab_size, model_dim, num_heads, num_layers):
            super().__init__()
            self.token_emb = nn.Embedding(vocab_size, model_dim)
            self.blocks = nn.ModuleList(
                [LearnableAlphaBlock(model_dim, num_heads) for _ in range(num_layers)])
            self.norm_f = RMSNorm(model_dim)
            self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)
            self.lm_head.weight = self.token_emb.weight
        def forward(self, idx, targets=None):
            x = self.token_emb(idx)
            for blk in self.blocks:
                x = blk(x)
            logits = self.lm_head(self.norm_f(x))
            loss = None
            if targets is not None:
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            return logits, loss
        def learned_alphas(self):
            return {li: blk.attn.alpha.mean().item()
                    for li, blk in enumerate(self.blocks)}

    # ── Muon optimizer ────────────────────────────────────────────────────────
    class Muon(optim.Optimizer):
        """Minimal Muon: momentum-preconditioned gradient (geometry-aware)."""
        def __init__(self, params, lr=3e-4, beta=0.95, eps=1e-8):
            super().__init__(params, dict(lr=lr, beta=beta, eps=eps))
        @torch.no_grad()
        def step(self):
            for g in self.param_groups:
                for p in g["params"]:
                    if p.grad is None:
                        continue
                    s = self.state[p]
                    if not s:
                        s["v"] = torch.zeros_like(p)
                    v = s["v"]
                    v.mul_(g["beta"]).addcmul_(p.grad, p.grad, value=1 - g["beta"])
                    p.add_(p.grad / (v.sqrt() + g["eps"]), alpha=-g["lr"])

    # ── data helpers ──────────────────────────────────────────────────────────
    import subprocess

    _TEXT_URLS = [
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "https://www.gutenberg.org/files/11/11-0.txt",
        "https://www.gutenberg.org/files/1342/1342-0.txt",
    ]

    _FALLBACK_TEXT = (
        "To be, or not to be, that is the question. "
        "All the world's a stage, and all the men and women merely players. "
        "It is a truth universally acknowledged, that a single man in possession of a good fortune, "
        "must be in want of a wife. "
        "Alice was beginning to get very tired of sitting by her sister on the bank. "
    ) * 250

    _data_cache = {}

    def _fetch_text_via_curl(url):
        cmd = ["curl", "-LfsS", "--max-time", "12", url]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except Exception:
            return None
        if proc.returncode != 0:
            return None
        text = (proc.stdout or "").strip()
        if len(text) < 2000:
            return None
        return text

    def build_token_data():
        if "data" in _data_cache:
            return _data_cache["data"]
        chunks = []
        used_urls = []
        for _url in _TEXT_URLS:
            _txt = _fetch_text_via_curl(_url)
            if _txt:
                chunks.append(_txt[:120_000])
                used_urls.append(_url)
        if not chunks:
            chunks = [_FALLBACK_TEXT]
        raw_text = "\n\n".join(chunks)
        tok = AutoTokenizer.from_pretrained("gpt2")
        ids = torch.tensor(
            tok.encode(raw_text, add_special_tokens=False),
            dtype=torch.long,
        )
        split = int(0.9 * len(ids))
        result = (ids[:split], ids[split:], tok.vocab_size)
        _data_cache["data"] = result
        _data_cache["info"] = {
            "urls_used": used_urls,
            "num_sources": len(used_urls) if used_urls else 1,
            "chars": len(raw_text),
            "source_mode": "curl" if used_urls else "fallback",
            "split_ratio": "90/10",
            "vocab_size": tok.vocab_size,
        }
        return result

    def get_token_data_info():
        if "info" not in _data_cache:
            build_token_data()
        return _data_cache["info"]

    def sample_positions(data, batch_size, seq_len, gen):
        high = len(data) - seq_len - 1
        return torch.randint(0, max(1, high), (batch_size,), generator=gen)

    def make_batch(data, positions, seq_len):
        x = torch.stack([data[p: p + seq_len] for p in positions])
        y = torch.stack([data[p + 1: p + seq_len + 1] for p in positions])
        return x, y

    def collect_tiny_metrics(model, val_data, seq_len, batch_size=8, num_batches=4,
                             seed=2026, device="cpu"):
        model.eval()
        gen = torch.Generator().manual_seed(seed)
        store = None
        with torch.no_grad():
            for _ in range(num_batches):
                pos = sample_positions(val_data, batch_size, seq_len, gen)
                x, _ = make_batch(val_data, pos, seq_len)
                _, _, lm = model(x.to(device), collect_metrics=True)
                if store is None:
                    store = {k: [0.0] * len(lm)
                             for k in ["value_cos", "diag_attn", "output_cos"]}
                for li, m in enumerate(lm):
                    for k in store:
                        store[k][li] += m[k]
        return {k: [v / num_batches for v in store[k]] for k in store}

    # ── 4-way training ────────────────────────────────────────────────────────
    def train_four_models(steps, seq_len, batch_size, model_dim, num_layers, lr=3e-4, on_step=None):
        train_data, val_data, vocab_size = build_token_data()
        sched_gen = torch.Generator().manual_seed(123)
        schedule = [sample_positions(train_data, batch_size, seq_len, sched_gen)
                    for _ in range(steps)]
        configs = [
            ("sa_adam", 0.0, "adamw"),
            ("sa_muon", 0.0, "muon"),
            ("xsa_adam", 1.0, "adamw"),
            ("xsa_muon", 1.0, "muon"),
        ]
        results = {}
        for name, alpha, opt_name in configs:
            torch.manual_seed(7)
            model = TinyTokenLM(vocab_size, model_dim, 4, num_layers, alpha).to(DEVICE)
            optimizer = (
                optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
                if opt_name == "adamw" else Muon(model.parameters(), lr=lr)
            )
            losses = []
            for step_idx, pos in enumerate(schedule):
                x, y = make_batch(train_data, pos, seq_len)
                _, loss, _ = model(x.to(DEVICE), y.to(DEVICE))
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                losses.append(loss.item())
                if on_step:
                    on_step(name, step_idx, steps, loss.item())
            with torch.no_grad():
                pv = sample_positions(val_data, min(batch_size, 32), seq_len,
                                      torch.Generator().manual_seed(99))
                xv, yv = make_batch(val_data, pv, seq_len)
                _, val_loss, _ = model(xv.to(DEVICE), yv.to(DEVICE))
            metrics = collect_tiny_metrics(model, val_data, seq_len, device=DEVICE)
            results[name] = {"losses": losses, "val_loss": val_loss.item(),
                             "metrics": metrics}
        return results, vocab_size

    # ── learnable-α training ──────────────────────────────────────────────────
    def train_learnable_alpha(steps, seq_len, batch_size, model_dim, num_layers, lr=3e-4, on_step=None):
        train_data, _, vocab_size = build_token_data()
        sched_gen = torch.Generator().manual_seed(123)
        schedule = [sample_positions(train_data, batch_size, seq_len, sched_gen)
                    for _ in range(steps)]
        torch.manual_seed(7)
        model = LearnableAlphaLM(vocab_size, model_dim, 4, num_layers).to(DEVICE)
        opt = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
        for step_idx, pos in enumerate(schedule):
            x, y = make_batch(train_data, pos, seq_len)
            _, loss = model(x.to(DEVICE), y.to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
            if on_step:
                on_step("learn_α", step_idx, len(schedule), loss.item())
        learned = model.learned_alphas()
        per_layer_head = {
            li: [float(a) for a in blk.attn.alpha.detach().cpu().tolist()]
            for li, blk in enumerate(model.blocks)
        }
        flat = np.asarray(
            [a for hs in per_layer_head.values() for a in hs], dtype=float
        )
        if flat.size:
            summary = {
                "mean": float(flat.mean()),
                "median": float(np.median(flat)),
                "std": float(flat.std()),
                "min": float(flat.min()),
                "max": float(flat.max()),
                "p10": float(np.percentile(flat, 10)),
                "p90": float(np.percentile(flat, 90)),
                "frac_ge_08": float((flat >= 0.8).mean()),
                "frac_ge_09": float((flat >= 0.9).mean()),
                "frac_ge_095": float((flat >= 0.95).mean()),
                "n_heads": int(flat.size),
            }
        else:
            summary = {
                "mean": float("nan"),
                "median": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "p10": float("nan"),
                "p90": float("nan"),
                "frac_ge_08": float("nan"),
                "frac_ge_09": float("nan"),
                "frac_ge_095": float("nan"),
                "n_heads": 0,
            }
        return {
            "per_layer": learned,
            "per_layer_head": per_layer_head,
            "summary": summary,
            "mean": summary["mean"],
        }

    # ── plotting ──────────────────────────────────────────────────────────────
    _CRUN = {"sa_adam": "#4C78A8", "sa_muon": "#72B7B2",
             "xsa_adam": "#E45756", "xsa_muon": "#F58518"}

    def plot_four_losses(results):
        labels = ["SA + AdamW", "SA + Muon", "XSA + AdamW", "XSA + Muon"]
        fig, ax = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
        for k, lbl in zip(results, labels):
            _y = np.asarray(results[k]["losses"], dtype=float)
            _x = np.arange(len(_y))
            ax.plot(
                _x, _y, color=_CRUN[k], linewidth=2.2, marker=None,
                label=lbl, alpha=0.95,
            )
            if _y.size:
                ax.scatter(
                    [_x[-1]], [_y[-1]], s=26, color=_CRUN[k],
                    edgecolor="white", linewidth=0.8, zorder=3,
                )
        ax.set_title("CPU training loss: 4 runs from same init & batch schedule",
                     fontsize=11)
        ax.set_xlabel("Step"); ax.set_ylabel("Cross-entropy loss")
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        ax.legend(frameon=False, fontsize=9)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        return fig

    return (
        Muon, TinyTokenLM, LearnableAlphaLM,
        project_exclusive,
        train_four_models, train_learnable_alpha, get_token_data_info,
        plot_four_losses,
    )


# ==============================================================================
# §0: Intro
# ==============================================================================

@app.cell
def _(mo):
    mo.md(r"""
    # Exclusive Self-Attention (arXiv:2603.09078)
    **Paper:** [arXiv:2603.09078](https://arxiv.org/abs/2603.09078): *Exclusive Self-Attention*
    
    > **First-run note:** the first run downloads required Hugging Face models/tokenizers and caches them locally, so initial cells may take longer.

    In trained transformers, attention outputs become progressively similar to each token's own value vector as depth grows. In the paper's 1.3B model trained on 100B tokens, the three per-layer metrics show an increasing trend. The paper calls this the **Attention-Similarity Bias** and proposes XSA to remove it.

    <center>
      <table align="center" style="text-align:center; margin-left:auto; margin-right:auto;">
        <thead>
          <tr><th>Section</th><th>What it does</th></tr>
        </thead>
        <tbody>
          <tr><td>§1 - 2</td><td>Intuition: synthetic geometry + live GPT-2</td></tr>
          <tr><td>§3 - 4</td><td>Reimplementation: 2 live models + 13-model survey</td></tr>
          <tr><td>§5</td><td>Stress tests: corpus + attention-sink isolation</td></tr>
          <tr><td>§6 - 7</td><td>XSA validation: learnable α + 4-way training</td></tr>
        </tbody>
      </table>
    </center>

    $$
    \text{Standard attention:}\quad
    y_i = \sum_j a_{i,j}\,v_j
    $$

    $$
    \text{XSA:}\quad
    z_i = y_i - (y_i \cdot \hat v_i)\,\hat v_i,\qquad
    \hat v_i = \frac{v_i}{\lVert v_i \rVert}
    $$
    """)
    return
# ==============================================================================
# §1: Synthetic Evidence: SA vs XSA on random Q, K, V (no model needed)
# ==============================================================================

@app.cell
def _(mo):
    mo.md(r"""
    ## §1. Synthetic Experiment: measuring bias with random Q, K, V

    Before looking at trained models, it helps to understand *why* there is an inherent bias. Consider a minimal 2-token example:

    - $Q \in \mathbb{R}^{2\times n}$, $K \in \mathbb{R}^{2\times n}$, $V \in \mathbb{R}^{2\times 3}$, with rows $Q=\begin{bmatrix}q_1\\q_2\end{bmatrix}$, $K=\begin{bmatrix}k_1\\k_2\end{bmatrix}$, $V=\begin{bmatrix}v_1\\v_2\end{bmatrix}$

    The scaled dot-product scores for token 1 are:
    $$
    s_{11}=\frac{q_1\cdot k_1}{\sqrt{n}},\qquad s_{12}=\frac{q_1\cdot k_2}{\sqrt{n}}.
    $$

    Row-wise softmax gives attention weights:
    $$
    P_{11}=\frac{e^{s_{11}}}{e^{s_{11}}+e^{s_{12}}},\qquad P_{12}=\frac{e^{s_{12}}}{e^{s_{11}}+e^{s_{12}}},\qquad P_{11}+P_{12}=1.
    $$

    The attention output for token 1 is therefore:
    $$
    o_1 = P_{11}v_1 + P_{12}v_2.
    $$

    Project $v_2$ onto $v_1$: write $v_2 = \alpha v_1 + r$ where $\alpha = \frac{v_2 \cdot v_1}{\lVert v_1 \rVert^2}$ and $r \perp v_1$. Substituting:

    $$
    o_1 = \underbrace{(P_{11}+\alpha P_{12})}_{\beta}\,v_1 + P_{12}r.
    $$

    The cosine between $o_1$ and $v_1$ follows directly from this decomposition:
    $$
    \cos(o_1, v_1) = \frac{\beta\,\lVert v_1 \rVert}{\sqrt{\beta^2\lVert v_1\rVert^2 + P_{12}^2\lVert r\rVert^2}}.
    $$

    The coefficient $\beta = P_{11} + \alpha P_{12}$ is the **inherent bias scale**: always positive as long as $P_{11} > 0$ or $\alpha > 0$. This is a **weight × geometry** effect: even when the model attends away from itself ($P_{11}$ small), similar value vectors ($\alpha \approx 1$) keep $\beta$ large, keeping the cosine high.

    Use the sliders to explore how $\alpha$ (value alignment) and the random seed (which determines $P_{11}$ via random scores) affect the geometry and cosine.
    """)
    return


@app.cell
def _(mo):
    sec1_n = mo.ui.slider(8, 256, value=64, step=8, label="q/k dimension n")
    sec1_align = mo.ui.slider(
        -1.0, 1.0, value=0.35, step=0.05,
        label="Alignment α between v₂ and v₁",
    )
    sec1_seed = mo.ui.slider(0, 99, value=42, step=1, label="Random seed")
    mo.hstack([sec1_n, sec1_align, sec1_seed], justify="start", gap=2)
    return sec1_align, sec1_n, sec1_seed


@app.cell
def _(math, mo, np, plt, sec1_align, sec1_n, sec1_seed, torch):
    torch.manual_seed(sec1_seed.value)
    _n = sec1_n.value

    _q1 = torch.randn(_n)
    _k1 = torch.randn(_n)
    _k2 = torch.randn(_n)

    _s11 = float(torch.dot(_q1, _k1) / math.sqrt(_n))
    _s12 = float(torch.dot(_q1, _k2) / math.sqrt(_n))

    _p1 = torch.softmax(torch.tensor([_s11, _s12]), dim=0).numpy()
    _P11, _P12 = float(_p1[0]), float(_p1[1])

    _alpha = float(sec1_align.value)
    _v1 = np.array([1.0, 0.0, 0.0], dtype=float)
    _orth_mag = float(np.sqrt(max(0.0, 1.0 - _alpha ** 2)))
    _v2 = np.array([_alpha, _orth_mag, 0.0], dtype=float)

    _o1 = _P11 * _v1 + _P12 * _v2
    _r = _v2 - _alpha * _v1
    _along = (_P11 + _alpha * _P12) * _v1
    _beta_inherent = float(_P11 + _alpha * _P12)
    _r_norm = float(np.linalg.norm(_r))

    _num = float(np.dot(_o1, _v1))
    _den = float(np.linalg.norm(_o1) * np.linalg.norm(_v1) + 1e-12)
    _cos_direct = _num / _den

    _fig, (_ax0, _ax1) = plt.subplots(1, 2, figsize=(12.0, 4.5), constrained_layout=True)

    # Panel 1: vector composition in the (v1, r) plane.
    _v1_2 = np.array([1.0, 0.0], dtype=float)
    _v2_2 = np.array([_alpha, _orth_mag], dtype=float)
    _along_2 = np.array([float(_along[0]), 0.0], dtype=float)
    _o1_2 = np.array([float(_o1[0]), float(_o1[1])], dtype=float)

    def _arr(_ax, _start, _end, _color, _label, _ls="-", _lw=2.2, _ox=0.03, _oy=0.03):
        _ax.annotate(
            "", xy=_end, xytext=_start,
            arrowprops=dict(arrowstyle="->", color=_color, lw=_lw, linestyle=_ls),
        )
        _ax.text(
            _end[0] + _ox, _end[1] + _oy, _label,
            color=_color, fontsize=9.5, fontweight="bold",
            ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=_color, alpha=0.9, linewidth=0.7),
        )

    _orig = np.zeros(2)
    _arr(_ax0, _orig,    _v1_2,    "#F08A24", r"$v_1$: self value",         _ox=0.03,  _oy=0.03)
    _arr(_ax0, _orig,    _v2_2,    "#2F6DA5", r"$v_2$: other value",        _ox=0.03,  _oy=0.03)
    _arr(_ax0, _orig,    _o1_2,    "#D64550", r"$o_1$: output",             _ox=0.03,  _oy=0.03)
    _arr(_ax0, _orig,    _along_2, "#E0A100", r"$\beta v_1$: self component",  _ls="--", _lw=2.0, _ox=0.03, _oy=-0.07)
    _arr(_ax0, _along_2, _o1_2,    "#18A3AC", r"$P_{12}r$: cross component",   _ls="-.", _lw=2.0, _ox=0.03, _oy=0.03)

    _x_vals = [0.0, _v1_2[0], _v2_2[0], _along_2[0], _o1_2[0]]
    _y_vals = [0.0, _v1_2[1], _v2_2[1], _along_2[1], _o1_2[1]]
    _xmin = min(_x_vals) - 0.15
    _xmax = max(_x_vals) + 0.55
    _ymin = min(_y_vals) - 0.15
    _ymax = max(_y_vals) + 0.15
    _span = max(_xmax - _xmin, _ymax - _ymin, 1.0)
    _cx, _cy = 0.5 * (_xmin + _xmax), 0.5 * (_ymin + _ymax)
    _ax0.set_xlim(_cx - 0.5 * _span, _cx + 0.5 * _span)
    _ax0.set_ylim(_cy - 0.5 * _span, _cy + 0.5 * _span)
    _ax0.axhline(0.0, color="#9aa5b1", lw=0.8)
    _ax0.axvline(0.0, color="#9aa5b1", lw=0.8)
    _ax0.set_aspect("equal")
    _ax0.set_xlabel(r"Along $v_1$ (self-value direction)")
    _ax0.set_ylabel(r"Orthogonal to $v_1$ (cross-token direction)")
    _ax0.set_title(r"Vector decomposition: $o_1 = \beta v_1 + P_{12}r$")
    _ax0.grid(True, alpha=0.4)

    # Panel 2: cosine as a function of P11 at fixed geometry.
    _p = np.linspace(0.0, 1.0, 300)
    _num_curve = _p + _alpha * (1.0 - _p)
    _den_curve = np.sqrt(_num_curve ** 2 + ((1.0 - _p) * _r_norm) ** 2 + 1e-12)
    _curve = _num_curve / _den_curve
    _ax1.plot(_p, _curve, color="#2EA46F", linewidth=2.4)
    _ax1.scatter([_P11], [_cos_direct], color="#D64550", s=45, zorder=3, edgecolor="white", linewidth=0.9)
    _ax1.axvline(_P11, color="#D64550", ls="--", lw=1.3)
    _ax1.axhline(1.0, color="#6b7280", ls=":", lw=0.9)
    _ax1.set_xlim(0.0, 1.0)
    _ax1.set_ylim(-1.02, 1.02)
    _ax1.set_xlabel(r"$P_{11}$ (attention mass on $v_1$)")
    _ax1.set_ylabel(r"$\cos(o_1, v_1)$")
    _ax1.set_title(r"Cosine sensitivity to $P_{11}$")
    _ax1.grid(True, axis="y")
    _ax1.grid(False, axis="x")

    for _ax in [_ax0, _ax1]:
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)

    mo.vstack([
        mo.hstack([
            mo.stat(f"{_P11:.3f}", label="P₁₁  (self-attn weight)", bordered=True),
            mo.stat(f"{_P12:.3f}", label="P₁₂  (cross-attn weight)", bordered=True),
            mo.stat(f"{_beta_inherent:.3f}", label="β = P₁₁ + α·P₁₂  (inherent bias)", bordered=True),
            mo.stat(f"{_cos_direct:.3f}", label="cos(o₁, v₁)  (output-value alignment)", bordered=True),
        ]),
        mo.as_html(_fig),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    With **fully random** $Q, K, V \in \mathbb{R}^{T\times D}$ and vanilla causal attention, the mean $\cos(y_i, v_i)$ is already non-trivially positive. **XSA** drives it to zero by construction:
    $$ z_i = y_i - \frac{y_i \cdot v_i}{\lVert v_i \rVert^2}\, v_i \implies \cos(z_i, v_i) = 0. $$

    Drag the sliders to see both effects.
    """)
    return


@app.cell
def _(mo):
    syn1_T = mo.ui.slider(8, 128, value=32, step=8, label="Sequence length T")
    syn1_D = mo.ui.slider(16, 256, value=64, step=16, label="Head dimension d")
    syn1_seed = mo.ui.slider(0, 99, value=42, label="Random seed")
    mo.hstack([syn1_T, syn1_D, syn1_seed], justify="start", gap=2)
    return syn1_D, syn1_T, syn1_seed


@app.cell
def _(F, mo, np, plt, syn1_D, syn1_T, syn1_seed, torch):
    torch.manual_seed(syn1_seed.value)
    _T, _D = syn1_T.value, syn1_D.value
    _Q = torch.randn(1, 1, _T, _D)
    _K = torch.randn(1, 1, _T, _D)
    _V = torch.randn(1, 1, _T, _D)

    _mask = torch.triu(torch.ones(_T, _T, dtype=torch.bool), diagonal=1)
    _scores = (_Q @ _K.transpose(-1, -2)) / (_D ** 0.5)
    _scores = _scores.masked_fill(_mask, float("-inf"))
    _A = _scores.softmax(dim=-1)
    _Y = (_A @ _V)[0, 0]                       # (T, D)
    _V_flat = _V[0, 0]

    _cos_sa = F.cosine_similarity(_Y, _V_flat, dim=-1).numpy()
    # XSA: z = y - (y · v_hat) v_hat
    _v_hat = F.normalize(_V_flat, dim=-1, eps=1e-8)
    _Z = _Y - (_Y * _v_hat).sum(-1, keepdim=True) * _v_hat
    _cos_xsa = F.cosine_similarity(_Z, _V_flat, dim=-1).numpy()

    _fig, _axes = plt.subplots(1, 2, figsize=(9.8, 3.7), constrained_layout=True)
    _bins = np.linspace(-1.0, 1.0, 25)
    _axes[0].hist(
        _cos_sa, bins=_bins, alpha=0.88, edgecolor="white",
        color="#D64550", linewidth=0.6,
    )
    _axes[0].axvline(float(np.mean(_cos_sa)), color="#1f2937", lw=1.8, ls="--")
    _axes[0].axvline(0.0, color="#6b7280", lw=1.0, ls=":")
    _axes[0].set_title(r"Standard SA: $\cos(y_i, v_i)$")
    _axes[0].set_xlabel("Cosine similarity")
    _axes[0].set_ylabel("Token count")

    _axes[1].hist(
        _cos_xsa, bins=_bins, alpha=0.9, edgecolor="white",
        color="#18A3AC", linewidth=0.6,
    )
    _axes[1].axvline(float(np.mean(_cos_xsa)), color="#1f2937", lw=1.8, ls="--")
    _axes[1].axvline(0.0, color="#6b7280", lw=1.0, ls=":")
    _axes[1].set_title(r"XSA: $\cos(z_i, v_i)$")
    _axes[1].set_xlabel("Cosine similarity")

    for _ax in _axes:
        _ax.set_xlim(-1.02, 1.02)
        _ax.grid(True, axis="y")
        _ax.grid(False, axis="x")
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)

    _mean_sa = float(np.mean(_cos_sa))
    _mean_xsa = float(np.mean(_cos_xsa))

    mo.vstack([
        mo.as_html(_fig),
        mo.hstack([
            mo.stat(f"{_mean_sa:.3f}", label="SA mean cos(y,v)", bordered=True),
            mo.stat(f"{_mean_xsa:.4f}", label="XSA mean cos(z,v)", bordered=True),
        ]),
        mo.callout(mo.md(
            f"With random weights the mean cosine is **{_mean_sa:.2f}** is present but modest. "
            "In a trained transformer this rises substantially with depth (§3 shows values reaching 0.6 - 0.9 in later layers). "
            "XSA removes it by construction."
        ), kind="info"),
    ])
    return


# ==============================================================================
# §1': Load GPT-2 (used for §2 intuition + §3A + §5)
# ==============================================================================

@app.cell
def _(AutoModelForCausalLM, AutoTokenizer, DEVICE):
    _GPT2_ID = "gpt2"
    tok_g2 = AutoTokenizer.from_pretrained(_GPT2_ID)
    if tok_g2.pad_token_id is None:
        tok_g2.pad_token = tok_g2.eos_token
    model_g2 = AutoModelForCausalLM.from_pretrained(
        _GPT2_ID, attn_implementation="eager"
    ).eval().to(DEVICE)
    _cfg = model_g2.config
    G2_N_LAYERS = _cfg.num_hidden_layers
    G2_N_HEADS = _cfg.num_attention_heads
    G2_HIDDEN = _cfg.hidden_size
    G2_HEAD_DIM = G2_HIDDEN // G2_N_HEADS
    return G2_HEAD_DIM, G2_HIDDEN, G2_N_HEADS, G2_N_LAYERS, model_g2, tok_g2


# ==============================================================================
# §2: Intuition using real tokens (GPT-2)
# ==============================================================================

@app.cell
def _(mo):
    mo.md(r"""
    ## §2. Intuition for the three metrics (live, real tokens)

    We use **GPT-2** for these intuition cells: it's small (124M), fast on CPU, and well-understood. The metrics we compute are architecture-agnostic.

    """)
    return


@app.cell
def _():
    DEFAULT_TEXT = "The quick brown fox jumps over the lazy dog and runs into the deep forest."
    return (DEFAULT_TEXT,)


@app.cell
def _(DEFAULT_TEXT, mo):
    # State holds the *committed* text: only updated when the user clicks the button.
    get_active_text, set_active_text = mo.state(DEFAULT_TEXT)
    return get_active_text, set_active_text


@app.cell
def _(mo):
    mo.md(r"""
    ## The three metrics

    Let $v_i \in \mathbb{R}^d$ be the value vector at token position $i$ (per attention head) and $a_{i,j}$ the softmax attention weight from position $i$ to position $j$. The **per-head pre-projection output** is

    $$
    y_i = \sum_j a_{i,j}\, v_j
    $$


    | Metric | Formula | What it measures |
    |:---|:---|:---|
    | `value_cos` | $\dfrac{1}{\lvert\{(i,j):i<j\}\rvert}\displaystyle\sum_{i<j} \cos(v_i, v_j)$ | Average alignment between value vectors at *different* positions: are they pointing the same way? |
    | `diag_attn` | $\dfrac{1}{S-1}\displaystyle\sum_{i>0} a_{i,i}$ | Average self-attention weight (position 0 excluded: $a_{0,0}=1$ trivially under causal masking) |
    | `output_cos` | $\dfrac{1}{S}\displaystyle\sum_i \cos(y_i, v_i)$ | Average alignment between each token's output and its own value vector |

    """)
    return


@app.cell
def _(DEFAULT_TEXT, mo, set_active_text):
    text_input = mo.ui.text_area(
        value=DEFAULT_TEXT,
        label="Input text",
        full_width=True,
        rows=2,
    )
    update_btn = mo.ui.button(
        label="🚀 Run on this text",
        kind="success",
        on_change=lambda _: set_active_text(text_input.value),
    )
    mo.vstack([text_input, update_btn], gap=0.6)
    return text_input, update_btn


@app.cell
def _(G2_N_HEADS, G2_N_LAYERS, mo):
    layer_slider = mo.ui.slider(
        start=0, stop=G2_N_LAYERS - 1, step=1,
        value=max(0, G2_N_LAYERS - 2),
        label=f"Layer (0 … {G2_N_LAYERS - 1})",
        show_value=True,
    )
    head_slider = mo.ui.slider(
        start=0, stop=G2_N_HEADS - 1, step=1,
        value=0,
        label=f"Head (0 … {G2_N_HEADS - 1})",
        show_value=True,
    )
    mo.hstack([layer_slider, head_slider], justify="start", gap=2)
    return head_slider, layer_slider


@app.cell
def _(
    DEVICE,
    G2_HEAD_DIM,
    G2_HIDDEN,
    G2_N_HEADS,
    G2_N_LAYERS,
    get_active_text,
    model_g2,
    tok_g2,
    torch,
):
    """Single forward pass on GPT-2: caches Q, K, V, attention for every layer.

    GPT-2 fused QKV layout is feature-major: c_attn output is [Q | K | V] of
    size hidden_size each along the last dim. We hook c_attn and slice all three.
    Triggered only when the user commits new text via the 'Run on this text' button.
    """
    _enc = tok_g2(get_active_text(), return_tensors="pt", add_special_tokens=False)
    input_ids = _enc.input_ids.to(DEVICE)
    tokens = [
        t.replace("Ġ", "·").replace("Ċ", "↵")
        for t in tok_g2.convert_ids_to_tokens(input_ids[0].tolist())
    ]

    q_cache = [None] * G2_N_LAYERS
    k_cache = [None] * G2_N_LAYERS
    v_cache = [None] * G2_N_LAYERS
    _hooks = []

    def _make_hook(idx):
        def _h(_m, _inp, out):
            b, s, _ = out.shape
            _q = out[..., :G2_HIDDEN]
            _k = out[..., G2_HIDDEN:2 * G2_HIDDEN]
            _v = out[..., 2 * G2_HIDDEN:]
            q_cache[idx] = _q.view(b, s, G2_N_HEADS, G2_HEAD_DIM).transpose(1, 2)
            k_cache[idx] = _k.view(b, s, G2_N_HEADS, G2_HEAD_DIM).transpose(1, 2)
            v_cache[idx] = _v.view(b, s, G2_N_HEADS, G2_HEAD_DIM).transpose(1, 2)
        return _h

    for _i, _blk in enumerate(model_g2.transformer.h):
        _hooks.append(_blk.attn.c_attn.register_forward_hook(_make_hook(_i)))

    with torch.no_grad():
        _out = model_g2(
            input_ids=input_ids,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    for _h in _hooks:
        _h.remove()

    attentions = _out.attentions       # list of (1, H, S, S)
    queries = q_cache                   # list of (1, H, S, D)
    keys = k_cache                       # list of (1, H, S, D)
    values = v_cache                    # list of (1, H, S, D)
    S = input_ids.shape[1]
    return S, attentions, keys, queries, tokens, values


# ------------------------------------------------------------------------------
# §2.1: value_cos
# ------------------------------------------------------------------------------

@app.cell
def _(mo):
    mo.md(r"""
    ### §2.1 `value_cos`: do value vectors across positions point the same way?

    For the selected layer + head, we show the full pairwise cosine matrix $\cos(v_i, v_j)$ over real tokens. The number the paper reports is the **mean of the strict upper triangle** (white dots, $i<j$).

    A high number ⇒ all value vectors are pointing in similar directions: they carry redundant information.
    """)
    return


@app.cell
def _(F, head_slider, layer_slider, mo, np, plt, tokens, values):
    _layer = layer_slider.value
    _head = head_slider.value
    _v = values[_layer][0, _head]          # (S, D)
    _vn = F.normalize(_v, dim=-1)
    _cos = (_vn @ _vn.T).cpu().numpy()

    _upper = np.triu(np.ones_like(_cos, dtype=bool), k=1)
    _avg = _cos[_upper].mean() if _upper.any() else float("nan")

    _fig, _ax = plt.subplots(figsize=(7.2, 6.0))
    _im = _ax.imshow(_cos, cmap="coolwarm", vmin=-1.0, vmax=1.0, interpolation="nearest")

    _yy, _xx = np.where(_upper)
    _ax.scatter(_xx, _yy, s=6, color="white", alpha=0.4, linewidths=0)

    _ax.set_title(
        f"Pairwise cos(v_i, v_j): layer {_layer}, head {_head}\n"
        f"value_cos (upper-triangle mean) = {_avg:+.3f}",
        fontsize=11,
    )
    _ax.set_xticks(range(len(tokens)))
    _ax.set_yticks(range(len(tokens)))
    _ax.set_xticklabels(tokens, rotation=90, fontsize=7)
    _ax.set_yticklabels(tokens, fontsize=7)
    _ax.grid(False)
    _ax.set_facecolor("#ffffff")
    _fig.colorbar(_im, ax=_ax, fraction=0.04, pad=0.02, label="cosine")
    _fig.tight_layout()
    mo.hstack([mo.as_html(_fig)], justify="center")
    return


# ------------------------------------------------------------------------------
# §2.2: diag_attn
# ------------------------------------------------------------------------------

@app.cell
def _(mo):
    mo.md(r"""
    ### §2.2 `diag_attn`: how much does each token attend to itself?

    Below is the **actual attention matrix** from the model for your sentence at the selected layer + head. The dashed red line marks the diagonal. The paper reports the mean of the diagonal (position 0 excluded, since $a_{0,0}=1$ is forced by the causal mask).

    A high number ⇒ each token attends mostly to itself; attention is "exclusive."
    """)
    return


@app.cell
def _(attentions, head_slider, layer_slider, mo, np, plt, tokens):
    _layer = layer_slider.value
    _head = head_slider.value
    _attn = attentions[_layer][0, _head].cpu().numpy()   # (S, S)
    _diag = np.diag(_attn)
    _avg_diag = _diag[1:].mean() if len(_diag) > 1 else float("nan")

    _fig, _ax = plt.subplots(figsize=(7.2, 6.0))
    _im = _ax.imshow(
        _attn, cmap="magma", vmin=0.0, vmax=max(_attn.max(), 1e-9), interpolation="nearest"
    )
    _S = _attn.shape[0]
    _ax.plot([0, _S - 1], [0, _S - 1],
             color="#ff3355", linewidth=1.2, linestyle="--")
    _ax.set_title(
        f"Attention a[i,j]: layer {_layer}, head {_head}\n"
        f"diag_attn (mean of a_{{i,i}}, i>0) = {_avg_diag:.3f}",
        fontsize=11,
    )
    _ax.set_xlabel("Key position j")
    _ax.set_ylabel("Query position i")
    _ax.set_xticks(range(len(tokens)))
    _ax.set_yticks(range(len(tokens)))
    _ax.set_xticklabels(tokens, rotation=90, fontsize=7)
    _ax.set_yticklabels(tokens, fontsize=7)
    _ax.grid(False)
    _ax.set_facecolor("#ffffff")
    _fig.colorbar(_im, ax=_ax, fraction=0.04, pad=0.02, label="attention")
    _fig.tight_layout()
    mo.hstack([mo.as_html(_fig)], justify="center")
    return


# ------------------------------------------------------------------------------
# §2.3: output_cos
# ------------------------------------------------------------------------------

@app.cell
def _(mo):
    mo.md(r"""
    ### §2.3 `output_cos`: is the attention output like the token's own value?

    For each position $i$ we compute $y_i = \sum_j a_{i,j} v_j$ and measure $\cos(y_i, v_i)$. If both `value_cos` and `diag_attn` are high, $y_i$ must be close to $v_i$

    The paper reports the mean over positions.
    """)
    return


@app.cell
def _(F, attentions, go, head_slider, layer_slider, tokens, values):
    _layer = layer_slider.value
    _head = head_slider.value
    _v = values[_layer][0, _head]          # (S, D)
    _a = attentions[_layer][0, _head]      # (S, S)
    _y = _a @ _v

    _vn = F.normalize(_v, dim=-1)
    _yn = F.normalize(_y, dim=-1)
    _per_pos = (_yn * _vn).sum(dim=-1).cpu().numpy()
    _avg = float(_per_pos.mean())
    _x = list(range(len(tokens)))
    _min = float(_per_pos.min())
    _max = float(_per_pos.max())
    _all_non_negative = _min >= 0.0
    _y_min = 0.0 if _all_non_negative else max(-1.05, _min - 0.12)
    _y_max = min(1.08, max(0.25 if _all_non_negative else 0.1, _max + 0.16))
    _show_labels = len(tokens) <= 28

    _fig = go.Figure()
    _fig.add_bar(
        x=_x,
        y=_per_pos,
        marker=dict(
            color=_per_pos,
            colorscale=[[0.0, "#B93745"], [0.5, "#EEF3F8"], [1.0, "#239866"]],
            cmin=-1, cmax=1,
            line=dict(color="rgba(255,255,255,0.95)", width=0.8),
        ),
        text=[f"{v:+.2f}" for v in _per_pos] if _show_labels else None,
        textposition="outside" if _show_labels else "none",
        textfont=dict(size=9, color="#334155"),
        customdata=[[t] for t in tokens],
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "position %{x}<br>"
            "cos(y, v) = %{y:+.3f}<extra></extra>"
        ),
    )
    _fig.add_hline(
        y=_avg, line_dash="dash", line_color="#1f2937", line_width=1.5,
    )
    _fig.add_annotation(
        xref="paper",
        yref="y",
        x=0.01,
        y=_avg,
        text=f"mean = {_avg:+.3f}",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        yshift=6,
        font=dict(size=11, color="#0f172a"),
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="rgba(100,116,139,0.35)",
        borderwidth=1,
    )
    _fig.add_hline(
        y=0,
        line_color="#64748b" if _all_non_negative else "#94a3b8",
        line_width=1.0,
    )
    _fig.update_layout(
        title=dict(
            text=f"<b>cos(y_i, v_i) per token</b> :  layer {_layer}, head {_head}",
            font=dict(size=14, color="#1e293b"),
        ),
        xaxis=dict(
            tickmode="array",
            tickvals=_x,
            ticktext=tokens,
            tickangle=-45,
            tickfont=dict(size=10, color="#334155"),
            showgrid=False, zeroline=False,
            automargin=True,
        ),
        yaxis=dict(
            range=[_y_min, _y_max],
            title="cosine similarity",
            gridcolor="#dce6f2",
            griddash="dot",
            tickfont=dict(color="#334155"),
            zeroline=False,
        ),
        bargap=0.2,
        template="xsa_clean+plotly_white",
        height=420,
        margin=dict(l=60, r=28, t=62, b=92),
        plot_bgcolor="#fbfdff",
        paper_bgcolor="#ffffff",
        uniformtext_minsize=8,
        uniformtext_mode="hide",
        showlegend=False,
    )
    _fig
    return



# ==============================================================================
# §3: The paper's per-layer plot, on two small live models
# ==============================================================================

@app.cell
def _(mo):
    mo.md(r"""
    ## §3. Reimplementation: the paper's per-layer plot, live on your text

    Three metrics for **every layer**, averaged across all heads, on two models:

    - **GPT-2 (124M)**: learned APE, LayerNorm, vanilla MHA
    - **SmolLM2-1.7B**: full RoPE, RMSNorm, vanilla MHA 

    """)
    return


@app.cell
def _(F, torch):
    def attention_similarity_metrics(attn_w, v_h, y_h, causal: bool = True):
        """The paper's three metrics: averaged over batch, heads, positions.

        attn_w : (B, H, S, S) softmax attention
        v_h    : (B, H, S, D) per-head V
        y_h    : (B, H, S, D) per-head attention output (= attn_w @ v_h)
        """
        B, H, S, D = v_h.shape
        vn = F.normalize(v_h, dim=-1)
        yn = F.normalize(y_h, dim=-1)

        # value_cos: upper triangle of V V^T
        pair = vn @ vn.transpose(-1, -2)
        up = torch.triu(
            torch.ones(S, S, dtype=torch.bool, device=v_h.device), diagonal=1
        )
        value_cos = pair[..., up].mean().item() if up.any() else float("nan")

        # diag_attn: mean of a_{i,i}, i>0
        diag = attn_w.diagonal(dim1=-2, dim2=-1)
        if causal and S > 1:
            diag = diag[..., 1:]
        diag_attn = diag.mean().item()

        # output_cos: mean over positions of cos(y_i, v_i)
        output_cos = (yn * vn).sum(dim=-1).mean().item()

        return {
            "value_cos": value_cos,
            "diag_attn": diag_attn,
            "output_cos": output_cos,
        }
    return (attention_similarity_metrics,)


# ------------------------------------------------------------------------------
# §3A: GPT-2 per-layer sweep (reuses the cached forward from §2)
# ------------------------------------------------------------------------------

@app.cell
def _(G2_N_LAYERS, attention_similarity_metrics, attentions, values):
    """Per-layer paper metrics on GPT-2: uses the §2 forward pass cache."""
    g2_per_layer = {"value_cos": [], "diag_attn": [], "output_cos": []}
    for _li in range(G2_N_LAYERS):
        _aw = attentions[_li]
        _vh = values[_li]
        _yh = _aw @ _vh
        _m = attention_similarity_metrics(_aw, _vh, _yh, causal=True)
        for _k in g2_per_layer:
            g2_per_layer[_k].append(_m[_k])
    return (g2_per_layer,)


@app.cell
def _(G2_N_LAYERS, g2_per_layer, np, plt):
    """Plot: GPT-2: paper's three metrics vs layer index."""
    _layers = np.arange(G2_N_LAYERS)
    _fig, _axes = plt.subplots(1, 3, figsize=(11.8, 3.4), constrained_layout=True)
    _panels = [
        ("value_cos", r"Avg $\cos(v_i, v_j)$, $i<j$", "#1f77b4"),
        ("diag_attn", r"Avg $a_{i,i}$, $i>0$",        "#ff7f0e"),
        ("output_cos", r"Avg $\cos(y_i, v_i)$",       "#2ca02c"),
    ]
    for _ax, (_k, _t, _c) in zip(_axes, _panels):
        _vals = np.asarray(g2_per_layer[_k], dtype=float)
        _ax.plot(
            _layers, _vals, color=_c, linewidth=2.4,
            marker="o", markersize=4.8, markeredgecolor="white", markeredgewidth=0.9,
        )
        _ax.fill_between(_layers, _vals, _vals.min(), color=_c, alpha=0.12)
        _ax.set_title(_t, fontsize=11)
        _ax.set_xlabel("Layer index")
        _ax.grid(True, axis="y")
        _ax.grid(False, axis="x")
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
    _fig.suptitle("§3A: GPT-2",
                  fontsize=12, fontweight="bold")
    _fig
    return


# ------------------------------------------------------------------------------
# §3B: SmolLM2-1.7B (loaded lazily, vanilla MHA + RoPE)
# ------------------------------------------------------------------------------

@app.cell
def _(AutoModelForCausalLM, AutoTokenizer, DEVICE):
    _SMOL_ID = "HuggingFaceTB/SmolLM2-1.7B"
    tok_sm = AutoTokenizer.from_pretrained(_SMOL_ID)
    if tok_sm.pad_token_id is None:
        tok_sm.pad_token = tok_sm.eos_token
    model_sm = AutoModelForCausalLM.from_pretrained(
        _SMOL_ID, attn_implementation="eager"
    ).eval().to(DEVICE)
    _cfg = model_sm.config
    SM_N_LAYERS = _cfg.num_hidden_layers
    SM_N_HEADS = _cfg.num_attention_heads
    SM_N_KV = getattr(_cfg, "num_key_value_heads", SM_N_HEADS)
    SM_HEAD_DIM = getattr(_cfg, "head_dim", _cfg.hidden_size // SM_N_HEADS)
    return SM_HEAD_DIM, SM_N_HEADS, SM_N_KV, SM_N_LAYERS, model_sm, tok_sm


@app.cell
def _(
    DEVICE,
    SM_HEAD_DIM,
    SM_N_KV,
    SM_N_LAYERS,
    attention_similarity_metrics,
    get_active_text,
    model_sm,
    tok_sm,
    torch,
):
    """Per-layer paper metrics on SmolLM2-1.7B for the same committed text.

    SmolLM2-1.7B has a separate v_proj (Llama architecture). Output of v_proj
    is reshaped to (B, S, n_kv, head_dim); for vanilla MHA n_kv == n_heads
    so no head expansion is needed, but we keep the GQA-aware code for safety.
    """
    _ids = tok_sm(
        get_active_text(), return_tensors="pt", add_special_tokens=False
    ).input_ids.to(DEVICE)

    _v_cache_s = [None] * SM_N_LAYERS
    _hooks_s = []

    def _mk(_idx):
        def _h(_m, _inp, out):
            b, s, _ = out.shape
            _v_cache_s[_idx] = out.view(b, s, SM_N_KV, SM_HEAD_DIM).transpose(1, 2)
        return _h

    for _i, _blk in enumerate(model_sm.model.layers):
        _hooks_s.append(_blk.self_attn.v_proj.register_forward_hook(_mk(_i)))

    with torch.no_grad():
        _out = model_sm(
            input_ids=_ids,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    for _h in _hooks_s:
        _h.remove()

    sm_per_layer = {"value_cos": [], "diag_attn": [], "output_cos": []}
    for _li in range(SM_N_LAYERS):
        _aw = _out.attentions[_li]
        _vh = _v_cache_s[_li]
        _nq = _aw.shape[1]
        if _vh.shape[1] != _nq:
            _vh = _vh.repeat_interleave(_nq // _vh.shape[1], dim=1)
        _yh = _aw @ _vh
        _m = attention_similarity_metrics(_aw, _vh, _yh, causal=True)
        for _k in sm_per_layer:
            sm_per_layer[_k].append(_m[_k])
    return (sm_per_layer,)


@app.cell
def _(SM_N_LAYERS, np, plt, sm_per_layer):
    """Plot: SmolLM2-1.7B: paper's three metrics vs layer index."""
    _layers = np.arange(SM_N_LAYERS)
    _fig, _axes = plt.subplots(1, 3, figsize=(11.8, 3.4), constrained_layout=True)
    _panels = [
        ("value_cos", r"Avg $\cos(v_i, v_j)$, $i<j$", "#1f77b4"),
        ("diag_attn", r"Avg $a_{i,i}$, $i>0$",        "#ff7f0e"),
        ("output_cos", r"Avg $\cos(y_i, v_i)$",       "#2ca02c"),
    ]
    for _ax, (_k, _t, _c) in zip(_axes, _panels):
        _vals = np.asarray(sm_per_layer[_k], dtype=float)
        _ax.plot(
            _layers, _vals, color=_c, linewidth=2.2,
            marker="o", markersize=4.5, markeredgecolor="white", markeredgewidth=0.9,
        )
        _ax.fill_between(_layers, _vals, _vals.min(), color=_c, alpha=0.12)
        _ax.set_title(_t, fontsize=11)
        _ax.set_xlabel("Layer index")
        _ax.grid(True, axis="y")
        _ax.grid(False, axis="x")
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
    _fig.suptitle(
        "§3B: SmolLM2-1.7B ",
        fontsize=12, fontweight="bold"
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    Both models show the existence of high attention-similarity bias, however the trends across layers vary.
          
    - Original GPT-2 shows **metrics decreasing** with depth.
    - SmolLM2-1.7B is closer to monotone increase but still noisy / non-monotone in places 
    
    (which makes sense because SmolLM2 follows a similar architecture to the paper's experimental setup of MHA + RoPE while GPT2 has absolute positional embeddings)

    The bias exists, but its direction and magnitude seem to be architecture-conditional.
    """)
    return


# ==============================================================================
# §3C: XSA: the paper's proposed fix, in 2D vector form
# ==============================================================================

@app.cell
def _(mo):
    mo.md(r"""
    ## §3C. The XSA fix, in 2 lines of PyTorch

    XSA adds one Gram-Schmidt step after standard attention:

    ```python
    v_hat = F.normalize(values, dim=-1)
    z = y - (y * v_hat).sum(-1, keepdim=True) * v_hat  # cos(z, v) = 0 by construction
    ```

    The plot below uses real GPT-2 vectors from your sentence in §2, rendered in the 2-D plane spanned by $v_i$ and $y_i^{\perp v_i}$ 
    """)
    return


@app.cell
def _(S, mo):
    token_slider = mo.ui.slider(
        start=0, stop=max(0, S - 1), step=1, value=min(3, max(0, S - 1)),
        label="Token index  i",
        show_value=True,
    )
    token_slider
    return (token_slider,)


@app.cell
def _(
    attentions, head_slider, keys, layer_slider, mo, np, plt,
    queries, token_slider, tokens, values,
):
    _layer = layer_slider.value
    _head = head_slider.value
    _i = min(token_slider.value, len(tokens) - 1)

    _v_all = values[_layer][0, _head]
    _q_all = queries[_layer][0, _head]
    _k_all = keys[_layer][0, _head]
    _a = attentions[_layer][0, _head]
    _y_all = _a @ _v_all

    v_raw = _v_all[_i].cpu().numpy()
    q_raw = _q_all[_i].cpu().numpy()
    k_raw = _k_all[_i].cpu().numpy()
    y_raw = _y_all[_i].cpu().numpy()

    # XSA: z = y - (y · v_hat) v_hat
    _v_norm = np.linalg.norm(v_raw) + 1e-12
    _v_hat = v_raw / _v_norm
    z_raw = y_raw - np.dot(y_raw, _v_hat) * _v_hat

    cos_y_v = float(np.dot(y_raw, v_raw) / (np.linalg.norm(y_raw) * np.linalg.norm(v_raw) + 1e-12))
    cos_z_v = float(np.dot(z_raw, v_raw) / (np.linalg.norm(z_raw) * np.linalg.norm(v_raw) + 1e-12))

    # GPT-2 BPE: · = leading space, ↵ = newline (we already substituted these in §2)
    _raw_tok = tokens[_i]
    _clean = _raw_tok.replace("·", " ").replace("↵", "\n").strip()
    tok_label = f'"{_clean}"' if _clean else '"<space>"'

    def _norm(x):
        return x / (np.linalg.norm(x) + 1e-8)

    e1 = _norm(v_raw)
    y_perp = y_raw - np.dot(y_raw, e1) * e1
    e2 = _norm(y_perp) if np.linalg.norm(y_perp) > 1e-8 else _norm(np.roll(e1, 1))

    def _to2d(x):
        return np.array([np.dot(x, e1), np.dot(x, e2)])

    q2 = _to2d(q_raw)
    k2 = _to2d(k_raw)
    v2 = _to2d(v_raw)
    y2 = _to2d(y_raw)
    z2 = _to2d(z_raw)
    proj2 = np.array([y2[0], 0.0])

    # Equal-length display (not to scale): keep direction, encode magnitudes in labels.
    _raw_vecs = {"q": q2, "k": k2, "v": v2, "y": y2, "proj": proj2, "z": z2}
    _mags = {name: float(np.linalg.norm(vec)) for name, vec in _raw_vecs.items()}
    _L = 1.0
    _disp = {
        name: (vec / (_mags[name] + 1e-12) * _L) if _mags[name] > 1e-8 else vec
        for name, vec in _raw_vecs.items()
    }
    q2, k2, v2, y2, proj2, z2 = (
        _disp["q"], _disp["k"], _disp["v"], _disp["y"], _disp["proj"], _disp["z"]
    )
    xmin, xmax = -1.28, 1.28
    ymin, ymax = -1.28, 1.28
    view = max(xmax - xmin, ymax - ymin)

    def _place_label(
        ax, vec, lbl, color, fontsize=11,
        radial_pad=0.05, tangential_shift=0.0,
    ):
        nrm = np.linalg.norm(vec)
        if nrm < 1e-6:
            return
        angle = np.arctan2(vec[1], vec[0])
        ux, uy = np.cos(angle), np.sin(angle)
        px, py = -uy, ux  # unit normal (counter-clockwise)
        pad = radial_pad * view
        nudge = tangential_shift * view
        tx = vec[0] + pad * ux + nudge * px
        ty = vec[1] + pad * uy + nudge * py
        tx = np.clip(tx, xmin + 0.04 * view, xmax - 0.04 * view)
        ty = np.clip(ty, ymin + 0.04 * view, ymax - 0.04 * view)
        ha = "left" if ux >= 0 else "right"
        va = "bottom" if uy >= 0 else "top"
        ax.text(tx, ty, lbl, color=color, fontsize=fontsize, fontweight="bold",
                ha=ha, va=va,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, ec="none"))

    fig_geom, ax_geom = plt.subplots(figsize=(6.6, 6.6))
    fig_geom.patch.set_facecolor("#ffffff")
    ax_geom.set_facecolor("#f7f7fb")
    orig = np.zeros(2)

    _arrow_cfg = [
        (q2,    "#8ecae6", f"$q_i$\n|q|={_mags['q']:.3f}",                 1.8, ":", 0.05, 0.0),
        (k2,    "#90be6d", f"$k_i$\n|k|={_mags['k']:.3f}",                 1.8, ":", 0.05, 0.0),
        (v2,    "#f77f00", f"$v_i$\n|v|={_mags['v']:.3f}",                 3.2, "-", 0.05, 0.035),
        (y2,    "#e63946", f"$y_i$ (SA)\n|y|={_mags['y']:.3f}",            2.4, "-", 0.05, 0.0),
        (proj2, "#ffbe0b", f"$\\mathrm{{proj}}_v y_i$\n|proj|={_mags['proj']:.3f}", 2.2, "--", 0.05, -0.035),
        (z2,    "#2ec4b6", f"$z_i$ (XSA)\n|z|={_mags['z']:.3f}",           2.4, "-", 0.05, 0.0),
    ]
    for vec, col, lbl, lw, ls, rpad, tshift in _arrow_cfg:
        if np.linalg.norm(vec) < 1e-6:
            continue
        ax_geom.annotate("", xy=vec, xytext=orig,
                         arrowprops=dict(arrowstyle="->", color=col, lw=lw, linestyle=ls))
        _place_label(ax_geom, vec, lbl, col, radial_pad=rpad, tangential_shift=tshift)

    # right-angle marker at the foot of the projection
    if np.linalg.norm(z2) > 0.01:
        sc = 0.035 * view
        pz = z2 / np.linalg.norm(z2) * sc
        _sx = np.sign(proj2[0]) if proj2[0] != 0 else 1.0
        px = np.array([_sx * sc, 0.0])
        sq = np.array([proj2 + pz, proj2 + pz + px, proj2 + px, proj2 + pz])
        ax_geom.plot(sq[:, 0], sq[:, 1], color="#666", lw=1, alpha=0.5)

    ax_geom.set_xlim(xmin, xmax); ax_geom.set_ylim(ymin, ymax)
    ax_geom.axhline(0, color="#aaa", lw=0.7); ax_geom.axvline(0, color="#aaa", lw=0.7)
    ax_geom.set_xlabel(r"Component along $v_i$  (self-copy axis)")
    ax_geom.set_ylabel(r"Orthogonal component  (contextual axis)")
    ax_geom.set_title(
        f"Token {_i}: {tok_label}  |  layer {_layer}, head {_head}\n"
        f"cos(y, v) = {cos_y_v:.3f}   →   cos(z, v) = {cos_z_v:.4f}",
        fontsize=11,
    )
    ax_geom.text(
        0.99, 0.99,
        "Not to scale: all vectors are drawn with equal length for readability.\n"
        "Use labels for true magnitudes.",
        transform=ax_geom.transAxes,
        ha="right", va="top", fontsize=8.5, color="#475569",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cbd5e1", alpha=0.92),
    )
    ax_geom.set_aspect("equal")
    ax_geom.grid(False)
    for sp in ax_geom.spines.values():
        sp.set_edgecolor("#d0d0dd")

    geom_view = mo.vstack([
        mo.hstack([
            mo.stat(f"{cos_y_v:.3f}", label="cos(y_i, v_i): SA",  bordered=True),
            mo.stat(f"{cos_z_v:.4f}", label="cos(z_i, v_i): XSA", bordered=True),
            mo.stat(
                f"{cos_y_v - cos_z_v:.3f}",
                label="Bias removed",
                bordered=True,
            ),
        ], justify="center"),
        mo.hstack([mo.as_html(fig_geom)], justify="center"),
    ])
    geom_view
    return


@app.cell
def _(mo):
    mo.md(r"""
    - $v_i$ (orange) is always horizontal by basis construction.
    - $z_i$ (teal) is always vertical and orthogonal to $v_i$ by construction.
    - The yellow projection shows how much of $y_i$ was self-copy; XSA returns only the contextual residual.
    """)
    return


# ==============================================================================
# §4: Cross-architecture survey (13 pretrained models, precomputed plots)
# ==============================================================================

@app.cell
def _(mo):
    mo.md(r"""
    ## §4. Cross-architecture survey: 13 pretrained models

    Three metrics on 64 random 256-token windows from a literary corpus for 11 language models, plus ViT and DiT runs on patch/token-like sequences (same metric definitions), spanning 2019 (GPT-2 / BERT / ViT / DiT) → 2025 (Llama-3.2, Qwen3, SmolLM2).

    <div style="display:flex; justify-content:center;">
      <table style="margin:0 auto; text-align:center;">
        <thead>
          <tr><th>Axis</th><th>Values represented</th></tr>
        </thead>
        <tbody>
          <tr><td>Position embedding</td><td>learned APE, partial RoPE, full RoPE, mRoPE</td></tr>
          <tr><td>Attention type</td><td>MHA causal, MHA bidirectional, MQA, GQA</td></tr>
          <tr><td>Normalization</td><td>LayerNorm, RMSNorm, RMSNorm + QK-Norm</td></tr>
          <tr><td>Scale</td><td>ViT-scale to 2.7B</td></tr>
        </tbody>
      </table>
    </div>
    """)
    return


@app.cell
def _(mo):
    _models = [
        ("GPT-2: learned APE, MHA, LayerNorm",                "gpt2_paper_metrics.png"),
        ("BERT-base: learned APE, MHA bidirectional, post-LN", "bert_paper_metrics.png"),
        ("ViT: learned APE, bidirectional MHA, LayerNorm",    "vit_paper_metrics.png"),
        ("DiT: diffusion transformer block",                  "dit_paper_metrics.png"),
        ("Phi-1: partial RoPE 50%, MHA, LayerNorm",           "phi_paper_metrics.png"),
        ("Phi-2: partial RoPE 40%, MHA, LayerNorm",           "phi2_paper_metrics.png"),
        ("Pythia-1.4B: partial RoPE 25%, MHA, LayerNorm",     "pythia_paper_metrics.png"),
        ("Gemma-2B: full RoPE, MQA 8:1, RMSNorm",             "gemma2b_paper_metrics.png"),
        ("Qwen2-VL-2B LM: mRoPE, GQA 12:2, RMSNorm",          "qwen2_vl_lm_paper_metrics.png"),
        ("Qwen3-0.6B: full RoPE, GQA 16:8, RMSNorm + QK-Norm", "qwen_paper_metrics.png"),
        ("Llama-3.2-1B: full RoPE llama3, GQA 32:8, RMSNorm", "llama3_2_1b_paper_metrics.png"),
        ("Qwen2.5-0.5B: full RoPE, GQA 14:2, RMSNorm",        "qwen2_5_0_5b_paper_metrics.png"),
        ("SmolLM2-360M: full RoPE, GQA 15:5, RMSNorm",        "smollm2_360m_paper_metrics.png"),
    ]
    model_picker = mo.ui.dropdown(
        options={label: fname for label, fname in _models},
        value="GPT-2: learned APE, MHA, LayerNorm",
        label="Model",
    )
    show_all = mo.ui.switch(value=False, label="Show all 13 at once")
    mo.hstack([model_picker, show_all], justify="start", gap=2)
    return model_picker, show_all


@app.cell
def _(mo, model_picker, os, show_all):
    import subprocess

    _all = [
        "gpt2_paper_metrics.png", "bert_paper_metrics.png",
        "vit_paper_metrics.png", "dit_paper_metrics.png",
        "phi_paper_metrics.png", "phi2_paper_metrics.png",
        "pythia_paper_metrics.png", "gemma2b_paper_metrics.png",
        "qwen2_vl_lm_paper_metrics.png", "qwen_paper_metrics.png",
        "llama3_2_1b_paper_metrics.png", "qwen2_5_0_5b_paper_metrics.png",
        "smollm2_360m_paper_metrics.png",
    ]
    _github_raw_base = "https://raw.githubusercontent.com/grasgor/xsa-marimo/main/outputs"

    def _existing_count(_dir, _names):
        if not os.path.isdir(_dir):
            return 0
        return sum(1 for _n in _names if os.path.isfile(os.path.join(_dir, _n)))

    def _download_missing(_target_dir, _missing):
        os.makedirs(_target_dir, exist_ok=True)
        downloaded, failed = [], []
        for _name in _missing:
            _dst = os.path.join(_target_dir, _name)
            _url = f"{_github_raw_base}/{_name}"
            try:
                _proc = subprocess.run(
                    ["curl", "-LfsS", "--max-time", "20", "-o", _dst, _url],
                    capture_output=True, text=True, check=False,
                )
            except Exception:
                failed.append(_name)
                continue
            if _proc.returncode == 0 and os.path.isfile(_dst) and os.path.getsize(_dst) > 0:
                downloaded.append(_name)
            else:
                try:
                    if os.path.isfile(_dst):
                        os.remove(_dst)
                except OSError:
                    pass
                failed.append(_name)
        return downloaded, failed

    _candidates = []
    if "__file__" in globals():
        _here = os.path.dirname(os.path.abspath(__file__))
        _candidates.append(os.path.join(_here, "outputs"))
    _candidates.extend([
        os.path.join(".", "outputs"),
        os.path.join(".", "paper_metric_plot", "outputs"),
    ])

    _plot_dir = _candidates[0]
    _best = -1
    for _cand in _candidates:
        _cnt = _existing_count(_cand, _all)
        if _cnt > _best:
            _best = _cnt
            _plot_dir = _cand

    _boot_msg = None
    _missing_any = [_n for _n in _all if not os.path.isfile(os.path.join(_plot_dir, _n))]
    if _missing_any:
        _target_dir = os.path.join(".", "outputs")
        _downloaded, _failed = _download_missing(_target_dir, _missing_any)
        if _existing_count(_target_dir, _all) >= _existing_count(_plot_dir, _all):
            _plot_dir = _target_dir
        if _downloaded:
            _boot_msg = mo.callout(
                mo.md(
                    f"Fetched `{len(_downloaded)}` missing plot image(s) from GitHub into "
                    f"`{_target_dir}`."
                ),
                kind="success",
            )
        elif _failed:
            _boot_msg = mo.callout(
                mo.md(
                    "Could not auto-download missing plot images from GitHub "
                    "(network may be unavailable in this runtime)."
                ),
                kind="warn",
            )

    if show_all.value:
        _items = []
        for _name in _all:
            _path = os.path.join(_plot_dir, _name)
            if os.path.isfile(_path):
                _items.append(mo.image(src=_path, width="100%"))
            else:
                _items.append(mo.md(f"*missing: {_name}*"))
        _main = mo.vstack(_items, gap=0.2)
    else:
        _path = os.path.join(_plot_dir, model_picker.value)
        if os.path.isfile(_path):
            _main = mo.image(src=_path, width="100%")
        else:
            _main = mo.md(
                f"**Missing image**: `{_path}`.\n\n"
                "Expected `outputs/*.png` in the working directory, or reachable GitHub raw URLs."
            )
    gallery = mo.vstack([_boot_msg, _main]) if _boot_msg is not None else _main
    gallery
    return (gallery,)


@app.cell
def _(mo):
    mo.md(r"""
    ### What the 13-model survey tells us

    - **Textbook paper match**: BERT shows clean monotone rise in all three metrics. Phi-2 is close.
    - **Vision models also show the bias**: both ViT and DiT show attention-similarity behavior under the same metrics.
    - **Opposite of the paper**: GPT-2 shows all three metrics *decreasing* with depth. Qwen2.5-0.5B has inverted `value_cos` (highest at layer 1, lowest at layer 22).
    - **Noisy / non-monotone**: Llama-3.2-1B, SmolLM2-360M, Qwen3-0.6B, Qwen2-VL-2B LM, Phi-1.
    - **Extreme magnitude**: Gemma-2B (MQA 8:1) has the highest `diag_attn` and `output_cos` values across the set: shared KV heads appear to amplify the effect.
    - **Universal**: every model except GPT-2 shows a **final-layer `diag_attn` spike**. This is the single most reproducible signal.

    The paper's claim holds strongly for a specific slice of this matrix (bidirectional MHA with learned APE, or mid-stack decoder MHA with partial RoPE) and breaks down elsewhere. The ViT + DiT signal suggests this is broader than language-only architectures; testing XSA directly in those vision stacks is interesting future work, but out of scope for this CPU-only notebook.
    """)
    return


# ==============================================================================
# §5: Stress tests (GPT-2, fast)
# ==============================================================================

@app.cell
def _(mo):
    mo.md(r"""
    ## §5. Stress tests

    If the bias is architectural the curve should be stable across inputs; if it's input-dependent the curve should shift. GPT-2 on three corpora: natural English, Python code, and a random-permuted token stream.
    """)
    return


@app.cell
def _(mo):
    stress_toggle = mo.ui.multiselect(
        options=["natural English", "Python code", "random-permuted tokens"],
        value=["natural English", "Python code", "random-permuted tokens"],
        label="Corpora to overlay",
    )
    stress_toggle
    return (stress_toggle,)


@app.cell
def _(
    DEVICE,
    G2_HEAD_DIM,
    G2_HIDDEN,
    G2_N_HEADS,
    G2_N_LAYERS,
    attention_similarity_metrics,
    model_g2,
    np,
    tok_g2,
    torch,
):
    """Compute per-layer metrics on three different inputs through GPT-2."""
    _samples = {
        "natural English": (
            "Alice was beginning to get very tired of sitting by her sister on the bank, "
            "and of having nothing to do. Once or twice she had peeped into the book her "
            "sister was reading, but it had no pictures or conversations in it, "
            "'and what is the use of a book,' thought Alice, 'without pictures or conversations?'"
        ),
        "Python code": (
            "def softmax(x, dim=-1):\n"
            "    x = x - x.max(dim=dim, keepdim=True).values\n"
            "    e = torch.exp(x)\n"
            "    return e / e.sum(dim=dim, keepdim=True)\n\n"
            "class Attention(nn.Module):\n"
            "    def __init__(self, dim, heads):\n"
            "        super().__init__()\n"
            "        self.qkv = nn.Linear(dim, 3 * dim, bias=False)\n"
            "        self.out = nn.Linear(dim, dim, bias=False)\n"
            "        self.heads = heads\n"
        ),
    }

    def _compute_for_ids(_ids):
        _v_cache = [None] * G2_N_LAYERS
        _hs = []
        def _mk(_idx):
            def _h(_m, _inp, out):
                b, s, _ = out.shape
                _v = out[..., 2 * G2_HIDDEN:]
                _v_cache[_idx] = _v.view(b, s, G2_N_HEADS, G2_HEAD_DIM).transpose(1, 2)
            return _h
        for _i, _blk in enumerate(model_g2.transformer.h):
            _hs.append(_blk.attn.c_attn.register_forward_hook(_mk(_i)))
        with torch.no_grad():
            _o = model_g2(input_ids=_ids, output_attentions=True,
                          use_cache=False, return_dict=True)
        for _h in _hs: _h.remove()
        _per = {"value_cos": [], "diag_attn": [], "output_cos": []}
        for _li in range(G2_N_LAYERS):
            _aw = _o.attentions[_li]
            _vh = _v_cache[_li]
            _yh = _aw @ _vh
            _m = attention_similarity_metrics(_aw, _vh, _yh, causal=True)
            for _k in _per: _per[_k].append(_m[_k])
        return _per

    _rng = np.random.default_rng(2026)
    _english_ids = tok_g2(_samples["natural English"], return_tensors="pt",
                          add_special_tokens=False).input_ids.to(DEVICE)
    _code_ids = tok_g2(_samples["Python code"], return_tensors="pt",
                       add_special_tokens=False).input_ids.to(DEVICE)
    _shuf = _english_ids[0].cpu().numpy().copy()
    _rng.shuffle(_shuf)
    _rand_ids = torch.tensor(_shuf, dtype=torch.long, device=DEVICE).unsqueeze(0)

    stress_results = {
        "natural English":         _compute_for_ids(_english_ids),
        "Python code":             _compute_for_ids(_code_ids),
        "random-permuted tokens":  _compute_for_ids(_rand_ids),
    }
    return (stress_results,)


@app.cell
def _(G2_N_LAYERS, np, plt, stress_results, stress_toggle):
    _layers = np.arange(G2_N_LAYERS)
    _fig, _axes = plt.subplots(1, 3, figsize=(12.2, 3.6), constrained_layout=True)
    _panels = [
        ("value_cos", r"Avg $\cos(v_i, v_j)$, $i<j$"),
        ("diag_attn", r"Avg $a_{i,i}$, $i>0$"),
        ("output_cos", r"Avg $\cos(y_i, v_i)$"),
    ]
    _colors = {
        "natural English":         "#1f77b4",
        "Python code":             "#ff7f0e",
        "random-permuted tokens":  "#888888",
    }
    for _ax, (_k, _t) in zip(_axes, _panels):
        for _name in stress_toggle.value:
            _vals = stress_results[_name][_k]
            _ax.plot(_layers, _vals, label=_name,
                     color=_colors.get(_name, "black"),
                     linewidth=2.2, marker="o", markersize=4.2,
                     markeredgecolor="white", markeredgewidth=0.8)
        _ax.set_title(_t, fontsize=11)
        _ax.set_xlabel("Layer index")
        _ax.grid(True, axis="y")
        _ax.grid(False, axis="x")
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
    _axes[0].legend(fontsize=8, loc="best", frameon=False)
    _fig.suptitle("GPT-2: metric vs corpus", fontsize=12, fontweight="bold")
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Attention-sink isolation

    One plausible confound: position 0 (and positions 1 - 3) tend to be "attention sinks": many heads attend there regardless of content (Xiao et al. 2024). If we compute the metrics while **dropping the first $k$ positions**, does the trend change?

    Implementation detail (strict test): we drop those first $k$ positions from both query and key axes, renormalize each remaining attention row, then recompute $y_i$, `diag_attn`, and `output_cos`.

    A large change ⇒ the signal is dominated by sink tokens, not a body-wide phenomenon. A small change ⇒ the bias is real across the sequence.
    """)
    return


@app.cell
def _(S, mo):
    drop_k = mo.ui.slider(
        start=0, stop=min(8, max(1, S - 2)), step=1, value=0,
        label="Positions to drop from start",
        show_value=True,
    )
    drop_k
    return (drop_k,)


@app.cell
def _(F, G2_N_LAYERS, attentions, drop_k, mo, np, plt, torch, values):
    """Recompute the three metrics on the user's text with first-k positions dropped.
    Reuses the §2 cached forward pass on GPT-2.
    """
    _k = drop_k.value

    def _metric_with_skip(_attn, _v, skip):
        # Strict sink isolation: drop first-k query/key positions and renormalize rows.
        _attn_sub = _attn[..., skip:, skip:]
        _row_sum = _attn_sub.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        _attn_sub = _attn_sub / _row_sum
        _row_err = (_attn_sub.sum(dim=-1) - 1.0).abs().max().item()
        _v_sub = _v[..., skip:, :]
        _vn = F.normalize(_v_sub, dim=-1)
        _y_sub = _attn_sub @ _v_sub
        _yn = F.normalize(_y_sub, dim=-1)
        _pair = _vn @ _vn.transpose(-1, -2)
        _S2 = _pair.shape[-1]
        _up = torch.triu(
            torch.ones(_S2, _S2, dtype=torch.bool, device=_v.device), diagonal=1
        )
        _vc = _pair[..., _up].mean().item() if _up.any() else float("nan")
        _diag = _attn_sub.diagonal(dim1=-2, dim2=-1)
        _da = _diag[..., 1:].mean().item() if _diag.shape[-1] > 1 else float("nan")
        _oc = (_yn * _vn).sum(dim=-1).mean().item()
        return _vc, _da, _oc, float(_row_err)

    _vals = {"value_cos": [], "diag_attn": [], "output_cos": []}
    _vals0 = {"value_cos": [], "diag_attn": [], "output_cos": []}
    _row_err0 = []
    _row_errk = []
    for _li in range(G2_N_LAYERS):
        _aw = attentions[_li]
        _vh = values[_li]
        _vc0, _da0, _oc0, _re0 = _metric_with_skip(_aw, _vh, skip=0)
        _vc, _da, _oc, _rek = _metric_with_skip(_aw, _vh, skip=_k)
        _vals0["value_cos"].append(_vc0)
        _vals0["diag_attn"].append(_da0)
        _vals0["output_cos"].append(_oc0)
        _vals["value_cos"].append(_vc)
        _vals["diag_attn"].append(_da)
        _vals["output_cos"].append(_oc)
        _row_err0.append(_re0)
        _row_errk.append(_rek)
    _arr0 = {k: np.asarray(v, dtype=float) for k, v in _vals0.items()}
    _arrk = {k: np.asarray(v, dtype=float) for k, v in _vals.items()}
    _delta = {k: (_arrk[k] - _arr0[k]) for k in _arr0}
    _mean0 = {k: float(np.nanmean(_arr0[k])) for k in _arr0}
    _meank = {k: float(np.nanmean(_arrk[k])) for k in _arrk}
    _mean_abs_delta = {
        k: float(np.nanmean(np.abs(_delta[k]))) for k in _delta
    }
    _mean_rel_delta = {
        k: 100.0 * _mean_abs_delta[k] / (abs(_mean0[k]) + 1e-8) for k in _delta
    }
    _max_rel = max(_mean_rel_delta.values()) if _mean_rel_delta else float("nan")
    _max_abs = max(_mean_abs_delta.values()) if _mean_abs_delta else float("nan")
    _row_err_max = max(_row_errk) if _row_errk else float("nan")
    _k0_consistency = _max_abs if _k == 0 else float("nan")
    _oc_peak_layer = int(np.nanargmax(np.abs(_delta["output_cos"])))
    _oc_peak = float(_delta["output_cos"][_oc_peak_layer])
    if _k == 0:
        _sink_kind = "neutral"
        _sink_msg = (
            "Move `k` above 0 to measure sink impact. The panel compares the full "
            "sequence (`drop 0`) against metrics after dropping early positions."
        )
    elif _max_rel < 5.0 and _max_abs < 0.02:
        _sink_kind = "success"
        _sink_msg = (
            f"Sink impact is **small** at `k={_k}`: curves staying similar is expected. "
            "Interpretation: this bias signal is not primarily driven by first-token sinks "
            "for this input; it persists across the body of the sequence."
        )
    elif _max_rel < 15.0:
        _sink_kind = "warn"
        _sink_msg = (
            f"Sink impact is **moderate** at `k={_k}`: early tokens contribute, but they "
            "do not fully dominate the trend. Treat sink effects as a partial confound."
        )
    else:
        _sink_kind = "warn"
        _sink_msg = (
            f"Sink impact is **large** at `k={_k}`: early sink positions strongly shape "
            "the measured curves. Be careful about generalizing without sink controls."
        )
    _sink_rows = "\n".join(
        (
            f"| `{m}` | {_mean0[m]:+.4f} | {_meank[m]:+.4f} | "
            f"{_mean_abs_delta[m]:.4f} | {_mean_rel_delta[m]:.1f}% |"
        )
        for m in ["value_cos", "diag_attn", "output_cos"]
    )
    _sink_table = (
        "| Metric | mean(drop 0) | mean(drop k) | mean abs Δ | relative abs Δ |\n"
        "|---|---:|---:|---:|---:|\n"
        + _sink_rows
    )

    _layers = np.arange(G2_N_LAYERS)
    _fig, _axes = plt.subplots(1, 3, figsize=(12.2, 3.6), constrained_layout=True)
    _panels = [
        ("value_cos", r"Avg $\cos(v_i, v_j)$, $i<j$"),
        ("diag_attn", r"Avg $a_{i,i}$"),
        ("output_cos", r"Avg $\cos(y_i, v_i)$"),
    ]
    for _ax, (_m, _t) in zip(_axes, _panels):
        _ax.plot(_layers, _vals0[_m], label="drop 0",
                 color="#7c8798", linewidth=2.0, marker="o", markersize=4,
                 markeredgecolor="white", markeredgewidth=0.8, linestyle="--")
        _ax.plot(_layers, _vals[_m], label=f"drop {_k}",
                 color="#D64550", linewidth=2.2, marker="o", markersize=4.8,
                 markeredgecolor="white", markeredgewidth=0.8)
        _ax.set_title(_t, fontsize=11)
        _ax.set_xlabel("Layer index")
        _ax.grid(True, axis="y")
        _ax.grid(False, axis="x")
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
    _axes[0].legend(fontsize=9, frameon=False)
    _fig.suptitle(
        f"GPT-2: attention-sink isolation: drop first {_k} positions",
        fontsize=12, fontweight="bold",
    )
    mo.vstack([
        mo.as_html(_fig),
        mo.hstack([
            mo.stat(f"{_max_rel:.1f}%", label="max mean relative shift", bordered=True),
            mo.stat(f"{_max_abs:.4f}", label="max mean absolute shift", bordered=True),
            mo.stat(f"{_row_err_max:.2e}", label="row-renorm max error", bordered=True),
            mo.stat(
                f"layer {_oc_peak_layer}: {_oc_peak:+.4f}",
                label="largest Δ output_cos layer",
                bordered=True,
            ),
        ], justify="start"),
        mo.md("#### Sink diagnostics\n\n" + _sink_table),
        mo.callout(
            mo.md(
                (
                    f"Sanity check (`k=0`): baseline consistency max |Δ| = `{_k0_consistency:.2e}`."
                    if _k == 0 else
                    f"Sanity check: row re-normalization max error = `{_row_err_max:.2e}`."
                )
            ),
            kind="neutral",
        ),
        mo.callout(mo.md(_sink_msg), kind=_sink_kind),
    ])
    return


# ==============================================================================
# §6: Learnable α-XSA: does the model want α = 1?
# ==============================================================================

@app.cell
def _(mo):
    mo.md(r"""
    ## §6. Learnable α-XSA: does the model actually want α = 1?

    The paper's XSA update is equivalent to setting $\alpha = 1$ in
    $$ z_i \;=\; y_i \;-\; \alpha\,\frac{y_i \cdot v_i}{\lVert v_i \rVert^2}\,v_i. $$

    Is that the *right* value, or just a convenient choice? We give each attention head its **own** $\alpha \in (0,1)$ (sigmoid-bounded) and let backprop decide. If the model converges to $\alpha \approx 1$, that's **loss-based evidence in this tiny setup** that strong orthogonalization is preferred.
    """)
    return


@app.cell
def _(mo):
    learned_alpha_state, set_learned_alpha_state = mo.state(1.0)
    learned_alpha_result_state, set_learned_alpha_result_state = mo.state(None)
    return (
        learned_alpha_result_state,
        learned_alpha_state,
        set_learned_alpha_result_state,
        set_learned_alpha_state,
    )


@app.cell
def _(learned_alpha_state, mo):
    learn_btn = mo.ui.run_button(
        label="🧪 Learn optimal α  (trains tiny LM ~30 s)", kind="success",
    )
    alpha_slider = mo.ui.slider(
        0.0, 1.0, step=0.01,
        value=round(learned_alpha_state(), 2),
        label="α (exclusion strength)",
        show_value=True,
    )
    alpha_dim = mo.ui.slider(16, 256, value=64, step=16,
                             label="Head dim (sweep)", show_value=True)
    alpha_seed = mo.ui.slider(0, 99, value=42, label="Seed (sweep)",
                              show_value=True)
    mo.vstack([
        mo.hstack([learn_btn, alpha_slider]),
        mo.hstack([alpha_dim, alpha_seed]),
    ])
    return alpha_dim, alpha_seed, alpha_slider, learn_btn


@app.cell
def _(
    learn_btn,
    learned_alpha_result_state,
    mo,
    set_learned_alpha_result_state,
    set_learned_alpha_state,
    time,
    train_learnable_alpha,
):
    if not learn_btn.value:
        learned_result = learned_alpha_result_state()
        if learned_result is None:
            learn_out = mo.callout(mo.md(
                "Click **Learn optimal α** to train a tiny LM where each attention "
                "head learns its own $\\alpha$ via backprop. The result updates the "
                "slider default. Until you run this, learned-α stats stay as `—`."
            ), kind="info")
        else:
            learn_out = mo.vstack([
                mo.callout(mo.md(
                    "Using the most recent learned-α run. Click **Learn optimal α** "
                    "again to retrain from scratch."
                ), kind="neutral"),
                mo.md(
                    f"Last learned mean α: `{learned_result['mean']:.3f}`"
                ),
            ])
    else:
        _TOTAL_STEPS = 300

        def _on_step_alpha(_name, step, total, loss):
            if step % 25 == 0 or step == total - 1:
                _done = step + 1
                _pct = _done / total
                _bar = "█" * int(_pct * 30) + "░" * (30 - int(_pct * 30))
                mo.output.replace(mo.vstack([
                    mo.md(f"### Training learnable-α LM on mixed web text…"),
                    mo.md(
                        f"`[{_bar}]` **{_done}/{total}** steps; "
                        f"loss: `{loss:.4f}`"
                    ),
                ]))

        _t0 = time.time()
        learned_result = train_learnable_alpha(
            steps=_TOTAL_STEPS, seq_len=64, batch_size=16, model_dim=128, num_layers=4,
            on_step=_on_step_alpha,
        )
        set_learned_alpha_result_state(learned_result)
        set_learned_alpha_state(learned_result["mean"])
        _sum = learned_result["summary"]
        _rows = "\n".join(
            f"| Layer {li} | {v:.3f} |"
            for li, v in learned_result["per_layer"].items()
        )
        _ph = learned_result.get("per_layer_head", {})
        _nhead = len(next(iter(_ph.values()))) if _ph else 0
        _head_header = " | ".join(f"H{h}" for h in range(_nhead))
        _head_sep = " | ".join(["---"] * _nhead)
        _head_rows = "\n".join(
            f"| Layer {li} | " + " | ".join(f"{a:.3f}" for a in _alphas) + " |"
            for li, _alphas in _ph.items()
        )
        learn_out = mo.vstack([
            mo.md(f"Training done in `{time.time() - _t0:.1f}s`"),
            mo.md(
                f"**Mean learned α across all layers/heads: `{learned_result['mean']:.3f}`**\n\n"
                "| Layer | Mean α |\n|---|---|\n" + _rows
            ),
            mo.hstack([
                mo.stat(f"{_sum['median']:.3f}", label="Median α", bordered=True),
                mo.stat(f"{_sum['p10']:.3f} - {_sum['p90']:.3f}", label="P10 - P90", bordered=True),
                mo.stat(f"{_sum['frac_ge_09'] * 100:.0f}%", label="Heads with α >= 0.90", bordered=True),
                mo.stat(f"{_sum['std']:.3f}", label="Head-to-head std", bordered=True),
            ], justify="start"),
            mo.md(
                "#### Learned α by head\n"
                f"| Layer | {_head_header} |\n"
                f"|---| {_head_sep} |\n"
                + _head_rows
                if _nhead > 0 else "#### Learned α by head\nNo head-level values available."
            ),
            mo.callout(mo.md(
                "The slider above has been set to the learned value. "
                "Drag it to explore how nearby values affect the bias."
            ), kind="success"),
        ])
    learn_out
    return (learned_result,)


@app.cell
def _(F, alpha_dim, alpha_seed, alpha_slider, learned_result, mo,
      np, plt, project_exclusive, torch):
    torch.manual_seed(alpha_seed.value)
    _T_a, _D_a = 64, alpha_dim.value
    _V_a = torch.randn(_T_a, _D_a)
    _Y_a = torch.randn(_T_a, _D_a)

    _alphas_sweep = np.linspace(0.0, 1.0, 41)
    _mean_curve = [
        F.cosine_similarity(project_exclusive(_Y_a, _V_a, float(_al)), _V_a, dim=-1).mean().item()
        for _al in _alphas_sweep
    ]

    _cur = alpha_slider.value
    _cos_cur = F.cosine_similarity(
        project_exclusive(_Y_a, _V_a, _cur), _V_a, dim=-1
    ).numpy()
    _cos_sa = F.cosine_similarity(_Y_a, _V_a, dim=-1).numpy()
    _mc = float(np.mean(_cos_cur))
    _ms = float(np.mean(_cos_sa))
    _alpha_inference = None
    if learned_result is not None:
        _s = learned_result["summary"]
        if _s["mean"] >= 0.9 and _s["frac_ge_09"] >= 0.6:
            _alpha_inference = (
                f"Learned α is concentrated near 1 (mean `{_s['mean']:.3f}`, "
                f"{_s['frac_ge_09'] * 100:.0f}% of heads ≥ 0.90). "
                "Inference (based on this run's optimization loss): the tiny LM "
                "prefers near-full projection removal."
            )
            _alpha_kind = "success"
        elif _s["mean"] >= 0.75:
            _alpha_inference = (
                f"Learned α is high but not saturated (mean `{_s['mean']:.3f}`). "
                "Inference (loss-based): strong exclusion helps, but this run "
                "leaves room for partial retention in some heads."
            )
            _alpha_kind = "warn"
        else:
            _alpha_inference = (
                f"Learned α stayed relatively low (mean `{_s['mean']:.3f}`). "
                "Inference (loss-based): this run does not support full "
                "orthogonalization as a dominant preference; check seed, "
                "training length, and model size."
            )
            _alpha_kind = "warn"

    _fig_a, _ax = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    _l, _r = _ax[0], _ax[1]

    _l.plot(_alphas_sweep, _mean_curve, color="#2F6DA5", linewidth=2.4, zorder=2)
    _l.fill_between(_alphas_sweep, _mean_curve, min(_mean_curve), color="#2F6DA5", alpha=0.12)
    _l.axvline(_cur, color="#D64550", linewidth=2.0, linestyle="--",
               label=f"slider  α = {_cur:.2f}")
    if learned_result is not None:
        _lv = learned_result["mean"]
        _l.axvline(_lv, color="#2EA46F", linewidth=1.8, linestyle=":",
                   label=f"learned α = {_lv:.3f}")
    _l.axhline(_mc, color="#D64550", linewidth=0.9, linestyle=":", alpha=0.6)
    _l.set_xlabel("α (exclusion strength)")
    _l.set_ylabel(r"Mean $\cos(z_i, v_i)$")
    _l.set_title(r"Output - value cosine vs $\alpha$")
    _l.grid(True, axis="y")
    _l.grid(False, axis="x")
    _l.legend(frameon=False, fontsize=9)
    _l.spines["top"].set_visible(False); _l.spines["right"].set_visible(False)

    _r.hist(_cos_sa, bins=25, alpha=0.45, color="#2F6DA5",
            label="SA (α=0)", edgecolor="white")
    _r.hist(_cos_cur, bins=25, alpha=0.8, color="#D64550",
            label=f"α = {_cur:.2f}", edgecolor="white")
    _r.axvline(_ms, color="#2F6DA5", lw=1.5, ls="--")
    _r.axvline(_mc, color="#D64550", lw=1.5, ls="--")
    if learned_result is not None:
        _lcos = F.cosine_similarity(
            project_exclusive(_Y_a, _V_a, learned_result["mean"]), _V_a, dim=-1
        ).numpy()
        _r.axvline(float(np.mean(_lcos)), color="#2EA46F", lw=1.8, ls=":",
                   label=f"learned α = {learned_result['mean']:.3f}")
    _r.set_xlabel(r"$\cos(z_i, v_i)$"); _r.set_ylabel("Count")
    _r.set_title(r"Distribution at current $\alpha$")
    _r.grid(True, axis="y")
    _r.grid(False, axis="x")
    _r.legend(frameon=False, fontsize=9)
    _r.spines["top"].set_visible(False); _r.spines["right"].set_visible(False)

    mo.vstack([
        mo.as_html(_fig_a),
        mo.hstack([
            mo.stat(f"α = {_cur:.2f}", label="Slider α", bordered=True),
            mo.stat(f"{_mc:.3f}", label="Output cosine (slider α)", bordered=True),
            mo.stat(f"{_ms:.3f}", label="SA baseline cosine", bordered=True),
            mo.stat(
                f"{(_ms - _mc) / (_ms + 1e-8) * 100:.0f}%",
                label="Bias reduced", bordered=True,
            ),
            mo.stat(
                f"{learned_result['mean']:.3f}" if learned_result is not None else "—",
                label="Learned mean α", bordered=True,
            ),
        ]),
        mo.hstack([
            mo.stat(
                f"{learned_result['summary']['frac_ge_09'] * 100:.0f}%"
                if learned_result is not None else "—",
                label="Heads with α >= 0.90",
                bordered=True,
            ),
            mo.stat(
                f"{learned_result['summary']['p10']:.3f} - "
                f"{learned_result['summary']['p90']:.3f}"
                if learned_result is not None else "—",
                label="Learned α P10 - P90",
                bordered=True,
            ),
        ], justify="start"),
        mo.callout(mo.md(_alpha_inference), kind=_alpha_kind)
        if _alpha_inference is not None else mo.callout(
            mo.md(
                "No learned α yet. Click **Learn optimal α** to fit the tiny LM; "
                "inferences in this section are reported from that run's loss dynamics."
            ),
            kind="warn",
        ),
        mo.callout(mo.md(
            "**Key insight:** the bias decreases linearly with α and reaches ≈ 0 "
            "at α = 1. In this notebook, the learned-α conclusion is based on "
            "optimization/validation loss behavior in a tiny model, not a "
            "downstream task benchmark."
        ), kind="success"),
    ])
    return


# ==============================================================================
# §7: 4-way training proof: SA vs XSA × AdamW vs Muon
# ==============================================================================

@app.cell
def _(mo):
    mo.md(r"""
    ## §7. Architecture vs Optimizer test

    XSA beats SA in the paper's large-scale runs. But *why*? Two competing hypotheses:

    - **Architecture hypothesis**: XSA is a better inductive bias regardless of optimizer.
    - **Optimization hypothesis**: the self-copy is an optimization artifact; a geometry-aware optimizer like Muon would fix it without changing the architecture.

    We train **4 tiny LMs from the same init, on the same batch schedule**:

    | | AdamW | Muon |
    |---|---|---|
    | **SA** (α = 0) | SA + AdamW | SA + Muon |
    | **XSA** (α = 1) | XSA + AdamW | XSA + Muon |

    Each model is a 4-layer 128-dim Transformer (RoPE + RMSNorm + SwiGLU). ~60 - 90 s on CPU at defaults. At this small scale, the test is unlikely to fully disambiguate, and proper pretraining runs are needed for a definitive answer. Inference below is from this run's **validation loss**, not downstream task scores.
    """)
    return


@app.cell
def _(mo):
    train_steps = mo.ui.slider(100, 1000, value=300, step=50,
                                label="Train steps", show_value=True)
    train_seq_len = mo.ui.slider(16, 96, value=64, step=16,
                                 label="Sequence length", show_value=True)
    train_batch = mo.ui.slider(8, 32, value=16, step=4,
                                label="Batch size", show_value=True)
    train_dim = mo.ui.slider(64, 192, value=128, step=32,
                              label="Model dim", show_value=True)
    train_layers = mo.ui.slider(2, 6, value=4, step=1,
                                 label="Layers", show_value=True)
    train_btn = mo.ui.run_button(
        label="🏋️ Train all 4 models", kind="success",
    )
    mo.vstack([
        mo.hstack([train_steps, train_seq_len, train_batch]),
        mo.hstack([train_dim, train_layers, train_btn]),
    ])
    return (train_batch, train_btn, train_dim, train_layers,
            train_seq_len, train_steps)


@app.cell
def _(get_token_data_info, mo, plot_four_losses, time, train_batch, train_btn, train_dim,
      train_four_models, train_layers, train_seq_len, train_steps):
    if not train_btn.value:
        try:
            _data_info = get_token_data_info()
        except Exception:
            _data_info = {
                "chars": 0,
                "num_sources": 0,
                "vocab_size": 50257,
                "split_ratio": "90/10",
                "urls_used": [],
                "source_mode": "fallback",
            }
        _urls_used = _data_info.get("urls_used", [])
        if _urls_used:
            _source_lines = "".join([f"- Source: `{u}`\n" for u in _urls_used])
        else:
            _source_lines = "- Source: local fallback text (URL fetch unavailable).\n"
        train_out = mo.vstack([
            mo.callout(mo.md(
                "Click **Train all 4 models** to fit SA+AdamW, SA+Muon, XSA+AdamW, "
                "XSA+Muon from the same init and compare training dynamics."
            ), kind="info"),
            mo.callout(mo.md(
                "**Training data:**\n"
                "- Mixed text fetched from common public corpora "
                "(Tiny Shakespeare + classic-book URLs from guteburg), with local fallback if fetch fails.\n"
                f"- Current cached corpus: `{_data_info['chars']:,}` chars from "
                f"`{_data_info['num_sources']}` source(s).\n"
                f"- Tokenized with GPT-2 BPE (`{_data_info['vocab_size']}` tokens).\n"
                f"- {_data_info['split_ratio']} train/validation split.\n"
                "- Objective: causal next-token cross-entropy.\n"
            ), kind="neutral"),
        ])
    else:
        _RUN_LABELS = {
            "sa_adam":  "SA  + AdamW  (run 1/4)",
            "sa_muon":  "SA  + Muon   (run 2/4)",
            "xsa_adam": "XSA + AdamW  (run 3/4)",
            "xsa_muon": "XSA + Muon   (run 4/4)",
        }
        _total_steps = train_steps.value * 4

        def _on_step_four(name, step, total_per_run, loss):
            if step % 25 == 0 or step == total_per_run - 1:
                _run_idx = list(_RUN_LABELS).index(name)
                _overall_done = _run_idx * total_per_run + step + 1
                _pct = _overall_done / _total_steps
                _bar = "█" * int(_pct * 30) + "░" * (30 - int(_pct * 30))
                mo.output.replace(mo.vstack([
                    mo.md(f"### Training on mixed web text…"),
                    mo.md(f"**{_RUN_LABELS[name]}**"),
                    mo.md(
                        f"`[{_bar}]` **{_overall_done}/{_total_steps}** total steps; "
                        f"step loss: `{loss:.4f}`"
                    ),
                ]))

        _t0 = time.time()
        _results, _vocab = train_four_models(
            steps=train_steps.value,
            seq_len=train_seq_len.value,
            batch_size=train_batch.value,
            model_dim=train_dim.value,
            num_layers=train_layers.value,
            on_step=_on_step_four,
        )
        _elapsed = time.time() - _t0
        _fig_loss = plot_four_losses(_results)

        _val_rows = "| | **AdamW** | **Muon** |\n|---|---|---|\n"
        _val_rows += (
            f"| **SA**  | {_results['sa_adam']['val_loss']:.4f} | "
            f"{_results['sa_muon']['val_loss']:.4f} |\n"
            f"| **XSA** | {_results['xsa_adam']['val_loss']:.4f} | "
            f"{_results['xsa_muon']['val_loss']:.4f} |"
        )

        _xsa_a = _results["xsa_adam"]["val_loss"]
        _sa_m = _results["sa_muon"]["val_loss"]
        if _xsa_a < _sa_m - 0.002:
            _interp = (
                f"**Based on validation loss in this tiny run:** "
                f"XSA+AdamW ({_xsa_a:.4f}) < SA+Muon ({_sa_m:.4f}). "
                "This favors an architecture benefit over an optimizer-only story."
            )
            _kind = "success"
        elif abs(_xsa_a - _sa_m) < 0.002:
            _interp = (
                f"**Based on validation loss in this tiny run:** "
                f"XSA+AdamW ≈ SA+Muon ({_xsa_a:.4f} vs {_sa_m:.4f}). "
                "The advantage may be partly optimizer-side at this scale."
            )
            _kind = "warn"
        else:
            _interp = (
                f"**Based on validation loss in this tiny run:** "
                f"SA+Muon ({_sa_m:.4f}) < XSA+AdamW ({_xsa_a:.4f}). "
                "At this scale, optimizer geometry helps more than XSA."
            )
            _kind = "warn"

        train_out = mo.vstack([
            mo.md(f"Training runtime: `{_elapsed:.1f}s`  |  Vocab: `{_vocab}` tokens"),
            mo.as_html(_fig_loss),
            mo.md("### Validation loss\n" + _val_rows),
            mo.callout(mo.md(_interp), kind=_kind),
        ])
    train_out
    return


# ==============================================================================
# §8: Key findings table
# ==============================================================================

@app.cell
def _(mo):
    mo.md(r"""
    ## §8. Key findings

    | Finding | Evidence |
    |---|---|
    | **Attention similarity bias is real, but architecture-conditional** | 13-model survey (§4): clean monotone in a minority of models (notably BERT and Phi-2), *opposite* in GPT-2, noisy elsewhere |
    | **XSA eliminates the bias by construction** | Gram-Schmidt geometry (§3C): $\cos(z, v) = 0$ exactly |
    | **Final-layer `diag_attn` spike is the most universal signal** | Present in most non-GPT-2 models; the most consistent cross-architecture pattern in this notebook |
    | **Bias is partly input-dependent, not a pure architectural invariant** | Stress test (§5): corpus change visibly shifts all three curves on GPT-2 |
    | **Tiny-run loss evidence favors high α (often near 1)** | Learnable-α experiment (§6): tiny LM optimization/validation loss tends to push α high |

    ---

    ### How XSA works in one picture

    ```python
    # Standard SA
    y_i = (attn_weights @ V)[i]          # high cos(y_i, v_i), with substantial self-copy

    # XSA: project out the self-value component
    v_hat = F.normalize(V[i], dim=-1)
    z_i = y_i - (y_i * v_hat).sum(-1, keepdim=True) * v_hat  # cos(z_i, v_i) = 0
    ```

    Two extra lines of code in any attention block, no extra parameters, ever so slightly slower due to an extra compute step. The paper reports consistent training/validation loss improvements and downstream evaluation gains across 0.7B - 2.7B models trained on 100B tokens.

    ---

    """)
    return


# ==============================================================================
# §9: What we learned (caveats + confidence calibration)
# ==============================================================================

@app.cell
def _(mo):
    mo.md(r"""
    ## §9. What we learned

    Attention-Similarity Bias is real in our experiments. Our main takeaway is: on architectures closer to the paper setup (especially **MHA + RoPE** style stacks), the trends are typically **monotonically increasing**, matching the paper's empirical trend.

    On other pretrained architectures, trends vary more by model/layer/metric. Even when monotonicity is not clean, we still observe a strong high-bias regime in many cases rather than an absence of the effect.

    In §7, we also probed architecture vs optimization geometry. On this small setup, **SA + Muon** gave a better loss curve than **XSA + AdamW**, suggesting optimization geometry may be a meaningful factor. At this scale, though, we cannot conclude strongly yet; larger runs are needed.

    We also observe the bias signal in vision transformers, indicating this is not language-only.
    ---

    ### Caveats

    | Caveat | Impact |
    |---|---|
    | `seq_length=256`, `num_sequences=64` (paper uses 2048 × 1024) | Bias may be more pronounced at longer contexts |
    | Single mixed web corpus family, modest diversity | Results may not generalise to other domains |
    | One seed per model, no error bars | Noisy models can't be reliably classified |
    """)
    return


if __name__ == "__main__":
    app.run()
