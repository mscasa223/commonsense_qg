# -*- coding: utf-8 -*-
"""
hsmm.py  –  Section 4: Expressive Form Learning (HSMM)
=======================================================
Covers:
  - HSMM params & algorithms (Viterbi, forward log-likelihood)
  - Hard-EM trainer → hsmm.ckpt
  - Inference wrapper (HSMMInfer)
  - Expressive Pool for sampling surface forms
"""
from __future__ import annotations

import os
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict

try:
    import numpy as np
except Exception:
    np = None


# =============================================================================
# 4.1  HSMM Parameters
# =============================================================================

@dataclass
class HSMMParams:
    num_states:   int
    vocab_size:   int
    max_duration: int
    log_pi:  'np.ndarray'   # [S]
    log_A:   'np.ndarray'   # [S, S]
    log_E:   'np.ndarray'   # [S, V]
    log_D:   'np.ndarray'   # [S, L]

    def normalize_(self):
        def log_normalize(logp, axis=-1):
            m = np.max(logp, axis=axis, keepdims=True)
            z = m + np.log(np.sum(np.exp(logp - m), axis=axis, keepdims=True))
            return logp - z
        self.log_pi = log_normalize(self.log_pi)
        self.log_A  = log_normalize(self.log_A,  axis=1)
        self.log_E  = log_normalize(self.log_E,  axis=1)
        self.log_D  = log_normalize(self.log_D,  axis=1)


def random_init(num_states: int, vocab_size: int,
                max_duration: int, seed: int = 13) -> HSMMParams:
    if np is None:
        raise RuntimeError('NumPy is required for HSMM.')
    rng    = np.random.RandomState(seed)
    log_pi = np.log(rng.dirichlet(np.ones(num_states)))
    log_A  = np.log(rng.dirichlet(np.ones(num_states), size=num_states))
    log_E  = np.log(rng.dirichlet(np.ones(vocab_size), size=num_states))
    log_D  = np.log(rng.dirichlet(np.ones(max_duration), size=num_states))
    params = HSMMParams(num_states, vocab_size, max_duration,
                        log_pi, log_A, log_E, log_D)
    params.normalize_()
    return params


# =============================================================================
# 4.2  Viterbi & Forward algorithms
# =============================================================================

def viterbi_hsmm(params: HSMMParams,
                 tokens: List[int]) -> List[Tuple[int, Tuple[int, int]]]:
    S, T, L = params.num_states, len(tokens), params.max_duration
    NEG = -1e15
    B = np.full((S, T, L), NEG, dtype=float)
    for s in range(S):
        for t in range(T):
            acc = 0.0
            for d in range(1, min(L, T - t) + 1):
                span  = tokens[t:t + d]
                acc  += np.sum(params.log_E[s, span])
                B[s, t, d - 1] = acc + params.log_D[s, d - 1]

    dp = np.full((T + 1, S), NEG)
    bk: Dict = {}
    for s in range(S):
        for d in range(1, min(L, T) + 1):
            sc = params.log_pi[s] + B[s, 0, d - 1]
            if sc > dp[d, s]:
                dp[d, s] = sc
                bk[(d, s)] = (0, -1, d)

    for t in range(1, T + 1):
        for s in range(S):
            val = dp[t, s]
            if val <= NEG / 2:
                continue
            for sp in range(S):
                for d in range(1, min(L, T - t) + 1):
                    sc = val + params.log_A[s, sp] + B[sp, t, d - 1]
                    if sc > dp[t + d, sp]:
                        dp[t + d, sp] = sc
                        bk[(t + d, sp)] = (t, s, d)

    t_star, s_star = max(
        ((t, s) for t in range(T + 1) for s in range(S)),
        key=lambda x: dp[x],
    )
    segs = []
    t, s = t_star, s_star
    while t > 0 and (t, s) in bk:
        t_prev, s_prev, d = bk[(t, s)]
        segs.append((s, (t - d, t)))
        t, s = t_prev, s_prev if s_prev != -1 else 0
    segs.reverse()
    return segs


def forward_loglik_hsmm(params: HSMMParams, tokens: List[int]) -> float:
    S, T, L = params.num_states, len(tokens), params.max_duration
    NEG = -1e15
    B = np.full((S, T, L), NEG, dtype=float)
    for s in range(S):
        for t in range(T):
            acc = 0.0
            for d in range(1, min(L, T - t) + 1):
                span  = tokens[t:t + d]
                acc  += np.sum(params.log_E[s, span])
                B[s, t, d - 1] = acc + params.log_D[s, d - 1]

    alpha = np.full((T + 1, S), NEG)
    for s in range(S):
        for d in range(1, min(L, T) + 1):
            alpha[d, s] = max(alpha[d, s],
                              params.log_pi[s] + B[s, 0, d - 1])

    for t in range(1, T + 1):
        for s in range(S):
            val = alpha[t, s]
            if val <= NEG / 2:
                continue
            for sp in range(S):
                for d in range(1, min(L, T - t) + 1):
                    alpha[t + d, sp] = np.logaddexp(
                        alpha[t + d, sp],
                        val + params.log_A[s, sp] + B[sp, t, d - 1],
                    )
    return float(np.max(alpha[T]))


# =============================================================================
# 4.3  Vocabulary helpers
# =============================================================================

def build_vocab(questions_tok: List[List[str]],
                min_freq: int = 1) -> Dict[str, int]:
    cnt = Counter()
    for toks in questions_tok:
        cnt.update(toks)
    vocab = {w: i for i, (w, c) in enumerate(cnt.items()) if c >= min_freq}
    vocab['<unk>'] = len(vocab)
    return vocab


def encode_corpus(questions_tok: List[List[str]],
                  vocab: Dict[str, int]) -> List[List[int]]:
    unk = vocab.get('<unk>')
    return [[vocab.get(w, unk) for w in toks] for toks in questions_tok]


# =============================================================================
# 4.4  Hard-EM trainer
# =============================================================================

def train_hsmm(questions_tok: List[List[str]],
               num_states: int = 20,
               max_duration: int = 5,
               iters: int = 10,
               out_dir: str = 'data/cache/forms') -> str:
    if np is None:
        raise RuntimeError('NumPy is required for HSMM training.')
    os.makedirs(out_dir, exist_ok=True)
    vocab  = build_vocab(questions_tok, min_freq=1)
    X      = encode_corpus(questions_tok, vocab)
    params = random_init(num_states=num_states,
                         vocab_size=len(vocab),
                         max_duration=max_duration)

    for _ in range(iters):
        E  = np.ones_like(params.log_E)  * 1e-3
        D  = np.ones_like(params.log_D)  * 1e-3
        A  = np.ones_like(params.log_A)  * 1e-3
        PI = np.ones_like(params.log_pi) * 1e-3

        for tokens in X:
            segs = viterbi_hsmm(params, tokens)
            if not segs:
                continue
            PI[segs[0][0]] += 1.0
            for i in range(len(segs) - 1):
                A[segs[i][0], segs[i + 1][0]] += 1.0
            for s, (a, b) in segs:
                span = tokens[a:b]
                for t in span:
                    E[s, t] += 1.0
                dur = min(b - a, params.max_duration)
                D[s, dur - 1] += 1.0

        def log_norm_rows(M):
            M = M / M.sum(axis=1, keepdims=True)
            return np.log(M)

        params.log_pi = np.log(PI / PI.sum())
        params.log_A  = log_norm_rows(A)
        params.log_E  = log_norm_rows(E)
        params.log_D  = log_norm_rows(D)

    ckpt = {
        'num_states':   int(params.num_states),
        'vocab':        vocab,
        'max_duration': int(params.max_duration),
        'log_pi': params.log_pi.tolist(),
        'log_A':  params.log_A.tolist(),
        'log_E':  params.log_E.tolist(),
        'log_D':  params.log_D.tolist(),
    }
    out_path = os.path.join(out_dir, 'hsmm.ckpt')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(ckpt, f)
    return out_path


# =============================================================================
# 4.5  Inference wrapper
# =============================================================================

class HSMMInfer:
    def __init__(self, ckpt_path: str):
        with open(ckpt_path, 'r', encoding='utf-8') as f:
            ckpt = json.load(f)
        self.vocab  = ckpt['vocab']
        self.ivocab = {i: w for w, i in self.vocab.items()}
        self.params = HSMMParams(
            num_states=ckpt['num_states'],
            vocab_size=len(self.vocab),
            max_duration=ckpt['max_duration'],
            log_pi=np.array(ckpt['log_pi']),
            log_A =np.array(ckpt['log_A']),
            log_E =np.array(ckpt['log_E']),
            log_D =np.array(ckpt['log_D']),
        )

    def encode(self, toks: List[str]) -> List[int]:
        unk = self.vocab.get('<unk>')
        return [self.vocab.get(w, unk) for w in toks]

    def segment(self, toks: List[str]) -> List[Tuple[int, List[str]]]:
        ids  = self.encode(toks)
        segs = viterbi_hsmm(self.params, ids)
        return [(s, toks[a:b]) for s, (a, b) in segs]


# =============================================================================
# 4.6  Expressive Pool (diversity via form sampling)
# =============================================================================

class ExpressivePool:
    def __init__(self):
        self.form_freq: Counter = Counter()
        self.form_lex: Dict[Tuple[int, ...], Counter] = defaultdict(Counter)

    def add_segmented(self,
                      seg_sequences:   List[List[int]],
                      token_sequences: List[List[str]]):
        for states, toks in zip(seg_sequences, token_sequences):
            key = tuple(states)
            self.form_freq[key] += 1
            if toks:
                self.form_lex[key][toks[0]] += 1

    def top_forms(self, k: int = 100) -> List[Tuple[Tuple[int, ...], int]]:
        return self.form_freq.most_common(k)

    def sample_form(self,
                    temperature: float = 1.0,
                    topk:        int   = 50) -> List[int]:
        cands = self.top_forms(k=topk)
        if not cands:
            return []
        scores = [c for _, c in cands]
        mx     = max(scores)
        probs  = [math.exp((c - mx) / max(temperature, 1e-6))
                  for _, c in cands]
        s      = sum(probs)
        probs  = [p / s for p in probs]
        r = np.random.rand() if np is not None else 0.5
        cum = 0.0
        for (form, _), p in zip(cands, probs):
            cum += p
            if r <= cum:
                return list(form)
        return list(cands[-1][0])

    def multi_sample_forms(self,
                           n:           int   = 5,
                           temperature: float = 1.0,
                           topk:        int   = 50) -> List[List[int]]:
        return [self.sample_form(temperature=temperature, topk=topk)
                for _ in range(n)]
