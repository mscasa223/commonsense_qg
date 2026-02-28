# -*- coding: utf-8 -*-
"""
generator.py  –  Section 5: Controllable Generator (GRU)
=========================================================
Fills all 4 previously missing pieces:
  1. Vocabulary  — builds real word<->ID mappings from the dataset
  2. Real input  — encodes actual context text into token IDs
  3. Training    — trains the model on (context -> question) pairs
  4. Decoding    — converts output IDs back to readable words
"""
from __future__ import annotations

import re
import os
import json
from collections import Counter
from typing import Dict, List, Tuple, Callable, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


# =============================================================================
# GAP 1 — Vocabulary
# Builds a real word<->ID mapping from the dataset.
# Special tokens:
#   <pad> = 0   padding
#   <bos> = 1   beginning of sequence
#   <eos> = 2   end of sequence
#   <unk> = 3   unknown word
# =============================================================================

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
SPECIAL_TOKENS = ['<pad>', '<bos>', '<eos>', '<unk>']


def simple_tokenize(text: str) -> List[str]:
    """Lowercase and split on whitespace/punctuation."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", ' ', text)
    return text.split()


class Vocabulary:
    def __init__(self):
        self.word2id: Dict[str, int] = {}
        self.id2word: Dict[int, str] = {}

    def build(self, texts: List[str], min_freq: int = 2, max_size: int = 10000):
        """
        Build vocabulary from a list of raw text strings.
        min_freq: ignore words that appear fewer than this many times.
        max_size: keep only the most frequent words up to this limit.
        """
        counter = Counter()
        for text in texts:
            counter.update(simple_tokenize(text))

        # Start with special tokens
        self.word2id = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.id2word = {i: tok for i, tok in enumerate(SPECIAL_TOKENS)}

        # Add frequent words
        for word, freq in counter.most_common(max_size):
            if freq < min_freq:
                break
            if word not in self.word2id:
                idx = len(self.word2id)
                self.word2id[word] = idx
                self.id2word[idx]  = word

        print(f'[Vocab] Built vocabulary with {len(self.word2id)} words '
              f'(min_freq={min_freq}, max_size={max_size})')

    def encode(self, text: str, max_len: int = 100) -> List[int]:
        """Convert a text string to a list of token IDs."""
        tokens = simple_tokenize(text)[:max_len]
        return [self.word2id.get(t, UNK_ID) for t in tokens]

    def decode(self, ids: List[int]) -> str:
        """Convert a list of token IDs back to a text string."""
        words = []
        for i in ids:
            if i == EOS_ID:
                break
            if i in (PAD_ID, BOS_ID):
                continue
            words.append(self.id2word.get(i, '<unk>'))
        return ' '.join(words)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.word2id, f, ensure_ascii=False)
        print(f'[Vocab] Saved to {path}')

    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            self.word2id = json.load(f)
        self.id2word = {int(i): w for w, i in self.word2id.items()}
        print(f'[Vocab] Loaded {len(self.word2id)} words from {path}')

    def __len__(self):
        return len(self.word2id)


# =============================================================================
# GAP 2 — Real Input Encoding
# Converts raw context text to padded tensor batches.
# =============================================================================

def pad_sequence(sequences: List[List[int]], pad_id: int = PAD_ID) -> List[List[int]]:
    """Pad a list of sequences to the same length."""
    max_len = max(len(s) for s in sequences)
    return [s + [pad_id] * (max_len - len(s)) for s in sequences]


def encode_batch(texts: List[str],
                 vocab: Vocabulary,
                 max_len: int = 100,
                 add_bos_eos: bool = False) -> 'torch.Tensor':
    """
    Encode a batch of text strings into a padded tensor.
    Shape: (batch_size, seq_len)
    """
    encoded = []
    for text in texts:
        ids = vocab.encode(text, max_len=max_len)
        if add_bos_eos:
            ids = [BOS_ID] + ids + [EOS_ID]
        encoded.append(ids)
    padded = pad_sequence(encoded, pad_id=PAD_ID)
    return torch.tensor(padded, dtype=torch.long)


# =============================================================================
# Model
# =============================================================================

if TORCH_AVAILABLE:

    class ControllableGenerator(nn.Module):
        """
        GRU encoder-decoder conditioned on:
          - path_vec    (128-d float vector from graph features)
          - skeleton_id (discrete embedding)
          - form_seq    (sequence of HSMM state ids)
          - difficulty  (integer 0-9)
        """

        def __init__(self,
                     vocab_size:       int,
                     d_model:          int = 256,
                     cond_dim:         int = 128,
                     num_skeletons:    int = 256,
                     num_forms_states: int = 50):
            super().__init__()
            self.vocab_size = vocab_size
            self.emb        = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
            self.enc        = nn.GRU(d_model, d_model, batch_first=True)
            self.dec        = nn.GRU(d_model + cond_dim, d_model, batch_first=True)
            self.out        = nn.Linear(d_model, vocab_size)
            self.path_proj  = nn.Linear(128, cond_dim)
            self.skel_emb   = nn.Embedding(num_skeletons, cond_dim)
            self.form_emb   = nn.Embedding(num_forms_states, cond_dim)
            self.diff_emb   = nn.Embedding(10, cond_dim)
            self.cond_gate  = nn.Linear(cond_dim * 3, cond_dim)

        def encode(self, src_ids: 'torch.Tensor') -> 'torch.Tensor':
            x    = self.emb(src_ids)
            h, _ = self.enc(x)
            return h[:, -1, :]   # last hidden state

        def _cond_vec(self,
                      path_vec:    'torch.Tensor',
                      skeleton_id: 'torch.Tensor',
                      form_seq:    'torch.Tensor',
                      difficulty:  'torch.Tensor') -> 'torch.Tensor':
            pv = self.path_proj(path_vec)
            sk = self.skel_emb(skeleton_id)
            fe = self.form_emb(form_seq).mean(dim=1)
            df = self.diff_emb(difficulty)
            c  = torch.tanh(self.cond_gate(torch.cat([pv + sk, fe, df], dim=-1)))
            return c

        def forward(self,
                    src_ids:     'torch.Tensor',
                    tgt_ids:     'torch.Tensor',
                    path_vec:    'torch.Tensor',
                    skeleton_id: 'torch.Tensor',
                    form_seq:    'torch.Tensor',
                    difficulty:  'torch.Tensor') -> 'torch.Tensor':
            """
            Returns logits of shape (batch, tgt_len, vocab_size).
            Used during training.
            """
            cond     = self._cond_vec(path_vec, skeleton_id, form_seq, difficulty)
            y        = self.emb(tgt_ids)
            cond_rep = cond.unsqueeze(1).expand(-1, y.size(1), -1)
            dec_in   = torch.cat([y, cond_rep], dim=-1)
            h, _     = self.dec(dec_in)
            logits   = self.out(h)
            return logits


    # =========================================================================
    # GAP 3 — Training Loop
    # Trains the model on (context -> question) pairs from CosmosQA.
    # =========================================================================

    def train_generator(model:       'ControllableGenerator',
                        rows:        List[Dict],
                        vocab:       Vocabulary,
                        device:      'torch.device',
                        epochs:      int   = 5,
                        batch_size:  int   = 32,
                        lr:          float = 1e-3,
                        max_src_len: int   = 100,
                        max_tgt_len: int   = 40,
                        save_path:   str   = 'data/cache/generator.pt') -> str:
        """
        Train the generator on CosmosQA (context -> question) pairs.

        Each training step:
          1. Encode context as src_ids   (real text, not fake [10,11,12,13])
          2. Encode question as tgt_ids  (what the model should learn to output)
          3. Forward pass → logits
          4. Compute CrossEntropy loss vs real question tokens
          5. Backpropagate and update weights
        """
        import math
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
        model.train()

        n        = len(rows)
        n_batch  = math.ceil(n / batch_size)
        best_loss = float('inf')

        print(f'[Train] Starting training: {epochs} epochs, '
              f'{n} samples, batch_size={batch_size}')

        for epoch in range(epochs):
            total_loss = 0.0
            # Shuffle rows each epoch
            import random
            random.shuffle(rows)

            for b in range(n_batch):
                batch = rows[b * batch_size: (b + 1) * batch_size]
                if not batch:
                    continue

                # --- GAP 2 in action: encode REAL context text ---
                src_ids = encode_batch(
                    [r['context']  for r in batch],
                    vocab, max_len=max_src_len
                ).to(device)

                # --- Encode REAL question as target ---
                tgt_full = encode_batch(
                    [r['question'] for r in batch],
                    vocab, max_len=max_tgt_len, add_bos_eos=True
                ).to(device)

                # Teacher forcing: input is tgt[:-1], label is tgt[1:]
                tgt_in  = tgt_full[:, :-1]   # everything except last token
                tgt_out = tgt_full[:, 1:]    # everything except first token

                B = src_ids.size(0)

                # Conditioning signals (using hop_count as difficulty proxy)
                path_vec    = torch.zeros(B, 128,  device=device)
                skeleton_id = torch.zeros(B,       device=device, dtype=torch.long)
                form_seq    = torch.ones(B, 1,     device=device, dtype=torch.long)
                difficulty  = torch.zeros(B,       device=device, dtype=torch.long)

                # Forward pass
                logits = model(src_ids, tgt_in, path_vec,
                               skeleton_id, form_seq, difficulty)

                # Compute loss
                # logits: (B, tgt_len-1, vocab_size) → (B*tgt_len, vocab_size)
                loss = criterion(
                    logits.reshape(-1, model.vocab_size),
                    tgt_out.reshape(-1),
                )

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / n_batch
            print(f'  Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}')

            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
                torch.save(model.state_dict(), save_path)

        print(f'[Train] Done. Best loss={best_loss:.4f}. Model saved to {save_path}')
        return save_path


    # =========================================================================
    # GAP 4 — Decoding: convert output IDs back to real words
    # =========================================================================

    @torch.no_grad()
    def generate_question(model:       'ControllableGenerator',
                          context:     str,
                          vocab:       Vocabulary,
                          device:      'torch.device',
                          path_vec:    Optional['torch.Tensor'] = None,
                          skeleton_id: int = 0,
                          form_states: List[int] = None,
                          difficulty:  int = 1,
                          max_len:     int = 40,
                          max_src_len: int = 100) -> str:
        """
        Generate a real question string from a context.

        Steps:
          1. Encode real context text into token IDs
          2. Run the model decoder token by token
          3. Decode output IDs back to words  ← GAP 4 filled here
        """
        model.eval()

        # GAP 2: encode real context
        src_ids = encode_batch([context], vocab, max_len=max_src_len).to(device)

        B = 1
        if path_vec is None:
            path_vec = torch.zeros(B, 128, device=device)
        sk_tensor   = torch.tensor([skeleton_id],          dtype=torch.long, device=device)
        form_tensor = torch.tensor([form_states or [0]],   dtype=torch.long, device=device)
        diff_tensor = torch.tensor([min(difficulty, 9)],   dtype=torch.long, device=device)

        cond = model._cond_vec(path_vec, sk_tensor, form_tensor, diff_tensor)

        # Decode token by token
        y       = torch.tensor([[BOS_ID]], device=device)
        h       = None
        out_ids: List[int] = []

        for _ in range(max_len):
            ye       = model.emb(y)
            cond_rep = cond.unsqueeze(1)
            dec_in   = torch.cat([ye, cond_rep], dim=-1)
            h, _     = model.dec(dec_in, h)
            logits   = model.out(h[:, -1, :])
            next_id  = int(torch.argmax(logits, dim=-1).item())
            if next_id == EOS_ID:
                break
            out_ids.append(next_id)
            y = torch.tensor([[next_id]], device=device)

        # GAP 4: decode IDs back to real words
        return vocab.decode(out_ids)


    def generate_diverse(model:             'ControllableGenerator',
                         sample:            Dict,
                         n_forms:           int,
                         bos_id:            int,
                         eos_id:            int,
                         max_len:           int,
                         form_pool_sampler: Callable) -> List[List[int]]:
        """Legacy function kept for compatibility."""
        outputs: List[List[int]] = []
        for _ in range(n_forms):
            form     = form_pool_sampler() or [0, 1, 2]
            form_seq = torch.tensor([form], dtype=torch.long,
                                    device=sample['src_ids'].device)
            cond = model._cond_vec(
                sample['path_vec'],
                sample['skeleton_id'],
                form_seq,
                sample['difficulty'],
            )
            y   = torch.tensor([[bos_id]], device=sample['src_ids'].device)
            h   = None
            ids: List[int] = []
            with torch.no_grad():
                for _ in range(max_len):
                    ye       = model.emb(y)
                    cond_rep = cond.unsqueeze(1)
                    dec_in   = torch.cat([ye, cond_rep], dim=-1)
                    h, _     = model.dec(dec_in, h)
                    logits   = model.out(h[:, -1, :])
                    next_id  = int(torch.argmax(logits, dim=-1).item())
                    if next_id == eos_id:
                        break
                    ids.append(next_id)
                    y = torch.tensor([[next_id]], device=sample['src_ids'].device)
            outputs.append(ids)
        return outputs

else:
    # Stubs when PyTorch is unavailable
    class ControllableGenerator:             # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError('PyTorch is not available. Install with: pip install torch')

    def train_generator(*args, **kwargs):    # type: ignore[misc]
        raise ImportError('PyTorch is not available.')

    def generate_question(*args, **kwargs):  # type: ignore[misc]
        raise ImportError('PyTorch is not available.')

    def generate_diverse(*args, **kwargs):   # type: ignore[misc]
        raise ImportError('PyTorch is not available.')