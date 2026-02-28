# -*- coding: utf-8 -*-
"""
graph.py  –  Section 2: Context + Commonsense → Inferential Graph
=================================================================
Covers:
  - Proxy-AMR parsing and context graph building
  - Commonsense KB retrieval (local TSV backend with caching)
  - Inferential graph merging
  - Subgraph / Path retrieval with lightweight beam search
  - Path feature encoding (bag-of-relations + hop count)
"""
from __future__ import annotations

import re
import os
import json
import math
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Iterable
from collections import Counter

try:
    import numpy as np
except Exception:
    np = None

# =============================================================================
# 2.1  Graph data structures
# =============================================================================

@dataclass
class Node:
    id: str
    label: str
    node_type: str = 'context'   # 'context' | 'commonsense' | 'amr'
    meta: Dict[str, str] = field(default_factory=dict)


@dataclass
class Edge:
    src: str
    rel: str
    tgt: str
    meta: Dict[str, str] = field(default_factory=dict)


@dataclass
class GraphData:
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)

    def add_node(self, node: Node):
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge):
        self.edges.append(edge)

    def neighbors(self, node_id: str) -> List[str]:
        return [e.tgt for e in self.edges if e.src == node_id]


@dataclass
class AMRGraph:
    concepts: List[str]
    relations: List[Tuple[str, str, str]]   # (head, role, tail)
    meta: Dict[str, str] = field(default_factory=dict)


# =============================================================================
# 2.1  Proxy-AMR parsing
# =============================================================================

def extract_entities(context: str) -> List[str]:
    """Heuristic entity extraction (proxy for NER)."""
    cap_spans = re.findall(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", context)
    singles   = re.findall(r"\b([A-Z][a-z]{2,})\b", context)
    quoted    = re.findall(r'"([^"]+)"', context)
    candidates = cap_spans + singles + quoted
    seen: Set[str] = set()
    entities: List[str] = []
    for c in candidates:
        c = c.strip()
        if c and c not in seen:
            seen.add(c)
            entities.append(c)
    return entities


def parse_amr(context: str, use_proxy: bool = True) -> AMRGraph:
    """Proxy-AMR: concepts are entities; edges are within-sentence co-occurrence."""
    if not use_proxy:
        raise NotImplementedError(
            "Real AMR backend not integrated in this scaffold."
        )
    entities = extract_entities(context)
    concepts = list(entities)
    relations: List[Tuple[str, str, str]] = []
    sentences = re.split(r"(?<=[.!?])\s+", context)
    for sent in sentences:
        sent_ents = [e for e in entities if e in sent]
        for i in range(len(sent_ents)):
            for j in range(i + 1, len(sent_ents)):
                h, t = sent_ents[i], sent_ents[j]
                relations.append((h, 'cooccur', t))
                relations.append((t, 'cooccur', h))
    return AMRGraph(concepts=concepts, relations=relations,
                    meta={'backend': 'proxy_cooccur'})


def amr_to_graph(amr: AMRGraph) -> GraphData:
    g = GraphData()
    label2id: Dict[str, str] = {}
    for i, c in enumerate(amr.concepts):
        nid = f"c{i}"
        g.add_node(Node(id=nid, label=c, node_type='amr', meta={'concept': c}))
        label2id[c] = nid
    for h, r, t in amr.relations:
        if h in label2id and t in label2id:
            g.add_edge(Edge(src=label2id[h], rel=r, tgt=label2id[t],
                            meta={'source': amr.meta.get('backend', 'amr')}))
    return g


# =============================================================================
# 2.2  KB Retriever (local TSV backend + disk cache)
# =============================================================================

Triple = Tuple[str, str, str]

CACHE_DIR_KB = os.path.join('data', 'cache', 'kb')
os.makedirs(CACHE_DIR_KB, exist_ok=True)


def normalize_concept(text: str) -> str:
    return text.strip().lower().replace(' ', '_')


class BaseKBBackend:
    def retrieve(self, concepts: List[str], k: int = 50,
                 hops: int = 2) -> List[Triple]:
        raise NotImplementedError


class LocalTSVBackend(BaseKBBackend):
    def __init__(self, sources: Optional[List[str]] = None):
        self.sources = sources or ['conceptnet.tsv', 'atomic.tsv']

    def _iter_triples(self) -> Iterable[Triple]:
        for src in self.sources:
            path = os.path.join(CACHE_DIR_KB, src)
            if not os.path.isfile(path):
                continue
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 3:
                        yield (parts[0], parts[1], parts[2])

    def retrieve(self, concepts: List[str], k: int = 50,
                 hops: int = 2) -> List[Triple]:
        norm_set = {normalize_concept(c) for c in concepts}
        triples: List[Triple] = []
        for h, r, t in self._iter_triples():
            if h in norm_set or t in norm_set:
                triples.append((h, r, t))
                if len(triples) >= k:
                    break
        return triples


def _kb_cache_key(concepts: List[str], k: int, hops: int) -> str:
    payload = json.dumps(
        {'c': [normalize_concept(c) for c in concepts], 'k': k, 'h': hops},
        sort_keys=True,
    )
    return hashlib.md5(payload.encode('utf-8')).hexdigest()


def retrieve_kb(concepts: List[str], k: int = 50, hops: int = 2,
                backend: Optional[BaseKBBackend] = None) -> List[Triple]:
    backend = backend or LocalTSVBackend()
    key = _kb_cache_key(concepts, k, hops)
    cache_path = os.path.join(CACHE_DIR_KB, f'{key}.json')
    if os.path.isfile(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    triples = backend.retrieve(concepts, k=k, hops=hops)
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(triples, f, ensure_ascii=False)
    return triples


# =============================================================================
# 2.3  Inferential Graph (context + KB merge)
# =============================================================================

@dataclass
class InferentialGraph(GraphData):
    label_index: Dict[str, str] = field(default_factory=dict)

    def index_labels(self):
        self.label_index = {n.label.lower(): nid
                            for nid, n in self.nodes.items()}

    def find_nodes_by_label(self, label: str) -> List[str]:
        key = label.lower()
        return [nid for nid, n in self.nodes.items()
                if n.label.lower() == key]


def build_inferential_graph(context_graph: GraphData,
                             kb_triples: List[Triple]) -> InferentialGraph:
    G = InferentialGraph(nodes=dict(context_graph.nodes),
                         edges=list(context_graph.edges))
    G.index_labels()

    def ensure_node(label: str, node_type: str, source: str) -> str:
        key = label.lower()
        if key in G.label_index:
            return G.label_index[key]
        nid = f"n{len(G.nodes)}"
        G.add_node(Node(id=nid, label=label, node_type=node_type,
                        meta={'source': source}))
        G.label_index[key] = nid
        return nid

    for h, r, t in kb_triples:
        h_lbl = h.replace('_', ' ')
        t_lbl = t.replace('_', ' ')
        h_id  = ensure_node(h_lbl, node_type='commonsense', source='kb')
        t_id  = ensure_node(t_lbl, node_type='commonsense', source='kb')
        G.add_edge(Edge(src=h_id, rel=r, tgt=t_id, meta={'source': 'kb'}))
    return G


# =============================================================================
# 2.4  Subgraph Retrieval (beam search)
# =============================================================================

@dataclass
class PathSkeleton:
    allowed_relations: Optional[Set[str]] = None
    max_hops: int = 3


@dataclass
class PathObj:
    nodes: List[str]
    edges: List[Edge]
    score: float


@dataclass
class Subgraph:
    paths: List[PathObj] = field(default_factory=list)

    def all_nodes(self) -> Set[str]:
        s: Set[str] = set()
        for p in self.paths:
            s.update(p.nodes)
        return s

    def all_edges(self) -> List[Edge]:
        es: List[Edge] = []
        for p in self.paths:
            es.extend(p.edges)
        return es


def _neighbors(G: InferentialGraph, nid: str,
               allowed: Optional[Set[str]]) -> List[Edge]:
    return [e for e in G.edges
            if e.src == nid and (allowed is None or e.rel in allowed)]


def beam_search_paths(G: InferentialGraph,
                      src_candidates: List[str],
                      tgt_candidates: List[str],
                      constraints: PathSkeleton,
                      beam_size: int = 5) -> List[PathObj]:
    targets = set(tgt_candidates)
    paths: List[PathObj] = []
    frontier: List[Tuple[float, List[str], List[Edge]]] = [
        (0.0, [s], []) for s in src_candidates
    ]
    for _ in range(constraints.max_hops):
        new_frontier: List[Tuple[float, List[str], List[Edge]]] = []
        for score, node_seq, edge_seq in frontier:
            last = node_seq[-1]
            for e in _neighbors(G, last, constraints.allowed_relations):
                nxt = e.tgt
                if nxt in node_seq:
                    continue
                new_score = (score
                             + (0.0 if (constraints.allowed_relations is None
                                        or e.rel in constraints.allowed_relations)
                                else 0.1)
                             + 0.2)
                nnodes = node_seq + [nxt]
                nedges = edge_seq + [e]
                if nxt in targets:
                    paths.append(PathObj(nodes=nnodes, edges=nedges,
                                         score=new_score))
                new_frontier.append((new_score, nnodes, nedges))
        new_frontier.sort(key=lambda x: x[0])
        frontier = new_frontier[:beam_size]
    paths.sort(key=lambda p: p.score)
    return paths


def retrieve_subgraph(G: InferentialGraph,
                      skeleton: PathSkeleton,
                      answer_labels: List[str],
                      topk: int = 3,
                      question_anchor_labels: Optional[List[str]] = None
                      ) -> Subgraph:
    tgt_ids: List[str] = []
    for lab in answer_labels:
        tgt_ids.extend(G.find_nodes_by_label(lab))
    tgt_ids = list(dict.fromkeys(tgt_ids))

    if question_anchor_labels:
        src_ids: List[str] = []
        for lab in question_anchor_labels:
            src_ids.extend(G.find_nodes_by_label(lab))
        src_ids = list(dict.fromkeys(src_ids))
    else:
        src_ids = [nid for nid, n in G.nodes.items()
                   if n.node_type in ('context', 'amr')]

    paths = beam_search_paths(G, src_ids, tgt_ids, skeleton)
    return Subgraph(paths=paths[:topk])


# =============================================================================
# 2.5  Path Features
# =============================================================================

def hop_count(subgraph: Subgraph) -> int:
    if not subgraph.paths:
        return 0
    return max(len(p.edges) for p in subgraph.paths)


def encode_path(subgraph: Subgraph,
                relation_vocab: Optional[Dict[str, int]] = None):
    rels      = [e.rel for e in subgraph.all_edges()]
    rel_counts = Counter(rels)
    if relation_vocab is not None and np is not None:
        vec = np.zeros(len(relation_vocab) + 1, dtype='float32')
        for r, c in rel_counts.items():
            idx = relation_vocab.get(r)
            if idx is not None:
                vec[idx] = float(c)
        vec[-1] = float(hop_count(subgraph))
        return vec
    feats = {f'rel::{r}': float(c) for r, c in rel_counts.items()}
    feats['hop_count'] = float(hop_count(subgraph))
    return feats


def features_to_vec128(feats: Dict[str, float]) -> 'np.ndarray':
    """Project dict features to a fixed 128-d vector (stable hashing)."""
    if np is None:
        raise RuntimeError('NumPy is required for features_to_vec128.')
    vec = np.zeros(128, dtype='float32')
    for k, v in feats.items():
        h   = int(hashlib.md5(k.encode('utf-8')).hexdigest(), 16)
        idx = h % 128
        vec[idx] += float(v)
    return vec
