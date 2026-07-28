from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
import math
from numbers import Real
from typing import Any, Callable

import networkx as nx
import numpy as np


@dataclass
class CSR:
    nodes: list
    index: dict | None
    indptr: np.ndarray
    indices: np.ndarray
    weights: np.ndarray


def _float64_weight(value) -> float:
    if not isinstance(value, Real):
        raise TypeError(f"edge weight {value!r} cannot be represented as float64")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"edge weight {value!r} must be finite")
    if converted != value:
        raise ValueError(f"edge weight {value!r} cannot be represented exactly as float64")
    return converted


def _edge_weight(G, u, v, data, weight):
    if weight is None:
        return 1.0
    if callable(weight):
        value = weight(u, v, data)
        return None if value is None else _float64_weight(value)
    if G.is_multigraph():
        return min(_float64_weight(attrs.get(weight, 1.0)) for attrs in data.values())
    return _float64_weight(data.get(weight, 1.0))


def _edge_weight_sum(G, u, v, data, weight):
    if not G.is_multigraph():
        return _edge_weight(G, u, v, data, weight)
    if weight is None:
        return float(len(data))
    if callable(weight):
        value = weight(u, v, data)
        return None if value is None else _float64_weight(value)
    total = sum(_float64_weight(attrs.get(weight, 1.0)) for attrs in data.values())
    return _float64_weight(total)


def csr(
    G,
    weight=None,
    *,
    reverse: bool = False,
    undirected: bool = False,
    sort_neighbors: Callable | None = None,
    aggregate_parallel: str = "min",
) -> CSR:
    nodes = list(G)
    indptr_values = [0]
    index_values = []
    weight_values = []
    indices_append = index_values.append
    weights_append = weight_values.append
    identity_index = all(type(node) is int and node == i for i, node in enumerate(nodes))
    index = None if identity_index else {node: i for i, node in enumerate(nodes)}
    directed = G.is_directed()
    multigraph = G.is_multigraph()
    unit_weights = weight is None and (not multigraph or aggregate_parallel != "sum")

    if not (undirected and directed) and sort_neighbors is None and unit_weights and identity_index:
        adjacency = G._pred if reverse and directed else G._adj
        degrees = np.fromiter(
            (len(row) for row in adjacency.values()), dtype=np.int64, count=len(nodes)
        )
        indptr = np.empty(len(nodes) + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(degrees, out=indptr[1:])
        indices = np.fromiter(
            chain.from_iterable(adjacency.values()),
            dtype=np.int64,
            count=int(indptr[-1]),
        )
        if not len(indices):
            indices = np.empty(1, dtype=np.int64)[:0]
        weights = np.ones(max(1, len(indices)), dtype=np.float64)[:len(indices)]
        return CSR(nodes, index, indptr, indices, weights)

    if undirected and directed:
        for u in nodes:
            neighbors = list(dict.fromkeys([*G.successors(u), *G.predecessors(u)]))
            entries = []
            for v in neighbors:
                values = []
                if G.has_edge(u, v):
                    values.append(_edge_weight(G, u, v, G.get_edge_data(u, v), weight))
                if G.has_edge(v, u):
                    values.append(_edge_weight(G, v, u, G.get_edge_data(v, u), weight))
                values = [x for x in values if x is not None]
                if values:
                    entries.append((v, min(values)))
            if sort_neighbors is not None:
                ordered = list(sort_neighbors([v for v, _ in entries]))
                values_by_node = {v: value for v, value in entries}
                entries = [(v, values_by_node[v]) for v in ordered]
            for v, value in entries:
                indices_append(v if identity_index else index[v])
                if not unit_weights:
                    weights_append(value)
            indptr_values.append(len(index_values))
    else:
        adjacency = G._pred if reverse and directed else G._adj
        reducer = _edge_weight_sum if aggregate_parallel == "sum" else _edge_weight
        for u in nodes:
            row = adjacency[u]
            if sort_neighbors is not None:
                entries = []
                for v, data in row.items():
                    if reverse and directed:
                        value = reducer(G, v, u, data, weight)
                    else:
                        value = reducer(G, u, v, data, weight)
                    if value is not None:
                        entries.append((v, value))
                ordered = list(sort_neighbors([v for v, _ in entries]))
                values_by_node = {v: value for v, value in entries}
                for v in ordered:
                    indices_append(v if identity_index else index[v])
                    if not unit_weights:
                        weights_append(values_by_node[v])
            elif unit_weights:
                if identity_index:
                    index_values.extend(row)
                else:
                    index_values.extend(index[v] for v in row)
            elif not multigraph and not callable(weight):
                for v, data in row.items():
                    indices_append(v if identity_index else index[v])
                    weights_append(_float64_weight(data.get(weight, 1.0)))
            elif not multigraph:
                for v, data in row.items():
                    value = weight(v, u, data) if reverse and directed else weight(u, v, data)
                    if value is not None:
                        indices_append(v if identity_index else index[v])
                        weights_append(_float64_weight(value))
            else:
                for v, data in row.items():
                    if reverse and directed:
                        value = reducer(G, v, u, data, weight)
                    else:
                        value = reducer(G, u, v, data, weight)
                    if value is not None:
                        indices_append(v if identity_index else index[v])
                        weights_append(value)
            indptr_values.append(len(index_values))

    indptr = np.asarray(indptr_values, dtype=np.int64)
    indices_owner = np.asarray(index_values or [0], dtype=np.int64)
    indices = indices_owner if index_values else indices_owner[:0]
    weights = (
        np.ones(max(1, len(index_values)), dtype=np.float64)[:len(index_values)]
        if unit_weights
        else np.asarray(weight_values or [0.0], dtype=np.float64)[:len(weight_values)]
    )
    return CSR(nodes, index, indptr, indices, weights)


def require_node(data: CSR, node: Any) -> int:
    if data.index is None:
        try:
            candidate = hash(node)
        except TypeError:
            candidate = -1
        if 0 <= candidate < len(data.nodes) and data.nodes[candidate] == node:
            return candidate
        raise nx.NodeNotFound(f"Source {node} is not in G")
    try:
        return data.index[node]
    except KeyError:
        raise nx.NodeNotFound(f"Source {node} is not in G") from None
