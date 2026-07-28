from __future__ import annotations

import networkx as nx
import numpy as np

from ._graph import csr
from ._lib import addr, f64, i64, lib


def betweenness_centrality(G, k=None, normalized=True, weight=None, endpoints=False, seed=None):
    if k is not None:
        raise NotImplementedError("sampled betweenness (k=...) is not implemented")
    data = csr(G, weight)
    n = len(data.nodes)
    scores, sigma, delta = (f64(n) for _ in range(3))
    order = i64(n)
    if weight is None:
        dist, queue = i64(n), i64(n)
        lib().mnx_betweenness(
            addr(data.indptr), addr(data.indices), n, int(endpoints), addr(scores),
            addr(dist), addr(sigma), addr(delta), addr(queue), addr(order),
        )
    else:
        if np.any(data.weights <= 0):
            raise nx.NetworkXError("weighted betweenness requires strictly positive edge weights")
        dist = f64(n)
        heap, pos = i64(n), i64(n)
        lib().mnx_betweenness_weighted(
            addr(data.indptr), addr(data.indices), addr(data.weights), n, int(endpoints),
            addr(scores), addr(dist), addr(sigma), addr(delta), addr(heap), addr(pos),
            addr(order),
        )
    if normalized:
        if endpoints:
            scale = 1.0 / (n * (n - 1)) if n >= 2 else None
        else:
            scale = 1.0 / ((n - 1) * (n - 2)) if n > 2 else None
    else:
        scale = 0.5 if not G.is_directed() else None
    if scale is not None:
        scores *= scale
    return dict(zip(data.nodes, scores.tolist()))


def closeness_centrality(G, u=None, distance=None, wf_improved=True):
    data = csr(G, distance, reverse=G.is_directed())
    n = len(data.nodes)
    scores = f64(n)
    if distance is None:
        queue, dist = i64(n), i64(n)
        lib().mnx_closeness_unweighted(
            addr(data.indptr), addr(data.indices), n, int(wf_improved),
            addr(scores), addr(queue), addr(dist),
        )
    else:
        if np.any(data.weights < 0):
            raise ValueError("negative edge weights are not supported")
        dist = f64(n)
        pred, heap, pos, order = (i64(n) for _ in range(4))
        lib().mnx_closeness_weighted(
            addr(data.indptr), addr(data.indices), addr(data.weights), n, int(wf_improved),
            addr(scores), addr(dist), addr(pred), addr(heap), addr(pos), addr(order),
        )
    values = dict(zip(data.nodes, scores.tolist()))
    if u is not None:
        if u not in values:
            raise nx.NodeNotFound(f"Node {u} is not in the graph.")
        return values[u]
    return values


def degree_centrality(G):
    n = len(G)
    if n <= 1:
        return {node: 1 for node in G}
    scale = 1.0 / (n - 1) if n > 1 else 1.0
    return {node: degree * scale for node, degree in G.degree()}


def in_degree_centrality(G):
    if not G.is_directed():
        raise nx.NetworkXNotImplemented("not implemented for undirected type")
    n = len(G)
    if n <= 1:
        return {node: 1 for node in G}
    scale = 1.0 / (n - 1) if n > 1 else 1.0
    return {node: degree * scale for node, degree in G.in_degree()}


def out_degree_centrality(G):
    if not G.is_directed():
        raise nx.NetworkXNotImplemented("not implemented for undirected type")
    n = len(G)
    if n <= 1:
        return {node: 1 for node in G}
    scale = 1.0 / (n - 1) if n > 1 else 1.0
    return {node: degree * scale for node, degree in G.out_degree()}


def pagerank(
    G, alpha=0.85, personalization=None, max_iter=100, tol=1.0e-6,
    nstart=None, weight="weight", dangling=None,
):
    if personalization is not None or nstart is not None or dangling is not None:
        raise NotImplementedError("custom personalization, nstart, and dangling are not implemented")
    if len(G) == 0:
        return {}
    data = csr(G, weight, aggregate_parallel="sum")
    if np.any(data.weights < 0):
        raise nx.NetworkXError("PageRank does not support negative edge weights")
    n = len(data.nodes)
    rank, next_values, strength = (f64(n) for _ in range(3))
    iterations = lib().mnx_pagerank(
        addr(data.indptr), addr(data.indices), addr(data.weights), n, float(alpha),
        int(max_iter), float(tol), addr(rank), addr(next_values), addr(strength),
    )
    if iterations < 0:
        raise nx.PowerIterationFailedConvergence(max_iter)
    return dict(zip(data.nodes, rank.tolist()))


def eigenvector_centrality(G, max_iter=100, tol=1.0e-6, nstart=None, weight=None):
    if nstart is not None:
        raise NotImplementedError("custom nstart is not implemented")
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept("cannot compute centrality for the null graph")
    data = csr(G, weight, aggregate_parallel="sum")
    n = len(data.nodes)
    vector, next_values = f64(n), f64(n)
    iterations = lib().mnx_eigenvector(
        addr(data.indptr), addr(data.indices), addr(data.weights), n, int(max_iter),
        float(tol), addr(vector), addr(next_values),
    )
    if iterations < 0:
        raise nx.PowerIterationFailedConvergence(max_iter)
    return dict(zip(data.nodes, vector.tolist()))
