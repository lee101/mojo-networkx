from __future__ import annotations

import math

import networkx as nx
import numpy as np

from ._graph import csr, require_node
from ._lib import addr, f64, i64, lib
from .traversal import _bfs


def _paths(data, order, pred, source_i):
    result = {}
    for value in order:
        node_i = int(value)
        chain = []
        while node_i >= 0:
            chain.append(data.nodes[node_i])
            if node_i == source_i:
                break
            node_i = int(pred[node_i])
        result[data.nodes[int(value)]] = chain[::-1]
    return result


def _distance_dict(data, order, dist):
    positions = order.tolist()
    nodes = positions if data.index is None else (data.nodes[i] for i in positions)
    return dict(zip(nodes, dist[order].tolist()))


def single_source_shortest_path_length(G, source, cutoff=None):
    data, order, dist, _ = _bfs(G, source, cutoff)
    return _distance_dict(data, order, dist)


def single_target_shortest_path_length(G, target, cutoff=None):
    data, order, dist, _ = _bfs(G, target, cutoff, reverse=True)
    return _distance_dict(data, order, dist)


def all_pairs_shortest_path_length(G, cutoff=None):
    return ((node, single_source_shortest_path_length(G, node, cutoff)) for node in G)


def single_source_shortest_path(G, source, cutoff=None):
    data, order, _, pred = _bfs(G, source, cutoff)
    return _paths(data, order, pred, require_node(data, source))


def single_target_shortest_path(G, target, cutoff=None):
    data, order, _, pred = _bfs(G, target, cutoff, reverse=True)
    reverse_paths = _paths(data, order, pred, require_node(data, target))
    return {node: path[::-1] for node, path in reverse_paths.items()}


def all_pairs_shortest_path(G, cutoff=None):
    return ((node, single_source_shortest_path(G, node, cutoff)) for node in G)


def _dijkstra(G, source, cutoff=None, weight="weight", reverse=False):
    data = csr(G, weight, reverse=reverse)
    source_i = require_node(data, source)
    if np.any(data.weights < 0):
        raise ValueError("Contradictory paths found: negative weights?")
    n = len(data.nodes)
    dist = f64(n)
    pred, heap, pos, order = (i64(n) for _ in range(4))
    limit = math.inf if cutoff is None else float(cutoff)
    count = lib().mnx_dijkstra(
        addr(data.indptr), addr(data.indices), addr(data.weights), n, source_i, limit,
        addr(dist), addr(pred), addr(heap), addr(pos), addr(order),
    )
    return data, order[:count], dist, pred


def single_source_dijkstra(G, source, target=None, cutoff=None, weight="weight"):
    data, order, dist, pred = _dijkstra(G, source, cutoff, weight)
    distances = _distance_dict(data, order, dist)
    paths = _paths(data, order, pred, require_node(data, source))
    if target is None:
        return distances, paths
    if target not in distances:
        raise nx.NetworkXNoPath(f"No path to {target}.")
    return distances[target], paths[target]


def single_source_dijkstra_path(G, source, cutoff=None, weight="weight"):
    return single_source_dijkstra(G, source, cutoff=cutoff, weight=weight)[1]


def single_source_dijkstra_path_length(G, source, cutoff=None, weight="weight"):
    data, order, dist, _ = _dijkstra(G, source, cutoff, weight)
    return _distance_dict(data, order, dist)


def dijkstra_path(G, source, target, weight="weight"):
    return single_source_dijkstra(G, source, target=target, weight=weight)[1]


def dijkstra_path_length(G, source, target, weight="weight"):
    distances = single_source_dijkstra_path_length(G, source, weight=weight)
    if target not in distances:
        raise nx.NetworkXNoPath(f"No path to {target}.")
    return distances[target]


def dijkstra_predecessor_and_distance(G, source, cutoff=None, weight="weight"):
    data, order, dist, _ = _dijkstra(G, source, cutoff, weight)
    reached = {int(i) for i in order}
    predecessors = {data.nodes[i]: [] for i in reached}
    for v in reached:
        for edge in range(int(data.indptr[v]), int(data.indptr[v + 1])):
            w = int(data.indices[edge])
            if w in reached and math.isclose(
                float(dist[v] + data.weights[edge]), float(dist[w]), rel_tol=1e-12, abs_tol=1e-12
            ):
                predecessors[data.nodes[w]].append(data.nodes[v])
    distances = {data.nodes[i]: float(dist[i]) for i in reached}
    return predecessors, distances


def all_pairs_dijkstra(G, cutoff=None, weight="weight"):
    return ((node, single_source_dijkstra(G, node, cutoff=cutoff, weight=weight)) for node in G)


def all_pairs_dijkstra_path(G, cutoff=None, weight="weight"):
    return ((node, single_source_dijkstra_path(G, node, cutoff, weight)) for node in G)


def all_pairs_dijkstra_path_length(G, cutoff=None, weight="weight"):
    return ((node, single_source_dijkstra_path_length(G, node, cutoff, weight)) for node in G)


def shortest_path(G, source=None, target=None, weight=None, method="dijkstra"):
    if method not in ("dijkstra", "bellman-ford"):
        raise ValueError(f"method not supported: {method}")
    if weight is not None and method == "bellman-ford":
        if source is None:
            if target is None:
                return dict(nx.all_pairs_bellman_ford_path(G, weight=weight))
            reverse = G.reverse(copy=False) if G.is_directed() else G
            paths = nx.single_source_bellman_ford_path(reverse, target, weight=weight)
            return {node: path[::-1] for node, path in paths.items()}
        if target is None:
            return nx.single_source_bellman_ford_path(G, source, weight=weight)
        return nx.bellman_ford_path(G, source, target, weight)
    single = single_source_shortest_path if weight is None else single_source_dijkstra_path
    if source is None:
        if target is None:
            return dict(all_pairs_shortest_path(G) if weight is None else all_pairs_dijkstra_path(G, weight=weight))
        reverse = G.reverse(copy=False) if G.is_directed() else G
        paths = single(reverse, target, weight=weight) if weight is not None else single(reverse, target)
        return {node: path[::-1] for node, path in paths.items()}
    paths = single(G, source, weight=weight) if weight is not None else single(G, source)
    if target is None:
        return paths
    try:
        return paths[target]
    except KeyError:
        raise nx.NetworkXNoPath(f"No path between {source} and {target}.") from None


def shortest_path_length(G, source=None, target=None, weight=None, method="dijkstra"):
    if method not in ("dijkstra", "bellman-ford"):
        raise ValueError(f"method not supported: {method}")
    if weight is not None and method == "bellman-ford":
        if source is None:
            if target is None:
                return nx.all_pairs_bellman_ford_path_length(G, weight=weight)
            reverse = G.reverse(copy=False) if G.is_directed() else G
            return nx.single_source_bellman_ford_path_length(reverse, target, weight=weight)
        if target is None:
            return nx.single_source_bellman_ford_path_length(G, source, weight=weight)
        return nx.bellman_ford_path_length(G, source, target, weight)
    single = single_source_shortest_path_length if weight is None else single_source_dijkstra_path_length
    if source is None:
        if target is None:
            return all_pairs_shortest_path_length(G) if weight is None else all_pairs_dijkstra_path_length(G, weight=weight)
        reverse = G.reverse(copy=False) if G.is_directed() else G
        return single(reverse, target, weight=weight) if weight is not None else single(reverse, target)
    lengths = single(G, source, weight=weight) if weight is not None else single(G, source)
    if target is None:
        return lengths
    try:
        return lengths[target]
    except KeyError:
        raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}") from None


def has_path(G, source, target):
    try:
        shortest_path(G, source, target)
        return True
    except nx.NetworkXNoPath:
        return False


def average_shortest_path_length(G, weight=None, method=None):
    n = len(G)
    if n == 0:
        raise nx.NetworkXPointlessConcept("the null graph has no paths")
    if n == 1:
        return 0
    if G.is_directed():
        if not nx.is_strongly_connected(G):
            raise nx.NetworkXError("Graph is not strongly connected.")
    elif not nx.is_connected(G):
        raise nx.NetworkXError("Graph is not connected.")
    total = sum(sum(lengths.values()) for _, lengths in shortest_path_length(G, weight=weight))
    return total / (n * (n - 1))
