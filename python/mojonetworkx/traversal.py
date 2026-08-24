from __future__ import annotations

import networkx as nx
import numpy as np

from ._graph import csr, require_node
from ._lib import addr, i64, lib


def _bfs(G, source, depth_limit=None, reverse=False, sort_neighbors=None):
    data = csr(G, reverse=reverse, sort_neighbors=sort_neighbors)
    source_i = require_node(data, source)
    n = len(data.nodes)
    limit = n if depth_limit is None else max(0, int(depth_limit))
    workspace = i64(4 * n)
    queue, dist, pred, order = (workspace[i * n:(i + 1) * n] for i in range(4))
    count = lib().mnx_bfs(
        addr(data.indptr), addr(data.indices), n, source_i, limit,
        addr(queue), addr(dist), addr(pred), addr(order),
    )
    return data, order[:count], dist, pred


def bfs_edges(G, source, reverse=False, depth_limit=None, sort_neighbors=None):
    data, order, _, pred = _bfs(G, source, depth_limit, reverse, sort_neighbors)
    for child in order[1:]:
        yield data.nodes[int(pred[child])], data.nodes[int(child)]


def bfs_tree(G, source, reverse=False, depth_limit=None, sort_neighbors=None):
    tree = nx.DiGraph()
    tree.add_node(source)
    tree.add_edges_from(bfs_edges(G, source, reverse, depth_limit, sort_neighbors))
    return tree


def bfs_predecessors(G, source, depth_limit=None, sort_neighbors=None):
    for parent, child in bfs_edges(G, source, False, depth_limit, sort_neighbors):
        yield child, parent


def bfs_successors(G, source, depth_limit=None, sort_neighbors=None):
    current = None
    children = []
    for parent, child in bfs_edges(G, source, False, depth_limit, sort_neighbors):
        if current is not None and parent != current:
            yield current, children
            children = []
        current = parent
        children.append(child)
    if current is not None:
        yield current, children


def bfs_layers(G, sources):
    if sources in G:
        sources = [sources]
    sources = list(sources)
    for source in sources:
        if source not in G:
            raise nx.NetworkXError(f"The node {source} is not in the graph.")
    seen = set(sources)
    layer = sources
    while layer:
        yield layer
        next_layer = []
        for node in layer:
            for child in G[node]:
                if child not in seen:
                    seen.add(child)
                    next_layer.append(child)
        layer = next_layer


def descendants_at_distance(G, source, distance):
    if source not in G:
        raise nx.NetworkXError(f"The node {source} is not in the graph.")
    if distance < 0:
        return set()
    for i, layer in enumerate(bfs_layers(G, source)):
        if i == distance:
            return set(layer)
    return set()


def _dfs(G, source, depth_limit=None, sort_neighbors=None):
    data = csr(G, sort_neighbors=sort_neighbors)
    source_i = -1 if source is None else require_node(data, source)
    n = len(data.nodes)
    limit = n if depth_limit is None else max(0, int(depth_limit))
    seen, stack_node, stack_edge, depth, pred, pre, post = (i64(n) for _ in range(7))
    count = lib().mnx_dfs(
        addr(data.indptr), addr(data.indices), n, source_i, limit, addr(seen),
        addr(stack_node), addr(stack_edge), addr(depth), addr(pred), addr(pre), addr(post),
    )
    return data, pre[:count], post[: int(depth[0])], pred


def dfs_edges(G, source=None, depth_limit=None, sort_neighbors=None):
    data, pre, _, pred = _dfs(G, source, depth_limit, sort_neighbors)
    for child in pre:
        if pred[child] >= 0:
            yield data.nodes[int(pred[child])], data.nodes[int(child)]


def dfs_tree(G, source=None, depth_limit=None, sort_neighbors=None):
    tree = nx.DiGraph()
    tree.add_nodes_from(G if source is None else [source])
    tree.add_edges_from(dfs_edges(G, source, depth_limit, sort_neighbors))
    return tree


def dfs_preorder_nodes(G, source=None, depth_limit=None, sort_neighbors=None):
    data, pre, _, _ = _dfs(G, source, depth_limit, sort_neighbors)
    return (data.nodes[int(i)] for i in pre)


def dfs_postorder_nodes(G, source=None, depth_limit=None, sort_neighbors=None):
    data, _, post, _ = _dfs(G, source, depth_limit, sort_neighbors)
    return (data.nodes[int(i)] for i in post)
