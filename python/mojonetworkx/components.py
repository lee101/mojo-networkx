from __future__ import annotations

import networkx as nx
import numpy as np

from ._graph import csr
from ._lib import addr, i64, lib


def _labels(G, weak=False):
    data = csr(G, undirected=weak)
    n = len(data.nodes)
    workspace = i64(2 * n)
    labels, queue = workspace[:n], workspace[n:]
    count = lib().mnx_components(
        addr(data.indptr), addr(data.indices), n, addr(labels), addr(queue)
    )
    return data.nodes, labels, count


def connected_components(G):
    if G.is_directed():
        raise nx.NetworkXNotImplemented("not implemented for directed type")
    nodes, labels, count = _labels(G)
    return ({nodes[i] for i in np.flatnonzero(labels == component)} for component in range(count))


def number_connected_components(G):
    return sum(1 for _ in connected_components(G))


def node_connected_component(G, n):
    if n not in G:
        raise nx.NodeNotFound(f"Node {n} is not in G")
    for component in connected_components(G):
        if n in component:
            return component
    raise AssertionError("unreachable")


def is_connected(G):
    if G.is_directed():
        raise nx.NetworkXNotImplemented("not implemented for directed type")
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept("Connectivity is undefined for the null graph.")
    return number_connected_components(G) == 1


def weakly_connected_components(G):
    if not G.is_directed():
        raise nx.NetworkXNotImplemented("not implemented for undirected type")
    nodes, labels, count = _labels(G, weak=True)
    return ({nodes[i] for i in np.flatnonzero(labels == component)} for component in range(count))


def number_weakly_connected_components(G):
    return sum(1 for _ in weakly_connected_components(G))


def is_weakly_connected(G):
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept("Connectivity is undefined for the null graph.")
    return number_weakly_connected_components(G) == 1


def strongly_connected_components(G):
    if not G.is_directed():
        raise nx.NetworkXNotImplemented("not implemented for undirected type")
    forward = csr(G)
    reverse = csr(G, reverse=True)
    n = len(forward.nodes)
    labels, seen, order, stack_node, stack_edge = (i64(n) for _ in range(5))
    count = lib().mnx_scc(
        addr(forward.indptr), addr(forward.indices), addr(reverse.indptr),
        addr(reverse.indices), n, addr(labels), addr(seen), addr(order),
        addr(stack_node), addr(stack_edge),
    )
    return (
        {forward.nodes[i] for i in np.flatnonzero(labels == component)}
        for component in range(count)
    )


def number_strongly_connected_components(G):
    return sum(1 for _ in strongly_connected_components(G))


def is_strongly_connected(G):
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept("Connectivity is undefined for the null graph.")
    return number_strongly_connected_components(G) == 1
