import networkx as nx
import pytest

import mojonetworkx as mnx


def canonical(components):
    return {frozenset(component) for component in components}


def test_undirected_components_parity():
    G = nx.Graph([(0, 1), (1, 2), (4, 5), (7, 8), (8, 9)])
    G.add_nodes_from([3, 6, 10])
    assert canonical(mnx.connected_components(G)) == canonical(nx.connected_components(G))
    assert mnx.number_connected_components(G) == nx.number_connected_components(G)
    assert mnx.node_connected_component(G, 8) == nx.node_connected_component(G, 8)
    assert mnx.is_connected(G) == nx.is_connected(G)
    assert mnx.is_connected(nx.path_graph(20))


def test_directed_components_parity():
    G = nx.DiGraph(
        [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 3), (5, 4), (6, 7)]
    )
    G.add_node(8)
    assert canonical(mnx.weakly_connected_components(G)) == canonical(
        nx.weakly_connected_components(G)
    )
    assert canonical(mnx.strongly_connected_components(G)) == canonical(
        nx.strongly_connected_components(G)
    )
    assert mnx.number_weakly_connected_components(G) == nx.number_weakly_connected_components(G)
    assert mnx.number_strongly_connected_components(
        G
    ) == nx.number_strongly_connected_components(G)
    assert mnx.is_weakly_connected(G) == nx.is_weakly_connected(G)
    assert mnx.is_strongly_connected(G) == nx.is_strongly_connected(G)


def test_component_errors():
    with pytest.raises(nx.NetworkXNotImplemented):
        list(mnx.connected_components(nx.DiGraph()))
    with pytest.raises(nx.NetworkXNotImplemented):
        list(mnx.strongly_connected_components(nx.Graph()))
    with pytest.raises(nx.NetworkXPointlessConcept):
        mnx.is_connected(nx.Graph())
    with pytest.raises(nx.NetworkXPointlessConcept):
        mnx.is_strongly_connected(nx.DiGraph())
    with pytest.raises(nx.NetworkXPointlessConcept):
        mnx.is_weakly_connected(nx.DiGraph())
