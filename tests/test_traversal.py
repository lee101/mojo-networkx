import networkx as nx
import pytest

import mojonetworkx as mnx


@pytest.fixture
def graph():
    G = nx.Graph()
    G.add_edges_from(
        [("root", "b"), ("root", "a"), ("a", "c"), ("b", "d"), ("c", "d"), ("d", "e")]
    )
    G.add_node("isolated")
    return G


@pytest.mark.parametrize("depth", [None, 0, 1, 3])
def test_bfs_edges_parity(graph, depth):
    assert list(mnx.bfs_edges(graph, "root", depth_limit=depth)) == list(
        nx.bfs_edges(graph, "root", depth_limit=depth)
    )


def test_bfs_sorted_and_derived_apis(graph):
    sorter = lambda values: sorted(values, reverse=True)
    assert list(mnx.bfs_edges(graph, "root", sort_neighbors=sorter)) == list(
        nx.bfs_edges(graph, "root", sort_neighbors=sorter)
    )
    assert list(mnx.bfs_predecessors(graph, "root")) == list(
        nx.bfs_predecessors(graph, "root")
    )
    assert list(mnx.bfs_successors(graph, "root")) == list(nx.bfs_successors(graph, "root"))
    assert list(mnx.bfs_tree(graph, "root").edges()) == list(nx.bfs_tree(graph, "root").edges())
    assert list(mnx.bfs_layers(graph, "root")) == list(nx.bfs_layers(graph, "root"))
    for distance in [-1, 0, 2, 9]:
        assert mnx.descendants_at_distance(graph, "root", distance) == nx.descendants_at_distance(
            graph, "root", distance
        )


def test_reverse_bfs_parity():
    G = nx.DiGraph([("a", "b"), ("c", "b"), ("b", "d"), ("d", "e")])
    assert list(mnx.bfs_edges(G, "d", reverse=True)) == list(
        nx.bfs_edges(G, "d", reverse=True)
    )


@pytest.mark.parametrize("depth", [None, 0, 1, 4])
def test_dfs_parity(graph, depth):
    assert list(mnx.dfs_edges(graph, "root", depth_limit=depth)) == list(
        nx.dfs_edges(graph, "root", depth_limit=depth)
    )
    assert list(mnx.dfs_preorder_nodes(graph, "root", depth_limit=depth)) == list(
        nx.dfs_preorder_nodes(graph, "root", depth_limit=depth)
    )
    assert list(mnx.dfs_postorder_nodes(graph, "root", depth_limit=depth)) == list(
        nx.dfs_postorder_nodes(graph, "root", depth_limit=depth)
    )
    assert list(mnx.dfs_tree(graph, "root", depth_limit=depth).edges) == list(
        nx.dfs_tree(graph, "root", depth_limit=depth).edges
    )


def test_missing_source(graph):
    with pytest.raises(nx.NodeNotFound):
        list(mnx.bfs_edges(graph, "missing"))
    with pytest.raises(nx.NodeNotFound):
        list(mnx.dfs_edges(graph, "missing"))


def test_dfs_forest_directed():
    G = nx.DiGraph([(0, 1), (2, 1), (3, 4)])
    G.add_node(5)
    assert list(mnx.dfs_edges(G)) == list(nx.dfs_edges(G))
    assert list(mnx.dfs_preorder_nodes(G)) == list(nx.dfs_preorder_nodes(G))
    assert list(mnx.dfs_postorder_nodes(G)) == list(nx.dfs_postorder_nodes(G))
