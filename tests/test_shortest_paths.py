import networkx as nx
import pytest

import mojonetworkx as mnx


@pytest.fixture
def graph():
    G = nx.DiGraph()
    G.add_weighted_edges_from(
        [
            ("s", "a", 2.0),
            ("s", "b", 5.0),
            ("a", "b", 1.0),
            ("a", "c", 6.0),
            ("b", "c", 2.0),
            ("c", "t", 1.5),
            ("b", "t", 9.0),
        ]
    )
    G.add_node("isolated")
    return G


@pytest.mark.parametrize("cutoff", [None, 0, 1, 3])
def test_unweighted_single_source_parity(graph, cutoff):
    assert mnx.single_source_shortest_path_length(
        graph, "s", cutoff
    ) == dict(nx.single_source_shortest_path_length(graph, "s", cutoff))
    assert mnx.single_source_shortest_path(graph, "s", cutoff) == nx.single_source_shortest_path(
        graph, "s", cutoff
    )


def test_target_and_all_pairs_parity(graph):
    assert mnx.single_target_shortest_path_length(
        graph, "t"
    ) == dict(nx.single_target_shortest_path_length(graph, "t"))
    assert mnx.single_target_shortest_path(graph, "t") == nx.single_target_shortest_path(
        graph, "t"
    )
    assert dict(mnx.all_pairs_shortest_path_length(graph)) == dict(
        nx.all_pairs_shortest_path_length(graph)
    )
    assert dict(mnx.all_pairs_shortest_path(graph)) == dict(nx.all_pairs_shortest_path(graph))


@pytest.mark.parametrize("cutoff", [None, 3.0, 6.0])
def test_dijkstra_parity(graph, cutoff):
    ours_dist, ours_paths = mnx.single_source_dijkstra(
        graph, "s", cutoff=cutoff, weight="weight"
    )
    ref_dist, ref_paths = nx.single_source_dijkstra(
        graph, "s", cutoff=cutoff, weight="weight"
    )
    assert ours_dist == pytest.approx(ref_dist)
    assert ours_paths == ref_paths
    assert mnx.single_source_dijkstra_path_length(
        graph, "s", cutoff, "weight"
    ) == pytest.approx(nx.single_source_dijkstra_path_length(graph, "s", cutoff, "weight"))
    assert mnx.single_source_dijkstra_path(
        graph, "s", cutoff, "weight"
    ) == nx.single_source_dijkstra_path(graph, "s", cutoff, "weight")


def test_dijkstra_target_predecessors_and_multigraph():
    G = nx.MultiGraph()
    G.add_edge("a", "b", weight=9)
    G.add_edge("a", "b", weight=2)
    G.add_edge("b", "c", weight=3)
    G.add_edge("a", "c", weight=20)
    assert mnx.dijkstra_path(G, "a", "c") == nx.dijkstra_path(G, "a", "c")
    assert mnx.dijkstra_path_length(G, "a", "c") == nx.dijkstra_path_length(G, "a", "c")
    assert mnx.dijkstra_predecessor_and_distance(G, "a") == nx.dijkstra_predecessor_and_distance(
        G, "a"
    )


def test_all_pairs_dijkstra_apis(graph):
    ours_lengths = dict(mnx.all_pairs_dijkstra_path_length(graph))
    reference_lengths = dict(nx.all_pairs_dijkstra_path_length(graph))
    assert ours_lengths.keys() == reference_lengths.keys()
    for node in ours_lengths:
        assert ours_lengths[node] == pytest.approx(reference_lengths[node])
    assert dict(mnx.all_pairs_dijkstra_path(graph)) == dict(nx.all_pairs_dijkstra_path(graph))
    ours = dict(mnx.all_pairs_dijkstra(graph))
    reference = dict(nx.all_pairs_dijkstra(graph))
    assert ours.keys() == reference.keys()
    for node in ours:
        assert ours[node][0] == pytest.approx(reference[node][0])
        assert ours[node][1] == reference[node][1]


def test_generic_shortest_path_contract(graph):
    assert mnx.shortest_path(graph, "s", "t") == nx.shortest_path(graph, "s", "t")
    assert mnx.shortest_path_length(graph, "s", "t") == nx.shortest_path_length(
        graph, "s", "t"
    )
    assert mnx.shortest_path(graph, "s", "t", weight="weight") == nx.shortest_path(
        graph, "s", "t", weight="weight"
    )
    assert mnx.shortest_path_length(
        graph, "s", "t", weight="weight"
    ) == pytest.approx(nx.shortest_path_length(graph, "s", "t", weight="weight"))
    assert mnx.has_path(graph, "s", "t")
    assert not mnx.has_path(graph, "s", "isolated")


def test_average_shortest_path_length():
    G = nx.cycle_graph(11)
    assert mnx.average_shortest_path_length(G) == nx.average_shortest_path_length(G)
    for edge in G.edges:
        G.edges[edge]["cost"] = 0.5
    assert mnx.average_shortest_path_length(G, "cost") == pytest.approx(
        nx.average_shortest_path_length(G, "cost")
    )


def test_dense_integer_equivalent_source():
    G = nx.path_graph(5)
    assert mnx.single_source_shortest_path_length(
        G, True
    ) == nx.single_source_shortest_path_length(G, True)


def test_negative_weight_rejected(graph):
    graph.add_edge("t", "x", weight=-1)
    with pytest.raises(ValueError):
        mnx.single_source_dijkstra_path_length(graph, "s")


@pytest.mark.parametrize("weight", [2**53 + 1, float("nan"), float("inf")])
def test_unrepresentable_weight_rejected(weight):
    G = nx.Graph()
    G.add_edge("a", "b", weight=weight)
    with pytest.raises((TypeError, ValueError), match="edge weight"):
        mnx.single_source_dijkstra_path_length(G, "a")


def test_bellman_ford_delegation():
    G = nx.DiGraph([(0, 1, {"cost": -2}), (1, 2, {"cost": 3}), (0, 2, {"cost": 5})])
    assert mnx.shortest_path(G, 0, 2, "cost", "bellman-ford") == nx.shortest_path(
        G, 0, 2, "cost", "bellman-ford"
    )
    assert mnx.shortest_path_length(G, 0, 2, "cost", "bellman-ford") == nx.shortest_path_length(
        G, 0, 2, "cost", "bellman-ford"
    )
