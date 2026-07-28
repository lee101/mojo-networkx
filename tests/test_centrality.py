import networkx as nx
import pytest

import mojonetworkx as mnx


def assert_scores(ours, reference, tolerance=1e-12):
    assert ours.keys() == reference.keys()
    for node in ours:
        assert ours[node] == pytest.approx(reference[node], abs=tolerance, rel=tolerance)


@pytest.fixture(scope="module")
def graph():
    return nx.karate_club_graph()


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("endpoints", [False, True])
def test_betweenness_parity(graph, normalized, endpoints):
    assert_scores(
        mnx.betweenness_centrality(graph, normalized=normalized, endpoints=endpoints),
        nx.betweenness_centrality(graph, normalized=normalized, endpoints=endpoints),
    )


def test_weighted_betweenness_parity():
    G = nx.gnp_random_graph(24, 0.18, seed=4)
    for i, edge in enumerate(G.edges):
        G.edges[edge]["length"] = 1.0 + (i % 7) / 10
    assert_scores(
        mnx.betweenness_centrality(G, weight="length"),
        nx.betweenness_centrality(G, weight="length"),
        1e-10,
    )


def test_closeness_parity(graph):
    assert_scores(mnx.closeness_centrality(graph), nx.closeness_centrality(graph))
    assert mnx.closeness_centrality(graph, 0) == pytest.approx(nx.closeness_centrality(graph, 0))


def test_directed_weighted_closeness_parity():
    G = nx.DiGraph()
    G.add_weighted_edges_from([(0, 1, 1.5), (1, 2, 2), (3, 2, 1), (2, 4, 0.5)])
    assert_scores(
        mnx.closeness_centrality(G, distance="weight"),
        nx.closeness_centrality(G, distance="weight"),
    )


def test_degree_centralities(graph):
    assert_scores(mnx.degree_centrality(graph), nx.degree_centrality(graph))
    G = nx.gn_graph(30, seed=2)
    assert_scores(mnx.in_degree_centrality(G), nx.in_degree_centrality(G))
    assert_scores(mnx.out_degree_centrality(G), nx.out_degree_centrality(G))
    with pytest.raises(nx.NetworkXNotImplemented):
        mnx.in_degree_centrality(graph)
    with pytest.raises(nx.NetworkXNotImplemented):
        mnx.out_degree_centrality(graph)


def test_pagerank_parity():
    G = nx.gn_graph(83, seed=3)
    assert_scores(mnx.pagerank(G), nx.pagerank(G), 1e-10)
    M = nx.MultiDiGraph(G)
    M.add_edge(3, 0, weight=4.0)
    assert_scores(mnx.pagerank(M), nx.pagerank(M), 1e-10)


def test_eigenvector_parity(graph):
    assert_scores(
        mnx.eigenvector_centrality(graph, max_iter=500),
        nx.eigenvector_centrality(graph, max_iter=500),
        1e-10,
    )


def test_null_and_singleton_cases():
    assert mnx.degree_centrality(nx.empty_graph(1)) == nx.degree_centrality(nx.empty_graph(1))
    assert mnx.pagerank(nx.Graph()) == {}
    with pytest.raises(nx.NetworkXPointlessConcept):
        mnx.eigenvector_centrality(nx.Graph())


def test_explicitly_unsupported_options():
    G = nx.path_graph(4)
    with pytest.raises(NotImplementedError):
        mnx.betweenness_centrality(G, k=2)
    with pytest.raises(NotImplementedError):
        mnx.pagerank(G, personalization={node: 1 for node in G})
    with pytest.raises(NotImplementedError):
        mnx.eigenvector_centrality(G, nstart={node: 1 for node in G})
