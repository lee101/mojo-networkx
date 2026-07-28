"""Honest end-to-end timings against NetworkX, including CSR conversion."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import networkx as nx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojonetworkx as mnx  # noqa: E402


def timeit(fn, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


CASES = []


def case(name):
    def decorate(builder):
        CASES.append((name, builder))
        return builder
    return decorate


@case("bfs_edges (BA, 100k nodes / 300k edges)")
def _():
    G = nx.barabasi_albert_graph(100_000, 3, seed=1)
    return lambda: list(mnx.bfs_edges(G, 0)), lambda: list(nx.bfs_edges(G, 0))


@case("shortest path lengths (BA, 100k / 300k)")
def _():
    G = nx.barabasi_albert_graph(100_000, 3, seed=2)
    return (
        lambda: mnx.single_source_shortest_path_length(G, 0),
        lambda: dict(nx.single_source_shortest_path_length(G, 0)),
    )


@case("Dijkstra (BA, 50k / 150k weighted)")
def _():
    G = nx.barabasi_albert_graph(50_000, 3, seed=3)
    for i, edge in enumerate(G.edges):
        G.edges[edge]["weight"] = 1.0 + (i % 19) * 0.05
    return (
        lambda: mnx.single_source_dijkstra_path_length(G, 0),
        lambda: nx.single_source_dijkstra_path_length(G, 0),
    )


@case("connected components (100k / 200k)")
def _():
    graphs = [nx.barabasi_albert_graph(5_000, 2, seed=i) for i in range(20)]
    G = nx.disjoint_union_all(graphs)
    return (
        lambda: list(mnx.connected_components(G)),
        lambda: list(nx.connected_components(G)),
    )


@case("betweenness centrality (BA, 800 / 2.4k)")
def _():
    G = nx.barabasi_albert_graph(800, 3, seed=4)
    return lambda: mnx.betweenness_centrality(G), lambda: nx.betweenness_centrality(G)


@case("closeness centrality (BA, 2k / 4k)")
def _():
    G = nx.barabasi_albert_graph(2_000, 2, seed=5)
    return lambda: mnx.closeness_centrality(G), lambda: nx.closeness_centrality(G)


@case("PageRank (directed BA, 100k / 600k arcs)")
def _():
    G = nx.barabasi_albert_graph(100_000, 3, seed=6).to_directed()
    return lambda: mnx.pagerank(G), lambda: nx.pagerank(G)


@case("eigenvector centrality (BA, 100k / 300k)")
def _():
    G = nx.barabasi_albert_graph(100_000, 3, seed=7)
    return (
        lambda: mnx.eigenvector_centrality(G, max_iter=500),
        lambda: nx.eigenvector_centrality(G, max_iter=500),
    )


def machine():
    model = platform.processor()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            model = next(
                line.split(":", 1)[1].strip()
                for line in handle
                if line.startswith("model name")
            )
    except (OSError, StopIteration):
        pass
    return f"{model}; {platform.system()} {platform.release()}; Python {platform.python_version()}"


def main():
    print(f"Machine: {machine()}")
    print(f"NetworkX: {nx.__version__}")
    print()
    print("| case | mojo-networkx | NetworkX | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, builder in CASES:
        ours, reference = builder()
        ours()
        a = timeit(ours)
        b = timeit(reference)
        ratio = b / a
        label = f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower"
        print(f"| {name} | {a * 1e3:.1f} ms | {b * 1e3:.1f} ms | {label} |")


if __name__ == "__main__":
    main()
