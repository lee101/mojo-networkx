# mojo-networkx

Graph traversal, shortest paths, connectivity, and centrality kernels implemented
in [Mojo](https://www.modular.com/mojo), with a Python API matching the covered
part of [NetworkX](https://networkx.org/).

The package accepts normal NetworkX `Graph`, `DiGraph`, `MultiGraph`, and
`MultiDiGraph` objects, including arbitrary hashable node labels. Graph storage
and mutation stay with NetworkX; the compute-heavy algorithm loops run in a
single compiled Mojo shared library.

```python
import networkx as nx
import mojonetworkx as mnx

G = nx.Graph()
G.add_weighted_edges_from([
    ("A", "B", 1.5),
    ("B", "C", 2.0),
    ("A", "C", 10.0),
])

print(mnx.shortest_path(G, "A", "C", weight="weight"))
# ['A', 'B', 'C']
print(mnx.shortest_path_length(G, "A", "C", weight="weight"))
# 3.5
```

## Coverage

| area | covered API |
| --- | --- |
| breadth-first traversal | `bfs_edges`, `bfs_tree`, `bfs_predecessors`, `bfs_successors`, `bfs_layers`, `descendants_at_distance`; reverse, depth limits, and neighbor sorting |
| depth-first traversal | `dfs_edges`, `dfs_tree`, `dfs_preorder_nodes`, `dfs_postorder_nodes`; depth limits and neighbor sorting |
| unweighted shortest paths | single-source, single-target, and all-pairs path and path-length APIs |
| weighted shortest paths | Dijkstra paths, lengths, predecessors, single-source and all-pairs APIs; cutoff and callable/string weights |
| generic path API | `shortest_path`, `shortest_path_length`, `has_path`, `average_shortest_path_length` |
| components | connected, weakly connected, and strongly connected component iterators, counts, and predicates |
| centrality | unweighted and positive-weight Brandes betweenness, closeness, degree/in-degree/out-degree, PageRank, and eigenvector centrality |

Results retain NetworkX node labels and use its normal dictionaries, sets,
generators, and graph classes. The tests compare actual values, traversal order,
paths, endpoint normalization, directed behavior, exceptions, and weighted
results against NetworkX 3.6.1.

This is deliberately not a complete NetworkX replacement. It does not cover
graph construction and mutation, file formats, generators, layouts,
isomorphism, flows, matching, communities, link prediction, or the many
specialized algorithm modules. Bellman-Ford is accepted by the generic API but
delegates to NetworkX. Sampled betweenness (`k`), non-positive weighted
betweenness, custom PageRank personalization/start/dangling vectors, and custom
eigenvector start vectors are not accelerated and raise a clear error.
Accelerated weights must be finite values exactly representable as `float64`;
values that would silently narrow are rejected.

## Install

The repository pins its own Mojo nightly and Python environment:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` produces `dist/libmojo-networkx.so`. The Python wrapper also
builds it on first import if it is absent. Run the benchmark only through its
locked task:

```bash
pixi run bench
```

## Performance

End-to-end timings include converting NetworkX's dictionary-of-dictionaries
adjacency into CSR on every call. Dense integer labels stream directly into
the final NumPy buffers, while arbitrary labels retain the general mapping
path. Conversion remains a substantial part of one-pass algorithms.

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux 6.8.0-136-generic, Python 3.13.14:

| case | mojo-networkx | NetworkX | result |
| --- | ---: | ---: | ---: |
| `bfs_edges` (BA, 100k nodes / 300k edges) | 272.7 ms | 829.5 ms | 3.04x faster |
| shortest path lengths (BA, 100k / 300k) | 137.1 ms | 203.2 ms | 1.48x faster |
| Dijkstra (BA, 50k / 150k weighted) | 551.6 ms | 679.5 ms | 1.23x faster |
| connected components (100k / 200k) | 142.7 ms | 159.2 ms | 1.12x faster |
| betweenness centrality (BA, 800 / 2.4k) | 52.1 ms | 1762.1 ms | 33.80x faster |
| closeness centrality (BA, 2k / 4k) | 91.7 ms | 2279.3 ms | 24.84x faster |
| PageRank (directed BA, 100k / 600k arcs) | 1290.3 ms | 1151.5 ms | 1.12x slower |
| eigenvector centrality (BA, 100k / 300k) | 733.8 ms | 44857.0 ms | 61.13x faster |

These are honest whole-call numbers, not isolated kernel timings. PageRank was
slightly slower in this run; benchmark results vary with system load.

No GPU path is provided. The covered sparse graph kernels perform well below
two floating-point operations per byte moved, and their irregular gathers and
scatters make transfer and launch overhead a net loss.

## How it works

The Python layer maps arbitrary nodes to dense integer IDs and flattens
adjacency into three contiguous NumPy buffers: `int64` CSR offsets, `int64`
neighbor IDs, and `float64` weights. Parallel edges use their minimum weight
for shortest-path algorithms and their summed weight for PageRank and
eigenvector centrality, matching NetworkX.

One `ctypes` call passes buffer addresses and extents to
`src/kernels.mojo`. Exported functions use `@export("name")` with `abi("C")`;
buffers cross the ABI as integer addresses and are reconstructed as
`UnsafePointer[..., AnyOrigin[mut=True]]`. Queues, heaps, predecessor arrays,
component labels, Brandes dependencies, and iteration buffers are all
NumPy-owned scratch memory retained for the duration of each `ctypes` call, so
cross-boundary ownership and lifetimes remain unambiguous.

Traversal uses iterative queues/stacks, Dijkstra uses an indexed binary heap,
strong components use iterative Kosaraju, and betweenness uses Brandes'
dependency accumulation. PageRank and eigenvector centrality are sparse
power-iteration kernels. All kernels live in one compilation unit because the
fixed Mojo shared-library build cost is much larger than the marginal cost of
another exported function.

## License

MIT
