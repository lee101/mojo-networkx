"""NetworkX-compatible graph algorithms accelerated by Mojo."""

from networkx import (
    DiGraph,
    Graph,
    MultiDiGraph,
    MultiGraph,
    NetworkXError,
    NetworkXNoPath,
    NodeNotFound,
)

from .centrality import (
    betweenness_centrality,
    closeness_centrality,
    degree_centrality,
    eigenvector_centrality,
    in_degree_centrality,
    out_degree_centrality,
    pagerank,
)
from .components import (
    connected_components,
    is_connected,
    is_strongly_connected,
    is_weakly_connected,
    node_connected_component,
    number_connected_components,
    number_strongly_connected_components,
    number_weakly_connected_components,
    strongly_connected_components,
    weakly_connected_components,
)
from .shortest_paths import (
    all_pairs_dijkstra,
    all_pairs_dijkstra_path,
    all_pairs_dijkstra_path_length,
    all_pairs_shortest_path,
    all_pairs_shortest_path_length,
    average_shortest_path_length,
    dijkstra_path,
    dijkstra_path_length,
    dijkstra_predecessor_and_distance,
    has_path,
    shortest_path,
    shortest_path_length,
    single_source_dijkstra,
    single_source_dijkstra_path,
    single_source_dijkstra_path_length,
    single_source_shortest_path,
    single_source_shortest_path_length,
    single_target_shortest_path,
    single_target_shortest_path_length,
)
from .traversal import (
    bfs_edges,
    bfs_layers,
    bfs_predecessors,
    bfs_successors,
    bfs_tree,
    descendants_at_distance,
    dfs_edges,
    dfs_postorder_nodes,
    dfs_preorder_nodes,
    dfs_tree,
)

__version__ = "0.1.0"
