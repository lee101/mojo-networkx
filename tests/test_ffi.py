import numpy as np
import pytest

from mojonetworkx._graph import csr
from mojonetworkx._lib import addr, f64, i64, lib

import networkx as nx


@pytest.mark.parametrize("factory", [i64, f64])
def test_empty_ffi_buffers_keep_non_null_owner(factory):
    array = factory(0)
    assert array.shape == (0,)
    assert addr(array) != 0


def test_edgeless_csr_buffers_are_non_null_and_typed():
    data = csr(nx.empty_graph(3))
    assert data.indptr.dtype == np.int64
    assert data.indices.dtype == np.int64
    assert data.weights.dtype == np.float64
    assert addr(data.indices) != 0
    assert addr(data.weights) != 0


@pytest.mark.parametrize(
    "array, error",
    [
        (np.empty(2, dtype=np.int32), TypeError),
        (np.empty((2, 2), dtype=np.int64), TypeError),
        (np.empty(4, dtype=np.int64)[::2], ValueError),
    ],
)
def test_ffi_rejects_wrong_dtype_shape_and_stride(array, error):
    with pytest.raises(error):
        addr(array)


@pytest.mark.parametrize("n", [17, 262_145])
def test_dijkstra_simd_tail_and_parallel_initialization(n):
    indptr = i64(n + 1)
    indptr.fill(0)
    indices = i64(0)
    weights = f64(0)
    dist = f64(n)
    workspace = i64(4 * n)
    pred, heap, pos, order = (workspace[i * n:(i + 1) * n] for i in range(4))
    count = lib().mnx_dijkstra(
        addr(indptr), addr(indices), addr(weights), n, 0, float("inf"),
        addr(dist), addr(pred), addr(heap), addr(pos), addr(order),
    )
    assert count == 1
    assert dist[0] == 0.0
    assert np.all(dist[1:] == np.finfo(np.float64).max)
    assert np.all(pred == -1)
    assert pos[0] == -2
    assert np.all(pos[1:] == -1)
