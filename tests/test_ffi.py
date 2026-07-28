import numpy as np
import pytest

from mojonetworkx._graph import csr
from mojonetworkx._lib import addr, f64, i64

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
