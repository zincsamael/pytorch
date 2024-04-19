# Owner(s): ["module: sparse"]
#
# Test to ensure sparsity information propagates properly into traced graph.
#

import sys
import unittest
from typing import Optional, Tuple

import torch

from torch._subclasses.fake_tensor import FakeTensor

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


# Common sparse data dtypes currently supported in torch.sparse.
SPARSE_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
]

# Common sparse index dtypes currently supported in torch.sparse.
SPARSE_ITYPES = [
    torch.int32,
    torch.int64,
]

# Sparse layouts currently supported in torch.sparse.
SPARSE_LAYOUTS = [
    torch.sparse_coo,
    torch.sparse_csr,
    torch.sparse_csc,
    torch.sparse_bsr,
    torch.sparse_bsc,
]


# Returns expected values.
def expected(
    t: torch.Tensor, itype: torch.dtype
) -> Tuple[int, Optional[Tuple[int, int]], torch.dtype]:
    # Compute batch dimension.
    batch_dim = t.ndim - t.sparse_dim() - t.dense_dim()
    # Determine block size from values shape.
    if t.layout in {torch.sparse_bsr, torch.sparse_bsc}:
        blocksize = t.values().shape[batch_dim + 1 : batch_dim + 3]
    else:
        blocksize = None
    # COO always makes the indices int64.
    index_type = torch.int64 if torch.sparse_coo else itype
    # Return expected values.
    return (batch_dim, blocksize, index_type)


# Ensures we can extract sparsity information from stored meta data.
def extract_sparse_tensor_metadata(
    t: torch.Tensor,
) -> Tuple[int, int, int, Optional[Tuple[int, int]], Optional[torch.dtype]]:
    batch_dim = t.ndim - t.dense_dim() - t.sparse_dim()
    # Set block size.
    if t.layout is torch.sparse_bsr or t.layout is torch.sparse_bsc:
        blocksize = t.values().shape[batch_dim + 1 : batch_dim + 3]
    else:
        blocksize = None
    # Set index type.
    if t.layout is torch.sparse_coo:
        idx_dtype = t._indices().dtype  # supports uncoalesced COO tensors
    elif t.layout is torch.sparse_csr or t.layout is torch.sparse_bsr:
        idx_dtype = t.col_indices().dtype
    else:
        idx_dtype = t.row_indices().dtype
    # Return sparse metadata.
    return (batch_dim, t.sparse_dim(), t.dense_dim(), blocksize, idx_dtype)


class IdNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class SumNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.sum()


class EltwiseNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(2 * torch.sin(-x))


class TestSparseProp(TestCase):
    def setUp(self):
        TestCase.setUp(self)

    @unittest.skipIf(
        sys.version_info >= (3, 12), "torch.compile is not supported on python 3.12+"
    )
    @parametrize("dtype", SPARSE_DTYPES)
    @parametrize("itype", SPARSE_ITYPES)
    @parametrize("layout", SPARSE_LAYOUTS)
    def test_idnet(self, dtype, itype, layout):
        if layout is not torch.sparse_coo:
            self.skipTest("TODO: support non-coo sparsity!")

        net = IdNet()
        for sparse_input in self.generate_simple_inputs(
            layout,
            device="cpu",
            dtype=dtype,
            index_dtype=itype,
        ):
            (batch_dim, blocksize, index_type) = expected(sparse_input, itype)
            # Build the traced graph.
            prog = torch.export.export(net, (sparse_input,))
            # Test arg/output.
            for i, node in enumerate(prog.graph.nodes):
                meta = node.meta.get("val", None)
                if i == 0:
                    self.assertIsInstance(meta, torch.Tensor)
                    self.assertEqual(meta.layout, layout)
                    self.assertEqual(meta.dtype, dtype)
                    (b, s, d, bsz, itp) = extract_sparse_tensor_metadata(meta)
                    self.assertEqual(b, batch_dim)
                    self.assertEqual(s, sparse_input.sparse_dim())
                    self.assertEqual(d, sparse_input.dense_dim())
                    self.assertEqual(bsz, blocksize)
                    self.assertEqual(itp, index_type)
                else:
                    self.assertEqual(meta, None)

    @unittest.skipIf(
        sys.version_info >= (3, 12), "torch.compile is not supported on python 3.12+"
    )
    @parametrize("dtype", SPARSE_DTYPES)
    @parametrize("itype", SPARSE_ITYPES)
    @parametrize("layout", SPARSE_LAYOUTS)
    def test_sumnet(self, dtype, itype, layout):
        if layout is not torch.sparse_coo:
            self.skipTest("TODO: support non-coo sparsity!")

        net = SumNet()
        for sparse_input in self.generate_simple_inputs(
            layout,
            device="cpu",
            dtype=dtype,
            index_dtype=itype,
        ):
            (batch_dim, blocksize, index_type) = expected(sparse_input, itype)
            # Build the traced graph.
            prog = torch.export.export(net, (sparse_input,))
            # Test arg/sum/output.
            for i, node in enumerate(prog.graph.nodes):
                meta = node.meta.get("val", None)
                if i == 0:
                    self.assertIsInstance(meta, torch.Tensor)
                    self.assertEqual(meta.layout, layout)
                    self.assertEqual(meta.dtype, dtype)
                    (b, s, d, bsz, itp) = extract_sparse_tensor_metadata(meta)
                    self.assertEqual(b, batch_dim)
                    self.assertEqual(s, sparse_input.sparse_dim())
                    self.assertEqual(d, sparse_input.dense_dim())
                    self.assertEqual(bsz, blocksize)
                    self.assertEqual(itp, index_type)
                elif i == 1:
                    self.assertIsInstance(meta, FakeTensor)
                    self.assertEqual(meta.layout, torch.strided)
                    self.assertEqual(meta.dtype, dtype)
                else:
                    self.assertEqual(meta, None)

    @unittest.skipIf(
        sys.version_info >= (3, 12), "torch.compile is not supported on python 3.12+"
    )
    @parametrize("dtype", SPARSE_DTYPES)
    @parametrize("itype", SPARSE_ITYPES)
    @parametrize("layout", SPARSE_LAYOUTS)
    def test_eltwisenet(self, dtype, itype, layout):
        if layout is not torch.sparse_coo:
            self.skipTest("TODO: support non-coo sparsity!")

        net = EltwiseNet()
        for sparse_input in self.generate_simple_inputs(
            layout,
            device="cpu",
            dtype=dtype,
            index_dtype=itype,
        ):
            (batch_dim, blocksize, index_type) = expected(sparse_input, itype)
            # Build the traced graph.
            prog = torch.export.export(net, (sparse_input,))
            # Test arg/neg/sin/mul/relu/output.
            for i, node in enumerate(prog.graph.nodes):
                meta = node.meta.get("val", None)
                if i <= 4:
                    self.assertIsInstance(meta, torch.Tensor if i == 0 else FakeTensor)
                    self.assertEqual(meta.layout, layout)
                    self.assertEqual(meta.dtype, dtype)
                    (b, s, d, bsz, itp) = extract_sparse_tensor_metadata(meta)
                    self.assertEqual(b, batch_dim)
                    self.assertEqual(s, sparse_input.sparse_dim())
                    self.assertEqual(d, sparse_input.dense_dim())
                    self.assertEqual(bsz, blocksize)
                    self.assertEqual(itp, index_type)
                else:
                    self.assertEqual(meta, None)


instantiate_parametrized_tests(TestSparseProp)

if __name__ == "__main__":
    run_tests()
