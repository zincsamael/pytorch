# Owner(s): ["module: sparse"]

import torch

from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
)
from typing import Optional, Tuple


# Common sparse data dtypes currently supported in torch.sparse.
SPARSE_DTYPES = [
    torch.int8,
    torch.int16,
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


class SumNet(torch.nn.Module):
    def __init__(self):
        super(SumNet, self).__init__()
        return

    def forward(self, x):
        return x.sum()


class EltwiseNet(torch.nn.Module):
    def __init__(self):
        super(EltwiseNet, self).__init__()
        return

    def forward(self, x):
        return 2 * torch.sin(-x)


class TestSparseProp(TestCase):
    def setUp(self):
        TestCase.setUp(self)

    @parametrize("dtype", SPARSE_DTYPES)
    @parametrize("itype", SPARSE_ITYPES)
    @parametrize("layout", SPARSE_LAYOUTS)
    def test_sumnet(self, dtype, itype, layout):
        # TODO: support more cases
        if layout != torch.sparse_coo:
            self.skipTest(f"layout support not yet implemented")
        if layout == torch.sparse_coo and itype != torch.int64:
            self.skipTest(f"COO only supports int64 index type")

        net = SumNet()
        for sparse_input in self.generate_simple_inputs(
            layout,
            device="cpu",
            dtype=dtype,
            index_dtype=itype,
        ):
            batch_dim = (
                sparse_input.ndim - sparse_input.sparse_dim() - sparse_input.dense_dim()
            )
            if layout in {torch.sparse_bsr, torch.sparse_bsc}:
                blocksize = sparse_input.values().shape[batch_dim + 1 : batch_dim + 3]
            else:
                blocksize = None
            # Build the trace graph.
            prog = torch.export.export(net, (sparse_input,))
            # Test arg/sum/output.
            for i, node in enumerate(prog.graph.nodes):
                meta = node.meta.get("tensor_meta", None)
                if i == 0:
                    self.assertEqual(meta.layout, layout)
                    self.assertEqual(meta.dtype, dtype)
                    self.assertEqual(meta.batch_dim, batch_dim)
                    self.assertEqual(meta.sparse_dim, sparse_input.sparse_dim())
                    self.assertEqual(meta.dense_dim, sparse_input.dense_dim())
                    self.assertEqual(meta.blocksize, blocksize)
                    self.assertEqual(meta.idx_dtype, itype)
                elif i == 1:
                    self.assertEqual(meta.layout, torch.strided)
                    self.assertEqual(meta.sparse_dim, None)
                else:
                    self.assertEqual(meta, None)

    @parametrize("dtype", SPARSE_DTYPES)
    @parametrize("itype", SPARSE_ITYPES)
    @parametrize("layout", SPARSE_LAYOUTS)
    def test_eltwisenet(self, dtype, itype, layout):
        # TODO: support more cases
        if layout != torch.sparse_coo:
            self.skipTest(f"layout support not yet implemented")
        if layout == torch.sparse_coo and itype != torch.int64:
            self.skipTest(f"COO only supports int64 index type")

        net = EltwiseNet()
        for sparse_input in self.generate_simple_inputs(
            layout,
            device="cpu",
            dtype=dtype,
            index_dtype=itype,
        ):
            batch_dim = (
                sparse_input.ndim - sparse_input.sparse_dim() - sparse_input.dense_dim()
            )
            if layout in {torch.sparse_bsr, torch.sparse_bsc}:
                blocksize = sparse_input.values().shape[batch_dim + 1 : batch_dim + 3]
            else:
                blocksize = None
            # Build the trace graph.
            prog = torch.export.export(net, (sparse_input,))
            # Test arg/neg/sin/mul/output.
            for i, node in enumerate(prog.graph.nodes):
                meta = node.meta.get("tensor_meta", None)
                if i <= 3:
                    self.assertEqual(meta.layout, layout)
                    self.assertEqual(meta.dtype, dtype)
                    self.assertEqual(meta.batch_dim, batch_dim)
                    self.assertEqual(meta.sparse_dim, sparse_input.sparse_dim())
                    self.assertEqual(meta.dense_dim, sparse_input.dense_dim())
                    self.assertEqual(meta.blocksize, blocksize)
                    self.assertEqual(meta.idx_dtype, itype)
                else:
                    self.assertEqual(meta, None)


instantiate_parametrized_tests(TestSparseProp)

if __name__ == "__main__":
    run_tests()
