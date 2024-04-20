# Owner(s): ["module: sparse"]
#
# Test to ensure sparsity information propagates properly into traced graph.
#

import sys
import unittest

import torch

from torch._subclasses.fake_tensor import FakeTensor
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TestCase,
)

# Various data types (preserved over operations).
DTYPES = [
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
]

# Various index types.
ITYPES = [torch.int32, torch.int64]


# Constructs a subtest for every sparse layout currently supported in torch.sparse.
def all_sparse_layouts(test_name="layout"):
    return parametrize(
        test_name,
        [
            subtest(torch.sparse_coo, name="SparseCOO"),
            subtest(torch.sparse_csr, name="SparseCSR"),
            subtest(torch.sparse_csc, name="SparseCSC"),
            subtest(torch.sparse_bsr, name="SparseBSR"),
            subtest(torch.sparse_bsc, name="SparseBSC"),
        ],
    )


#
# Various network examples.
#


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
        return self.relu(2 * torch.abs(-x))


#
# The test driver.
#


class TestSparseProp(TestCase):
    def setUp(self):
        TestCase.setUp(self)

    @unittest.skipIf(
        sys.version_info >= (3, 12), "torch.compile is not supported on python 3.12+"
    )
    @parametrize("dtype", DTYPES)
    @parametrize("itype", ITYPES)
    @all_sparse_layouts("layout")
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
            # Build the traced graph.
            prog = torch.export.export(net, (sparse_input,))
            # Test arg/output.
            for i, node in enumerate(prog.graph.nodes):
                meta = node.meta.get("val", None)
                if i == 0:
                    self.assertEqual(meta, sparse_input.to("meta"))
                else:
                    self.assertEqual(meta, None)

    @unittest.skipIf(
        sys.version_info >= (3, 12), "torch.compile is not supported on python 3.12+"
    )
    @parametrize("dtype", DTYPES)
    @parametrize("itype", ITYPES)
    @all_sparse_layouts("layout")
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
            # Build the traced graph.
            prog = torch.export.export(net, (sparse_input,))
            # Test arg/sum/output.
            for i, node in enumerate(prog.graph.nodes):
                meta = node.meta.get("val", None)
                if i == 0:
                    self.assertEqual(meta, sparse_input.to("meta"))
                elif i == 1:
                    self.assertIsInstance(meta, FakeTensor)
                    self.assertEqual(meta.layout, torch.strided)
                    self.assertEqual(meta.dtype, dtype)
                else:
                    self.assertEqual(meta, None)

    @unittest.skipIf(
        sys.version_info >= (3, 12), "torch.compile is not supported on python 3.12+"
    )
    @parametrize("dtype", DTYPES)
    @parametrize("itype", ITYPES)
    @all_sparse_layouts("layout")
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
            # Build the traced graph.
            prog = torch.export.export(net, (sparse_input,))
            # Test arg/neg/abs/mul/relu/output.
            for i, node in enumerate(prog.graph.nodes):
                meta = node.meta.get("val", None)
                if i <= 4:
                    self.assertEqual(meta, sparse_input.to("meta"))
                else:
                    self.assertEqual(meta, None)


instantiate_parametrized_tests(TestSparseProp)

if __name__ == "__main__":
    run_tests()
