# Owner(s): ["module: sparse"]

import torch

from torch.testing._internal.common_utils import TestCase, run_tests
from typing import Optional, Tuple


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


# All sparse layouts currently supported in torch.sparse.
# TODO: make this work for all
SPARSE_LAYOUTS = [
    torch.sparse_coo,
    #    torch.sparse_csr,
    #    torch.sparse_csc,
    #    torch.sparse_bsr,
    #    torch.sparse_bsc,
]


def make_sparse(layout: torch.layout) -> Tuple[torch.Tensor, Optional[Tuple[int, int]]]:
    dense = torch.ones(4, 8)
    if layout == torch.sparse_coo:
        return (dense.to_sparse_coo(), None)
    if layout == torch.sparse_csr:
        return (dense.to_sparse_csr(), None)
    if layout == torch.sparse_csc:
        return (dense.to_sparse_csc(), None)
    if layout == torch.sparse_bsr:
        return (dense.to_sparse_bsr((2, 2)), (2, 2))
    if layout == torch.sparse_bsc:
        return (dense.to_sparse_bsc((2, 4)), (2, 4))


class TestSparseProp(TestCase):
    def setUp(self):
        TestCase.setUp(self)

    def test_sumnet(self):
        net = SumNet()
        for sparse_layout in SPARSE_LAYOUTS:
            sparse_input, blocksize = make_sparse(sparse_layout)
            prog = torch.export.export(net, (sparse_input,))
            # Test arg/sum/output.
            for i, node in enumerate(prog.graph.nodes):
                meta = node.meta.get("tensor_meta", None)
                if i == 0:
                    self.assertEqual(meta.layout, sparse_layout)
                    self.assertEqual(meta.batch_dim, 0)
                    self.assertEqual(meta.sparse_dim, 2)
                    self.assertEqual(meta.dense_dim, 0)
                    self.assertEqual(meta.blocksize, blocksize)
                    self.assertEqual(meta.dtype, torch.float32)
                elif i == 1:
                    self.assertEqual(meta.layout, torch.strided)
                    self.assertEqual(meta.sparse_dim, None)
                else:
                    self.assertEqual(meta, None)

    def test_eltwisenet(self):
        net = EltwiseNet()
        for sparse_layout in SPARSE_LAYOUTS:
            sparse_input, blocksize = make_sparse(sparse_layout)
            prog = torch.export.export(net, (sparse_input,))
            # Test arg/neg/sin/mul/output.
            for i, node in enumerate(prog.graph.nodes):
                meta = node.meta.get("tensor_meta", None)
                if i <= 3:
                    self.assertEqual(meta.layout, sparse_layout)
                    self.assertEqual(meta.batch_dim, 0)
                    self.assertEqual(meta.sparse_dim, 2)
                    self.assertEqual(meta.dense_dim, 0)
                    self.assertEqual(meta.blocksize, blocksize)
                    self.assertEqual(meta.dtype, torch.float32)
                else:
                    self.assertEqual(meta, None)


if __name__ == "__main__":
    run_tests()
