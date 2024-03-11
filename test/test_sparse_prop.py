# Owner(s): ["module: sparse"]

import torch

from torch.testing._internal.common_utils import TestCase, run_tests


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


def make_sparse(layout: torch.layout, dense: torch.Tensor) -> torch.Tensor:
    if layout == torch.sparse_coo:
        return dense.to_sparse_coo()
    if layout == torch.sparse_csr:
        return dense.to_sparse_csr()
    if layout == torch.sparse_csc:
        return dense.to_sparse_csc()
    if layout == torch.sparse_bsr:
        return dense.to_sparse_bsr((2, 2))
    if layout == torch.sparse_bsc:
        return dense.to_sparse_bsc((2, 2))


class TestSparseProp(TestCase):
    def setUp(self):
        TestCase.setUp(self)

    # TODO: test csr versions, and other of these

    def test_sumnet(self):
        net = SumNet()
        for sparse_layout in SPARSE_LAYOUTS:
            sparse_input = make_sparse(sparse_layout, torch.ones(4, 8))
            prog = torch.export.export(net, (sparse_input,))
            # Test arg/sum/output.
            for i, node in enumerate(prog.graph.nodes):
                meta = node.meta.get("tensor_meta", None)
                if i == 0:
                    self.assertEqual(meta.layout, sparse_layout)
                    self.assertEqual(meta.sparse_dim, 2)
                elif i == 1:
                    self.assertEqual(meta.layout, torch.strided)
                    self.assertEqual(meta.sparse_dim, None)
                else:
                    self.assertEqual(meta, None)

    def test_eltwisenet_coo(self):
        net = EltwiseNet()
        for sparse_layout in SPARSE_LAYOUTS:
            sparse_input = make_sparse(sparse_layout, torch.ones(4, 8))
            prog = torch.export.export(net, (sparse_input,))
            # Test arg/neg/sin/mul/output
            for i, node in enumerate(prog.graph.nodes):
                meta = node.meta.get("tensor_meta", None)
                if i <= 3:
                    self.assertEqual(meta.layout, sparse_layout)
                    self.assertEqual(meta.sparse_dim, 2)
                else:
                    self.assertEqual(meta, None)


if __name__ == "__main__":
    run_tests()
