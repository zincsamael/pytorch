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


class TestSparseProp(TestCase):
    def setUp(self):
        TestCase.setUp(self)

    # TODO: test csr versions, and other of these

    def test_sumnet_coo(self):
        coo_input = torch.ones(2, 4).to_sparse_coo()
        prog = torch.export.export(SumNet(), (coo_input,))
        # Test arg/sum/output.
        for i, node in enumerate(prog.graph.nodes):
            meta = node.meta.get("tensor_meta", None)
            if i == 0:
                self.assertEqual(meta.layout, torch.sparse_coo)
                self.assertEqual(meta.sparse_dim, 2)
            elif i == 1:
                self.assertEqual(meta.layout, torch.strided)
                self.assertEqual(meta.sparse_dim, None)
            else:
                self.assertEqual(meta, None)

    def test_eltwisenet_coo(self):
        coo_input = torch.ones(2, 4).to_sparse_coo()
        prog = torch.export.export(EltwiseNet(), (coo_input,))
        # Test arg/neg/sin/mul/output
        for i, node in enumerate(prog.graph.nodes):
            meta = node.meta.get("tensor_meta", None)
            if i <= 3:
                self.assertEqual(meta.layout, torch.sparse_coo)
                self.assertEqual(meta.sparse_dim, 2)
            else:
                self.assertEqual(meta, None)


if __name__ == "__main__":
    run_tests()
