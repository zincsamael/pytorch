# Owner(s): ["oncall: distributed"]

from datetime import timedelta
from multiprocessing.pool import ThreadPool

import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import run_tests, TestCase


class TestCollectives(TestCase):
    def test_barrier(self) -> None:
        store = dist.HashStore()

        world_size = 2

        def f(rank: int) -> None:
            collectives = dist.StoreCollectives(store, rank, world_size)
            collectives.barrier("foo", timedelta(seconds=10), True)

        with ThreadPool(world_size) as pool:
            pool.map(f, range(world_size))

    def test_broadcast(self) -> None:
        store = dist.HashStore()

        world_size = 4
        timeout = timedelta(seconds=10)

        def f(rank: int) -> None:
            collectives = dist.StoreCollectives(store, rank, world_size)
            if rank == 2:
                collectives.broadcast_send("foo", "data", timeout)
            else:
                out = collectives.broadcast_recv("foo", timeout)
                self.assertEqual(out, "data")

        with ThreadPool(world_size) as pool:
            pool.map(f, range(world_size))

    def test_gather(self) -> None:
        store = dist.HashStore()

        world_size = 4
        timeout = timedelta(seconds=10)

        def f(rank: int) -> None:
            collectives = dist.StoreCollectives(store, rank, world_size)
            if rank == 2:
                out = collectives.gather_recv("foo", str(rank), timeout)
                self.assertEqual(out, ["0", "1", "2", "3"])
            else:
                collectives.gather_send("foo", str(rank), timeout)

        with ThreadPool(world_size) as pool:
            pool.map(f, range(world_size))

    def test_scatter(self) -> None:
        store = dist.HashStore()

        world_size = 4
        timeout = timedelta(seconds=10)

        def f(rank: int) -> None:
            collectives = dist.StoreCollectives(store, rank, world_size)
            if rank == 2:
                out = collectives.scatter_send(
                    "foo", [str(i) for i in range(world_size)], timeout
                )
            else:
                out = collectives.scatter_recv("foo", timeout)
            self.assertEqual(out, str(rank))

        with ThreadPool(world_size) as pool:
            pool.map(f, range(world_size))

    def test_all_sum(self) -> None:
        store = dist.HashStore()

        world_size = 4
        timeout = timedelta(seconds=10)

        def f(rank: int) -> None:
            collectives = dist.StoreCollectives(store, rank, world_size)
            out = collectives.all_sum("foo", rank, timeout)
            self.assertEqual(out, sum(range(world_size)))

        with ThreadPool(world_size) as pool:
            pool.map(f, range(world_size))

    def test_broadcast_timeout(self) -> None:
        store = dist.HashStore()

        world_size = 4
        timeout = timedelta(milliseconds=1)
        collectives = dist.StoreCollectives(store, 1, world_size)
        with self.assertRaisesRegex(Exception, "Wait timeout"):
            collectives.broadcast_recv("foo", timeout)

    def test_gather_timeout(self) -> None:
        store = dist.HashStore()

        world_size = 4
        timeout = timedelta(milliseconds=1)
        collectives = dist.StoreCollectives(store, 1, world_size)
        with self.assertRaisesRegex(
            Exception, "gather failed -- missing ranks: 0, 2, 3"
        ):
            collectives.gather_recv("foo", "data", timeout)

    def test_scatter_timeout(self) -> None:
        store = dist.HashStore()

        world_size = 4
        timeout = timedelta(milliseconds=1)
        collectives = dist.StoreCollectives(store, 1, world_size)
        with self.assertRaisesRegex(Exception, "Wait timeout"):
            collectives.scatter_recv("foo", timeout)

    def test_all_gather_timeout(self) -> None:
        store = dist.HashStore()

        world_size = 4
        timeout = timedelta(milliseconds=1)
        collectives = dist.StoreCollectives(store, 1, world_size)
        with self.assertRaisesRegex(
            Exception, "all_gather failed -- missing ranks: 0, 2, 3"
        ):
            collectives.all_gather("foo", "data", timeout)

    def test_barrier_timeout(self) -> None:
        store = dist.HashStore()

        world_size = 4
        timeout = timedelta(milliseconds=1)
        collectives = dist.StoreCollectives(store, 1, world_size)
        with self.assertRaisesRegex(
            Exception, "barrier failed -- missing ranks: 0, 2, 3"
        ):
            collectives.barrier("foo", timeout, True)

    def test_all_sum_timeout(self) -> None:
        store = dist.HashStore()

        world_size = 4
        timeout = timedelta(milliseconds=1)
        collectives = dist.StoreCollectives(store, 1, world_size)
        with self.assertRaisesRegex(
            Exception, "barrier failed -- missing ranks: 0, 2, 3"
        ):
            collectives.all_sum("foo", 1, timeout)


if __name__ == "__main__":
    assert (
        not torch.cuda._initialized
    ), "test_distributed must not have initialized CUDA context on main process"

    run_tests()
