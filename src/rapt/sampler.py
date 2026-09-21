from __future__ import annotations
import math
from typing import Iterator, List, Sequence
import numpy as np
from torch.utils.data import Sampler


class TaskBalancedBatchSampler(Sampler[List[int]]):
    """
    Dataset-local batch sampler. `task_ids` must correspond to the dataset rows,
    not global row indices.
    """

    def __init__(self, task_ids: Sequence, samples_per_task: int = 8, seed: int = 0):
        self.task_ids = np.asarray(task_ids)
        self.tasks = np.array(sorted(np.unique(self.task_ids).tolist()))
        self.samples_per_task = int(samples_per_task)
        self.seed = int(seed)

        self.by_task = {t: np.flatnonzero(self.task_ids == t) for t in self.tasks}
        self.batch_size = len(self.tasks) * self.samples_per_task
        self.n_batches = max(1, math.ceil(len(self.task_ids) / self.batch_size))
        self.epoch = 0

    def __len__(self):
        return self.n_batches

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        pools, cursors = {}, {}

        for t, idx in self.by_task.items():
            p = idx.copy()
            rng.shuffle(p)
            pools[t] = p
            cursors[t] = 0

        for _ in range(self.n_batches):
            batch = []
            for t in self.tasks:
                pool = pools[t]
                cur = cursors[t]
                need = self.samples_per_task
                chosen = []

                while need > 0:
                    remain = len(pool) - cur
                    if remain >= need:
                        chosen.extend(pool[cur:cur + need].tolist())
                        cur += need
                        need = 0
                    else:
                        if remain:
                            chosen.extend(pool[cur:].tolist())
                            need -= remain
                        pool = self.by_task[t].copy()
                        rng.shuffle(pool)
                        cur = 0

                pools[t], cursors[t] = pool, cur
                batch.extend(chosen)

            rng.shuffle(batch)
            yield batch

        self.epoch += 1
