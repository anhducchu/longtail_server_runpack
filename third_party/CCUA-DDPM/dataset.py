"""CIFAR-100-LT dataset used by the baseline campaign."""

from __future__ import annotations

import numpy as np
import torchvision.datasets as datasets


class ImbalanceCIFAR100(datasets.CIFAR100):
    cls_num = 100

    def __init__(
        self,
        root,
        imb_type="exp",
        imb_factor=0.01,
        rand_number=0,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        super().__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        image_counts = self._image_counts(imb_type, imb_factor)
        self.num_per_cls_dict = {}
        self._make_imbalanced(image_counts)

    def _image_counts(self, imbalance_type: str, imbalance_factor: float) -> list[int]:
        maximum = len(self.data) / self.cls_num
        if imbalance_type == "exp":
            return [
                int(maximum * imbalance_factor ** (index / (self.cls_num - 1.0)))
                for index in range(self.cls_num)
            ]
        if imbalance_type == "step":
            head = [int(maximum)] * (self.cls_num // 2)
            tail = [int(maximum * imbalance_factor)] * (self.cls_num // 2)
            return head + tail
        return [int(maximum)] * self.cls_num

    def _make_imbalanced(self, image_counts: list[int]) -> None:
        new_data = []
        new_targets = []
        targets = np.asarray(self.targets, dtype=np.int64)
        for class_id, image_count in zip(np.unique(targets), image_counts):
            self.num_per_cls_dict[int(class_id)] = image_count
            indices = np.flatnonzero(targets == class_id)
            np.random.shuffle(indices)
            selected = indices[:image_count]
            new_data.append(self.data[selected, ...])
            new_targets.extend([int(class_id)] * image_count)
        self.data = np.vstack(new_data)
        self.targets = new_targets

    def get_cls_num_list(self) -> list[int]:
        return [self.num_per_cls_dict[index] for index in range(self.cls_num)]
