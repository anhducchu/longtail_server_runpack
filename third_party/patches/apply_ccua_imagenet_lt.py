#!/usr/bin/env python3
"""Add the manifest-backed ImageNet-LT loader to the pinned CCUA U-Net tree.

The released CCUA ImageNet path wraps ``ImageFolder`` and therefore expects a
complete ImageNet train directory.  ImageNet-LT is a published subset with a
specific 115,846-image manifest; silently constructing a new exponential split
from all ImageNet images would change the benchmark.  This patch adds an
explicit ``data_type=imagenet_lt`` branch that reads exactly that manifest.
"""
from __future__ import annotations

import argparse
from pathlib import Path


MARKER = ".ltx_ccua_imagenet_lt_patch_v1"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"CCUA ImageNet-LT patch anchor missing: {label}")
    return text.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    marker = repo / MARKER
    if marker.exists():
        return

    dataset_path = repo / "dataset.py"
    main_path = repo / "main.py"
    dataset = dataset_path.read_text(encoding="utf-8")
    main = main_path.read_text(encoding="utf-8")

    loader = r'''


class ImageNetLTManifest(torch.utils.data.Dataset):
    """ImageNet-LT train split described by ``<relative path> <label>`` rows."""

    def __init__(self, root, manifest, num_class=1000, transform=None, target_transform=None):
        self.root = os.path.abspath(root)
        self.transform = transform
        self.target_transform = target_transform
        self.samples = []
        with open(manifest, encoding="utf-8") as handle:
            for lineno, raw in enumerate(handle, 1):
                fields = raw.split()
                if not fields or raw.lstrip().startswith("#"):
                    continue
                if len(fields) != 2:
                    raise ValueError(f"ImageNet-LT manifest line {lineno} must be '<relative_image> <label>'")
                relative, label_raw = fields
                label = int(label_raw)
                if not 0 <= label < int(num_class):
                    raise ValueError(f"ImageNet-LT label {label} at line {lineno} is outside 0..{int(num_class) - 1}")
                image_path = relative if os.path.isabs(relative) else os.path.join(self.root, relative)
                if not os.path.isfile(image_path):
                    raise FileNotFoundError(f"ImageNet-LT image missing at line {lineno}: {image_path}")
                self.samples.append((image_path, label))
        if not self.samples:
            raise ValueError(f"ImageNet-LT manifest is empty: {manifest}")
        self.imgs = self.samples
        self.targets = [label for _, label in self.samples]
        self.classes = [str(index) for index in range(int(num_class))]
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.loader = datasets.folder.default_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target
'''
    dataset = replace_once(dataset, "class ImbalanceCIFAR10", loader + "\n\nclass ImbalanceCIFAR10", "manifest loader")

    main = replace_once(
        main,
        "from dataset import ImbalanceCIFAR100, ImbalanceCIFAR10, ImbalanceImageNet, ImbalanceTinyImageNet, ImageNet, PlacesLT, PlacesLD, SubsetPerLabel",
        "from dataset import ImbalanceCIFAR100, ImbalanceCIFAR10, ImbalanceImageNet, ImbalanceTinyImageNet, ImageNet, ImageNetLTManifest, PlacesLT, PlacesLD, SubsetPerLabel",
        "dataset import",
    )
    main = replace_once(
        main,
        "flags.DEFINE_float('imb_factor', 0.01, help='imb_factor for long tail dataset')",
        "flags.DEFINE_float('imb_factor', 0.01, help='imb_factor for long tail dataset')\nflags.DEFINE_string('train_manifest', '', help='ImageNet-LT train manifest for data_type=imagenet_lt')",
        "train manifest flag",
    )
    main = replace_once(
        main,
        "    elif FLAGS.data_type == 'imgnetlt':\n        FLAGS.data_path = FLAGS.data_path\n        dataset = ImbalanceImageNet(root=FLAGS.data_path,",
        "    elif FLAGS.data_type == 'imagenet_lt':\n        if not FLAGS.train_manifest:\n            raise ValueError('data_type=imagenet_lt requires --train_manifest')\n        dataset = ImageNetLTManifest(root=FLAGS.data_path,\n                                     manifest=FLAGS.train_manifest,\n                                     num_class=FLAGS.num_class,\n                                     transform=tran_transform)\n    elif FLAGS.data_type == 'imgnetlt':\n        FLAGS.data_path = FLAGS.data_path\n        dataset = ImbalanceImageNet(root=FLAGS.data_path,",
        "ImageNet-LT data branch",
    )

    dataset_path.write_text(dataset, encoding="utf-8")
    main_path.write_text(main, encoding="utf-8")
    marker.write_text("manifest-backed ImageNet-LT loader for the pinned CCUA U-Net source\n", encoding="utf-8")


if __name__ == "__main__":
    main()
