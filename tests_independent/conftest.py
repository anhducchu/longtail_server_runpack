"""Independent verification suite for ``scaffold/``.

    python -m pytest tests_independent                 # CPU, synthetic, ~1 min
    python -m pytest tests_independent --real-assets   # + pinned CIFAR-100 / Inception / VGG assets

The oracles live in ``indep_helpers.py`` and are derived from the README maths,
numpy, scipy and torch.distributions. They share no code with ``tests/``.
"""
import pytest
import torch


def pytest_addoption(parser):
    parser.addoption('--real-assets', action='store_true', default=False,
                     help='also run the tests that need the pinned datasets, statistics and weights')


def pytest_configure(config):
    config.addinivalue_line('markers', 'real_assets: needs data/, third_party/CCUA-DDPM/stats or pretrained weights')
    torch.set_num_threads(4)


def pytest_collection_modifyitems(config, items):
    if config.getoption('--real-assets'):
        return
    skip = pytest.mark.skip(reason='needs --real-assets')
    for item in items:
        if 'real_assets' in item.keywords:
            item.add_marker(skip)
