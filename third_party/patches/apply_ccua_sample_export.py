#!/usr/bin/env python3
"""Give the official CCUA sampler an explicit output path and uniform labels.

Two operational changes, both opt-in and both confined to ``main.py --sample``:

* ``--sample_output`` redirects the generated arrays to an absolute path.  The
  upstream filename is derived from a nine-field mangling of ``sample_name``
  (category, N, step, omega, scheduler, gamma, seed, DDIM steps), so predicting
  it in the adapter would couple the runner to float formatting of two flags.
  It *redirects* rather than adds a second write: a 50k CIFAR array is ~600 MB,
  and saving both copies would cost ~11 GB across an 18-task campaign.

* ``--uniform_labels`` replaces the iid ``torch.randint`` conditioning draw with
  an exact class-uniform schedule.  Every method in the unified table must be
  scored on the same label support, otherwise the 50k FID compares different
  class mixtures rather than different models.

Training and the reverse process are untouched.
"""
from __future__ import annotations

import argparse
from pathlib import Path


MARKER = ".ltx_ccua_sample_export_v1"


def once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"CCUA sample-export anchor missing: {label}")
    return text.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    marker = repo / MARKER
    if marker.exists():
        return
    path = repo / "main.py"
    text = path.read_text(encoding="utf-8")

    flag_anchor = "flags.DEFINE_bool('sampled', False, help='evaluate sampled images')"
    if "flags.DEFINE_string('sample_output'" not in text:
        if flag_anchor not in text:
            raise RuntimeError("CCUA sample-export anchor missing: flags")
        pos = text.index(flag_anchor) + len(flag_anchor)
        text = (
            text[:pos]
            + "\nflags.DEFINE_string('sample_output', '', help='absolute .npy path for generated arrays')"
            + "\nflags.DEFINE_bool('uniform_labels', False, help='sample an exact class-uniform label schedule during evaluation')"
            + text[pos:]
        )

    old_labels = (
        "                batch_idx = torch.randint(len(classes), size=(x_T.shape[0],))\n"
        "                batch_labels = classes[batch_idx].to(device)\n"
    )
    new_labels = (
        "                if FLAGS.uniform_labels:\n"
        "                    batch_labels = classes[torch.arange(i, i + batch_size) % len(classes)].to(device)\n"
        "                else:\n"
        "                    batch_idx = torch.randint(len(classes), size=(x_T.shape[0],))\n"
        "                    batch_labels = classes[batch_idx].to(device)\n"
    )
    text = once(text, old_labels, new_labels, "sampler labels")

    old_save = (
        "        np.save(os.path.join(FLAGS.logdir, '{}_{}_samples_ema_{}.npy'.format(\n"
        "                             FLAGS.sample_method, FLAGS.omega,\n"
        "                             FLAGS.sample_name)), images)\n"
        "        if FLAGS.sample_method != 'uncond':\n"
        "            labels = torch.cat(labels, dim=0).cpu().numpy()\n"
        "            np.save(os.path.join(FLAGS.logdir, '{}_{}_labels_ema_{}.npy'.format(\n"
        "                                 FLAGS.sample_method, FLAGS.omega,\n"
        "                                 FLAGS.sample_name)), labels)\n"
    )
    new_save = (
        "        _ltx_upstream = os.path.join(FLAGS.logdir, '{}_{}_samples_ema_{}.npy'.format(\n"
        "                             FLAGS.sample_method, FLAGS.omega,\n"
        "                             FLAGS.sample_name))\n"
        "        _ltx_samples_path = FLAGS.sample_output or _ltx_upstream\n"
        "        np.save(_ltx_samples_path, images)\n"
        "        if FLAGS.sample_method != 'uncond':\n"
        "            labels = torch.cat(labels, dim=0).cpu().numpy()\n"
        "            _ltx_labels_path = (FLAGS.sample_output + '.labels.npy') if FLAGS.sample_output else \\\n"
        "                os.path.join(FLAGS.logdir, '{}_{}_labels_ema_{}.npy'.format(\n"
        "                                 FLAGS.sample_method, FLAGS.omega,\n"
        "                                 FLAGS.sample_name))\n"
        "            np.save(_ltx_labels_path, labels)\n"
    )
    text = once(text, old_save, new_save, "sample output")

    path.write_text(text, encoding="utf-8")
    marker.write_text("explicit sample_output path + exact class-uniform eval labels\n", encoding="utf-8")


if __name__ == "__main__":
    main()
