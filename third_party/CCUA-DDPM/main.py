"""Baseline-only CCUA-DDPM trainer and sampler for CIFAR-100-LT."""

from __future__ import annotations

import copy
import os
import tempfile
import warnings

import numpy as np
import torch
from absl import app, flags
from tensorboardX import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm import trange

from dataset import ImbalanceCIFAR100
from diffusion import (
    GaussianDiffusionSampler,
    GaussianDiffusionSamplerDDIM,
    GaussianDiffusionTrainer,
)
from model.model import UNet


def atomic_torch_save(payload, destination) -> None:
    """Atomically replace a checkpoint after its bytes reach disk."""
    destination = os.fspath(destination)
    parent = os.path.dirname(destination) or "."
    os.makedirs(parent, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(destination)}.", suffix=".tmp", dir=parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        try:
            directory = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            pass
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 42, "random seed")
flags.DEFINE_bool("train", False, "train or resume training")
flags.DEFINE_bool("sample", False, "sample from a checkpoint")
flags.DEFINE_bool("resume", False, "resume full training state")
flags.DEFINE_string("resume_dir", "./", "directory containing the resume checkpoint")
flags.DEFINE_integer("ckpt_step", 0, "checkpoint step")

flags.DEFINE_integer("ch", 128, "base U-Net channels")
flags.DEFINE_multi_integer("ch_mult", [1, 2, 2, 2], "U-Net channel multipliers")
flags.DEFINE_multi_integer("attn", [1], "U-Net attention levels")
flags.DEFINE_integer("num_res_blocks", 2, "U-Net residual blocks per level")
flags.DEFINE_float("dropout", 0.1, "U-Net dropout")

flags.DEFINE_float("beta_1", 1e-4, "first diffusion beta")
flags.DEFINE_float("beta_T", 0.02, "last diffusion beta")
flags.DEFINE_integer("T", 1000, "diffusion steps")
flags.DEFINE_enum("var_type", "fixedlarge", ["fixedlarge", "fixedsmall"], "variance type")

flags.DEFINE_float("lr", 2e-4, "learning rate")
flags.DEFINE_float("grad_clip", 1.0, "gradient clipping norm")
flags.DEFINE_integer("total_steps", 300001, "updates to execute in this process")
flags.DEFINE_integer("warmup", 5000, "learning-rate warmup updates")
flags.DEFINE_integer("batch_size", 64, "training batch size")
flags.DEFINE_integer("num_workers", 4, "data-loader workers")
flags.DEFINE_float("ema_decay", 0.9999, "EMA decay")
flags.DEFINE_integer("save_step", 100000, "checkpoint interval; zero disables")
flags.DEFINE_integer("sample_step", 50000, "preview interval; zero disables")
flags.DEFINE_integer("sample_size", 64, "training preview size")

flags.DEFINE_enum("data_type", "cifar100lt", ["cifar100lt"], "dataset contract")
flags.DEFINE_string("data_path", ".cache/data", "CIFAR data root")
flags.DEFINE_float("imb_factor", 0.01, "long-tail imbalance factor")
flags.DEFINE_integer("num_class", 100, "number of classes")
flags.DEFINE_integer("img_size", 32, "image size")
flags.DEFINE_string("logdir", "./logs/", "run directory")

flags.DEFINE_integer("num_images", 50000, "number of generated images")
flags.DEFINE_integer("sample_batch_size", 0, "sampling batch size; zero reuses batch_size")
flags.DEFINE_enum("sample_method", "ddim", ["ddim", "ddpm"], "sampling method")
flags.DEFINE_integer("ddim_skip_step", 10, "DDIM/DDPM sampling stride")
flags.DEFINE_float("omega", 1.5, "classifier-free guidance scale")
flags.DEFINE_string("omega_scheduler", "constant", "guidance scheduler")
flags.DEFINE_float("gamma", 100, "gamma guidance-scheduler parameter")
flags.DEFINE_string("sample_output", "", "required output .npy path")
flags.DEFINE_bool("uniform_labels", False, "use an exact class-uniform label schedule")

flags.DEFINE_bool("transfer_x0", False, "enable the T2H transfer objective")
flags.DEFINE_bool("cbdm", False, "enable the CBDM objective")
flags.DEFINE_float("cb_tau", 1.0, "CBDM temperature")
flags.DEFINE_float("ccua_al", 0.0, "CCUA alignment-loss weight")
flags.DEFINE_float("ccua_ucl", 0.0, "CCUA unconditional-contrastive-loss weight")

DEVICE = torch.device("cuda:0")


def ema(source, target, decay: float) -> None:
    source_state = source.state_dict()
    for key, value in target.state_dict().items():
        value.data.copy_(value.data * decay + source_state[key].data * (1 - decay))


def infiniteloop(dataloader):
    while True:
        yield from dataloader


def warmup_lr(step: int) -> float:
    return min(step, FLAGS.warmup) / FLAGS.warmup


def _model() -> UNet:
    return UNet(
        T=FLAGS.T,
        ch=FLAGS.ch,
        ch_mult=FLAGS.ch_mult,
        attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks,
        dropout=FLAGS.dropout,
        cond=True,
        num_class=FLAGS.num_class,
    )


def train() -> None:
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize([FLAGS.img_size, FLAGS.img_size], antialias=True),
    ])
    dataset = ImbalanceCIFAR100(
        root=FLAGS.data_path,
        imb_type="exp",
        imb_factor=FLAGS.imb_factor,
        rand_number=0,
        train=True,
        transform=transform,
        download=True,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
    )
    batches = infiniteloop(dataloader)
    print(
        f"Dataset {FLAGS.data_type} contains {len(dataset.targets)} images "
        f"with {len(np.unique(dataset.targets))} classes"
    )

    counts = torch.tensor(np.unique(dataset.targets, return_counts=True)[1], dtype=torch.float32)
    class_weights = (counts / counts.sum()).unsqueeze(0)

    net_model = _model()
    ema_model = copy.deepcopy(net_model)
    optimizer = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr)
    trainer = GaussianDiffusionTrainer(
        net_model,
        FLAGS.beta_1,
        FLAGS.beta_T,
        FLAGS.T,
        FLAGS.num_class,
        True,
        class_weights,
        transfer_x0=FLAGS.transfer_x0,
        cbdm=FLAGS.cbdm,
        cb_tau=FLAGS.cb_tau,
        ccua_ucl=FLAGS.ccua_ucl,
        ccua_al=FLAGS.ccua_al,
    ).to(DEVICE)
    ema_sampler = GaussianDiffusionSampler(
        ema_model,
        FLAGS.beta_1,
        FLAGS.beta_T,
        FLAGS.T,
        FLAGS.num_class,
        FLAGS.img_size,
        FLAGS.var_type,
    ).to(DEVICE)

    if FLAGS.resume:
        checkpoint_path = os.path.join(FLAGS.resume_dir, f"ckpt_{FLAGS.ckpt_step}.pt")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        net_model.load_state_dict(checkpoint["net_model"])
        ema_model.load_state_dict(checkpoint["ema_model"])
        optimizer.load_state_dict(checkpoint["optim"])
        scheduler.load_state_dict(checkpoint["sched"])
        print(f"Loaded full training state from {checkpoint_path}")

    sample_dir = os.path.join(FLAGS.logdir, "sample")
    os.makedirs(sample_dir, exist_ok=True)
    writer = SummaryWriter(FLAGS.logdir)
    fixed_noise = torch.randn(
        min(FLAGS.sample_size, 100), 3, FLAGS.img_size, FLAGS.img_size, device=DEVICE
    )
    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), "w", encoding="utf-8") as stream:
        stream.write(FLAGS.flags_into_string())
    parameter_count = sum(parameter.data.nelement() for parameter in net_model.parameters())
    print(f"Model params: {parameter_count / 1024 / 1024:.2f} M")

    with trange(FLAGS.total_steps, dynamic_ncols=True) as progress:
        for step in progress:
            optimizer.zero_grad()
            images, labels = next(batches)
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            ddpm, cbdm, ccua_ucl, ccua_al = trainer(images, labels)
            ddpm = ddpm.mean()
            cbdm = cbdm.mean()
            ccua_ucl = ccua_ucl.mean()
            ccua_al = ccua_al.mean()
            loss = ddpm + cbdm + ccua_ucl * FLAGS.ccua_ucl + ccua_al * FLAGS.ccua_al
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
            optimizer.step()
            scheduler.step()
            ema(net_model, ema_model, FLAGS.ema_decay)

            writer.add_scalar("loss", loss, step)
            writer.add_scalar("loss_ddpm", ddpm, step)
            writer.add_scalar("loss_cbdm", cbdm, step)
            writer.add_scalar("loss_ccua_ucl", ccua_ucl, step)
            writer.add_scalar("loss_ccua_al", ccua_al, step)
            progress.set_postfix(
                loss=f"{loss:.5f}",
                loss_ddpm=f"{ddpm:.5f}",
                loss_cbdm=f"{cbdm:.5f}",
                loss_ccua_ucl=f"{ccua_ucl:.5f}",
                loss_ccua_al=f"{ccua_al:.5f}",
            )

            if FLAGS.sample_step > 0 and step and step % FLAGS.sample_step == 0:
                net_model.eval()
                with torch.no_grad():
                    preview, _ = ema_sampler(fixed_noise)
                    grid = (make_grid(preview) + 1) / 2
                    save_image(grid, os.path.join(sample_dir, f"{step}.png"))
                    writer.add_image("sample", grid, step)
                net_model.train()

            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                completed_step = step + FLAGS.ckpt_step
                atomic_torch_save(
                    {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": scheduler.state_dict(),
                        "optim": optimizer.state_dict(),
                        "step": completed_step,
                        "fixed_x_T": fixed_noise,
                    },
                    os.path.join(FLAGS.logdir, f"ckpt_{completed_step}.pt"),
                )
    writer.close()


def sample() -> None:
    if not FLAGS.sample_output:
        raise ValueError("--sample_output is required")
    model = _model()
    sampler = GaussianDiffusionSamplerDDIM(
        model,
        FLAGS.beta_1,
        FLAGS.beta_T,
        FLAGS.T,
        img_size=FLAGS.img_size,
        var_type=FLAGS.var_type,
        omega=FLAGS.omega,
        omega_scheduler=FLAGS.omega_scheduler,
        gamma=FLAGS.gamma,
        cond=True,
    ).to(DEVICE)
    checkpoint = torch.load(
        os.path.join(FLAGS.logdir, f"ckpt_{FLAGS.ckpt_step}.pt"), map_location="cpu"
    )
    model.load_state_dict(checkpoint["ema_model"])
    model.eval()

    classes = torch.arange(FLAGS.num_class)
    sample_batch_size = int(FLAGS.sample_batch_size or FLAGS.batch_size)
    if sample_batch_size <= 0:
        raise ValueError("sample_batch_size must be positive")
    image_batches = []
    label_batches = []
    with torch.inference_mode():
        for start in trange(0, FLAGS.num_images, sample_batch_size, desc="generating images"):
            batch_size = min(sample_batch_size, FLAGS.num_images - start)
            noise = torch.randn(batch_size, 3, FLAGS.img_size, FLAGS.img_size, device=DEVICE)
            if FLAGS.uniform_labels:
                labels = classes[torch.arange(start, start + batch_size) % len(classes)].to(DEVICE)
            else:
                labels = classes[torch.randint(len(classes), size=(batch_size,))].to(DEVICE)
            generated = sampler(
                noise,
                y=labels,
                method=FLAGS.sample_method,
                skip=FLAGS.ddim_skip_step,
                return_timestep_status=False,
            )
            image_batches.append(((generated + 1) / 2).cpu())
            label_batches.append(labels.cpu())

    images = torch.cat(image_batches).numpy()
    labels = torch.cat(label_batches).numpy()
    np.save(FLAGS.sample_output, images)
    np.save(FLAGS.sample_output + ".labels.npy", labels)
    save_image(
        torch.as_tensor(images[:256]),
        os.path.join(FLAGS.logdir, "visual_ema.png"),
        nrow=8,
    )


def main(_argv) -> None:
    warnings.simplefilter(action="ignore", category=FutureWarning)
    torch.manual_seed(FLAGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FLAGS.seed)
    if FLAGS.train:
        train()
    if FLAGS.sample:
        sample()


if __name__ == "__main__":
    app.run(main)
