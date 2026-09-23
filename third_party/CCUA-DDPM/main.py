import copy
import json
import os
import tempfile
import warnings
from absl import app, flags
from tqdm import trange

import torch
import numpy as np
from torchvision.datasets import ImageFolder
# import kagglehub


from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.utils import make_grid, save_image
from torchvision import transforms
try:
    from tensorboardX import SummaryWriter
except Exception as err:
    pass
from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler, GaussianDiffusionSamplerDDIM
from model.model import UNet
from model.classifier import HalveUNetClassifier

from dataset import ImbalanceCIFAR100, ImbalanceCIFAR10, ImbalanceImageNet, ImbalanceTinyImageNet, ImageNet, ImageNetLTManifest, PlacesLT, PlacesLD, SubsetPerLabel
# from score_new.both import get_inception_and_fid_score as get_inception_and_fid_score_new


def atomic_torch_save(payload, destination):
    """Atomically replace a checkpoint after its bytes reach disk."""
    destination = os.fspath(destination)
    parent = os.path.dirname(destination) or "."
    os.makedirs(parent, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(destination)}.",
        suffix=".tmp",
        dir=parent,
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        try:
            directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass



FLAGS = flags.FLAGS
flags.DEFINE_integer('seed', 42, help='random seed')
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('resume',False,help='resume from a checkpoint')
flags.DEFINE_string('resume_dir','./',help='the resumed checkpoint')
flags.DEFINE_integer('ckpt_step', 0 ,help='resumed checkpoint step')
flags.DEFINE_bool('eval', False, help='deprecated, please use evaluate.py')
flags.DEFINE_bool('sample', False, help='load model.pt and run inference')

# UNet
flags.DEFINE_integer('ch', 128, help='base channel of UNet')

flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
flags.DEFINE_bool('conditional', False, help='conditional generation')
# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
# Training
flags.DEFINE_float('lr', 2e-4, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help='gradient norm clipping')
flags.DEFINE_integer('total_steps', 300001, help='total training steps')
flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 64, help='training batch size')
flags.DEFINE_integer('sample_batch_size', 0, help='sampling batch size; 0 reuses batch_size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help='ema decay rate')
flags.DEFINE_bool('parallel', False, help='multi gpu training')
flags.DEFINE_bool('cfg', False, help='whether to train unconditional generation with with 10%  probability')

# Dataset
flags.DEFINE_string('data_type', 'cifar100', help='data type, must be in \
                                                  [cifar10, cifar100, cifar10lt, cifar100lt, \
                                                  imgnetlt, tinyimgnetlt, \
                                                  placeslt]')
flags.DEFINE_string('data_path', '.cache/data', help='data path')
flags.DEFINE_float('imb_factor', 0.01, help='imb_factor for long tail dataset')
flags.DEFINE_string('train_manifest', '', help='ImageNet-LT train manifest for data_type=imagenet_lt')
flags.DEFINE_integer('num_class', 0, help='number of class of the pretrained model')

# Logging & Sampling
flags.DEFINE_string('logdir', './logs/', help='log directory')
flags.DEFINE_integer('sample_size', 64, 'sampling size of images')
flags.DEFINE_integer('sample_step', 50000, help='frequency of sampling')
flags.DEFINE_string('sample_method', 'cfg', help='sampling method, must be in [ddim, ddpm / cfg, uncond]')
flags.DEFINE_float('omega', 1.5, help='guidance strength')
flags.DEFINE_string('omega_scheduler', 'constant', help='omega scheduler')
flags.DEFINE_float('gamma', 100, help='only works when omega_scheduler is "gamma"')
# DDIM Sampling Related
flags.DEFINE_integer('ddim_skip_step', 10, help="ddim step")

# Evaluation
flags.DEFINE_integer('save_step', 100000, help='frequency of saving checkpoints, 0 to disable during training')
# flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')
flags.DEFINE_string('feats_cache', './stats/cifar10_feats.npy', help='FEATS cache')
flags.DEFINE_string('sample_name', 'saved', help='name for a set of samples to be saved or to be evaluated')
flags.DEFINE_bool('sampled', False, help='evaluate sampled images')
flags.DEFINE_string('sample_output', '', help='absolute .npy path for generated arrays')
flags.DEFINE_bool('uniform_labels', False, help='sample an exact class-uniform label schedule during evaluation')
# Extra Evaluation Metrics
flags.DEFINE_bool('prd', False, help='evaluate precision and recall (F_beta), only evaluated with 50k samples')
flags.DEFINE_bool('improved_prd', False, help='evaluate improved precision and recall, only evaluated with 50k samples')
# Evaluation for a specific category
flags.DEFINE_string('category', 'all', help='the specific category for evaluation')


# SOTA Methods
# OCLT hyperparameter
flags.DEFINE_bool('transfer_x0',False,help='transfering x0 to other index based on L2 norm')
flags.DEFINE_bool('transfer_tr_tau',False,help='transfering x0 with adjusted tau')
flags.DEFINE_bool('transfer_mixing',False,help='whether to using transfer')
flags.DEFINE_bool('bal_sample',False,help='whether to using transfer')
flags.DEFINE_string('transfer_mode', 'full', help='transfer_mode')
flags.DEFINE_float('tr_tau', 1.0, help='weight for transfer power')

# CBDM hyperparameter
flags.DEFINE_bool('cbdm', False, help='training with CBDM')
flags.DEFINE_float('cb_tau', 1.0, help='temperature for CBDM')

# CCUA hyperparameter
flags.DEFINE_float('ccua_al', 0.0, help='ccua al loss')
flags.DEFINE_float('ccua_ucl', 0.0, help='ccua ucl loss')
flags.DEFINE_bool('brs', False, help='whether to use batch resample')
flags.DEFINE_float('brs_factor', 0.1, help='imb_factor after batch resample')

device = torch.device('cuda:0')


def uniform_sampling(n, N, k):
    return np.stack([np.random.randint(int(N/n)*i, int(N/n)*(i+1), k) for i in range(n)])

def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))

def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x, y

def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup

def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr



def train():
    tran_transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize([FLAGS.img_size, FLAGS.img_size], antialias=True)
    ])

    if FLAGS.data_type == 'cifar10':
        dataset = CIFAR10(
                root=FLAGS.data_path,
                train=True,
                download=True,
                transform=tran_transform
                )
    elif FLAGS.data_type == 'cifar100':
        dataset = CIFAR100(
                root=FLAGS.data_path,
                train=True,
                download=True,
                transform=tran_transform)
    elif FLAGS.data_type == 'cifar10lt':
        dataset = ImbalanceCIFAR10(
                root=FLAGS.data_path,
                imb_type='exp',
                imb_factor=FLAGS.imb_factor,
                rand_number=0,
                brs=FLAGS.brs,
                brs_factor=FLAGS.brs_factor,
                train=True,
                transform=tran_transform,
                target_transform=None,
                download=True,
                )
    elif FLAGS.data_type == 'cifar100lt':
        dataset = ImbalanceCIFAR100(
                root=FLAGS.data_path,
                imb_type='exp',
                imb_factor=FLAGS.imb_factor,
                rand_number=0,
                brs=FLAGS.brs,
                brs_factor=FLAGS.brs_factor,
                train=True,
                transform=tran_transform,
                download=True,
        )
    elif FLAGS.data_type == 'imagenet_lt':
        if not FLAGS.train_manifest:
            raise ValueError('data_type=imagenet_lt requires --train_manifest')
        dataset = ImageNetLTManifest(root=FLAGS.data_path,
                                     manifest=FLAGS.train_manifest,
                                     num_class=FLAGS.num_class,
                                     transform=tran_transform)
    elif FLAGS.data_type == 'imgnetlt':
        FLAGS.data_path = FLAGS.data_path
        dataset = ImbalanceImageNet(root=FLAGS.data_path,
                                    split="train",
                                    imb_type='exp',
                                    imb_factor=FLAGS.imb_factor,
                                    rand_number=0,
                                    brs=FLAGS.brs,
                                    brs_factor=FLAGS.brs_factor,
                                    transform=tran_transform
                                    )
    elif FLAGS.data_type == 'tinyimgnetlt':
        FLAGS.data_path = FLAGS.data_path
        dataset = ImbalanceTinyImageNet(root=FLAGS.data_path,
                                        split="train",
                                        imb_type='exp',
                                        imb_factor=FLAGS.imb_factor,
                                        rand_number=0,
                                        brs=FLAGS.brs,
                                        brs_factor=FLAGS.brs_factor,
                                        transform=tran_transform
                                    )
    elif FLAGS.data_type == 'placeslt':
        FLAGS.data_path = FLAGS.data_path
        dataset = PlacesLT(root=FLAGS.data_path, #/data/datasets/Places/',
                           split='train-standard', small=True,
                           imb_type="exp",
                           imb_factor=FLAGS.imb_factor,
                           rand_number=0,
                           brs=FLAGS.brs,
                           brs_factor=FLAGS.brs_factor,
                           transform=tran_transform,
                           download=False)
    else:
        raise NotImplementedError(
            'Please enter a valid data type!!!.')


    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size,
        shuffle=True, num_workers=FLAGS.num_workers, drop_last=True)

    datalooper = infiniteloop(dataloader)
    ref_datalooper = None
    print('Dataset {} contains {} images with {} classes'.format(
        FLAGS.data_type, len(dataset.targets), len(np.unique(dataset.targets))))


    # get class weights for the current dataset
    def class_counter(all_labels):
        all_classes_count = torch.Tensor(np.unique(all_labels, return_counts=True)[1])
        return all_classes_count / all_classes_count.sum()
    weight = class_counter(dataset.targets).unsqueeze(0)
    print(weight, weight.shape)
    weight_transfer_matrix = weight.T @ weight
    weight_power_matrix = torch.pow(weight_transfer_matrix,FLAGS.tr_tau)


    net_model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout,
        cond=FLAGS.conditional, num_class=int(FLAGS.num_class))
    ema_model = copy.deepcopy(net_model)

    # training setup
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    trainer = GaussianDiffusionTrainer(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, dataset,
        FLAGS.num_class, FLAGS.cfg, weight,
        transfer_x0=FLAGS.transfer_x0,transfer_tr_tau=FLAGS.transfer_tr_tau,
        transfer_mode=FLAGS.transfer_mode,label_weight_tr = weight_power_matrix,
        cbdm=FLAGS.cbdm, cb_tau=FLAGS.cb_tau,
        ccua_ucl=FLAGS.ccua_ucl, ccua_al=FLAGS.ccua_al,
        ).to(device)
    net_sampler = GaussianDiffusionSampler(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.num_class, FLAGS.img_size, FLAGS.var_type).to(device)
    ema_sampler = GaussianDiffusionSampler(
        ema_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.num_class, FLAGS.img_size, FLAGS.var_type).to(device)

    if FLAGS.resume:
        print(os.path.join(FLAGS.resume_dir,
                                       'ckpt_{}.pt'.format(FLAGS.ckpt_step)))
        ckpt = torch.load(os.path.join(FLAGS.resume_dir,
                                       'ckpt_{}.pt'.format(FLAGS.ckpt_step)),
                                        map_location='cpu')
        net_model.load_state_dict(ckpt['net_model'])
        ema_model.load_state_dict(ckpt['ema_model'])
        optim.load_state_dict(ckpt['optim'])
        sched.load_state_dict(ckpt['sched'])
        print('Loading checkpoint sussessfully from {}'.format(os.path.join(FLAGS.resume_dir,
                                       'ckpt_{}.pt'.format(FLAGS.ckpt_step))))

    if FLAGS.parallel:
        trainer = torch.nn.DataParallel(trainer)
        net_sampler = torch.nn.DataParallel(net_sampler)
        ema_sampler = torch.nn.DataParallel(ema_sampler)


    # log setup
    if not os.path.exists(os.path.join(FLAGS.logdir, 'sample')):
        os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
    else:
        print('LOGDIR already exists.')
    writer = SummaryWriter(FLAGS.logdir)
    writer.flush()

    # fix seeds for generation to keep generated images comparable
    fixed_x_T = torch.randn(min(FLAGS.sample_size, 100), 3, FLAGS.img_size, FLAGS.img_size)
    fixed_x_T = fixed_x_T.to(device)

    # backup all arguments
    with open(os.path.join(FLAGS.logdir, 'flagfile.txt'), 'w') as f:
        f.write(FLAGS.flags_into_string())

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))

    # start training
    with trange(0, FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            # train
            optim.zero_grad()
            # uncond_flag_from_out = False
            if ref_datalooper is not None:
                if torch.rand(1)[0] < 1/10:
                    x_0,y_0 = next(ref_datalooper)
                    # uncond_flag_from_out = True
                else:
                    x_0,y_0 = next(datalooper)
            else:
                x_0,y_0 = next(datalooper)

            x_0 = x_0.to(device)
            y_0 = y_0.to(device)
            loss_ddpm, loss_cbdm, loss_ccua_ucl, loss_ccua_al = trainer(x_0, y_0)
            loss_ddpm = loss_ddpm.mean()
            loss_cbdm = loss_cbdm.mean()
            loss_ccua_ucl = loss_ccua_ucl.mean()
            loss_ccua_al = loss_ccua_al.mean()

            loss = loss_ddpm \
                   + loss_cbdm \
                   + loss_ccua_ucl * FLAGS.ccua_ucl \
                   + loss_ccua_al * FLAGS.ccua_al
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)

            # logs
            writer.add_scalar('loss', loss, step)
            writer.add_scalar('loss_ddpm', loss_ddpm, step)
            writer.add_scalar('loss_cbdm', loss_cbdm, step)
            writer.add_scalar('loss_ccua_ucl', loss_ccua_ucl, step)
            writer.add_scalar('loss_ccua_al', loss_ccua_al, step)
            pbar.set_postfix(loss='%.5f' % loss,
                             loss_ddpm='%.5f' % loss_ddpm,
                             loss_cbdm='%.5f' % loss_cbdm,
                             loss_ccua_ucl='%.5f' % loss_ccua_ucl,
                             loss_ccua_al='%.5f' % loss_ccua_al,)

            # sample
            if FLAGS.sample_step > 0 and step != 0 and step % FLAGS.sample_step == 0:
                net_model.eval()
                with torch.no_grad():
                    x_0, _  = ema_sampler(fixed_x_T)
                    grid = (make_grid(x_0) + 1) / 2
                    path = os.path.join(
                        FLAGS.logdir, 'sample', '%d.png' % step)
                    save_image(grid, path)
                    writer.add_image('sample', grid, step)
                net_model.train()

            # save
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                ckpt = {
                    'net_model': net_model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'sched': sched.state_dict(),
                    'optim': optim.state_dict(),
                    'step': step + FLAGS.ckpt_step,
                    'fixed_x_T': fixed_x_T,
                }
                atomic_torch_save(
                    ckpt,
                    os.path.join(FLAGS.logdir, 'ckpt_{}.pt'.format(step + FLAGS.ckpt_step)),
                )

            # # evaluate
            # if FLAGS.eval_step > 0 and step % FLAGS.eval_step == 0:
            #     # net_IS, net_FID, _ = evaluate(net_sampler, net_model)
            #     # ema_IS, ema_FID = evaluate(ema_sampler, ema_model, False)
            #     ema_IS, ema_FID, _, _ = evaluate_ddim(ema_sampler, ema_model, False)
            #
            #     metrics = {
            #         'IS': ema_IS[0],
            #         'IS_std': ema_IS[1],
            #         'FID': ema_FID
            #     }
            #     print(step, metrics)
            #     pbar.write(
            #         '%d/%d ' % (step, FLAGS.total_steps) +
            #         ', '.join('%s:%.5f' % (k, v) for k, v in metrics.items()))
            #     for name, value in metrics.items():
            #         writer.add_scalar(name, value, step)
            #     writer.flush()
            #     with open(os.path.join(FLAGS.logdir, 'eval.txt'), 'a') as f:
            #         metrics['step'] = step
            #         f.write(json.dumps(metrics) + '\n')
    writer.close()



def sample_ddim(sampler, model, sampled):
    if FLAGS.category == 'all': classes = torch.arange(FLAGS.num_class)
    elif FLAGS.category == 'head': classes = torch.arange(FLAGS.num_class//3)
    elif FLAGS.category == 'body': classes = torch.arange(FLAGS.num_class//3, 1+2*FLAGS.num_class//3)
    elif FLAGS.category == 'tail': classes = torch.arange(1+2*FLAGS.num_class//3, FLAGS.num_class)
    else: classes = torch.tensor(list(map(int, FLAGS.category.replace(' ', '').split(','))))
    print('sampled: ', sampled)
    if not sampled:
        model.eval()
        sample_batch_size = int(FLAGS.sample_batch_size or FLAGS.batch_size)
        if sample_batch_size <= 0:
            raise ValueError('sample_batch_size must be positive')
        with torch.inference_mode():
            images = []
            labels = []
            desc = 'generating images'
            for i in trange(0, FLAGS.num_images, sample_batch_size, desc=desc):
                batch_size = min(sample_batch_size, FLAGS.num_images - i)
                x_T = torch.randn(
                    (batch_size, 3, FLAGS.img_size, FLAGS.img_size),
                    device=device,
                )
                if FLAGS.uniform_labels:
                    batch_labels = classes[torch.arange(i, i + batch_size) % len(classes)].to(device)
                else:
                    batch_idx = torch.randint(len(classes), size=(x_T.shape[0],))
                    batch_labels = classes[batch_idx].to(device)
                print(f'label len: {batch_labels.shape[0]}, label range: {batch_labels.min()} ~ {batch_labels.max()}')
                batch_images = sampler(x_T, y=batch_labels,
                                       method=FLAGS.sample_method,
                                       skip=FLAGS.ddim_skip_step,
                                       return_timestep_status=False)
                images.append(((batch_images + 1) / 2).cpu())
                if FLAGS.sample_method!='uncond' and batch_labels is not None:
                    labels.append(batch_labels)
            images = torch.cat(images, dim=0).cpu().numpy()
        _ltx_upstream = os.path.join(FLAGS.logdir, '{}_{}_samples_ema_{}.npy'.format(
                             FLAGS.sample_method, FLAGS.omega,
                             FLAGS.sample_name))
        _ltx_samples_path = FLAGS.sample_output or _ltx_upstream
        np.save(_ltx_samples_path, images)
        if FLAGS.sample_method != 'uncond':
            labels = torch.cat(labels, dim=0).cpu().numpy()
            _ltx_labels_path = (FLAGS.sample_output + '.labels.npy') if FLAGS.sample_output else \
                os.path.join(FLAGS.logdir, '{}_{}_labels_ema_{}.npy'.format(
                                 FLAGS.sample_method, FLAGS.omega,
                                 FLAGS.sample_name))
            np.save(_ltx_labels_path, labels)
        model.train()
    else:
        labels = None
        images = np.load(os.path.join(FLAGS.logdir, '{}_{}_samples_ema_{}.npy'.format(
                                      FLAGS.sample_method, FLAGS.omega,
                                      FLAGS.sample_name)))

        if FLAGS.sample_method != 'uncond':
            labels = np.load(os.path.join(FLAGS.logdir, '{}_{}_labels_ema_{}.npy'.format(
                                          FLAGS.sample_method, FLAGS.omega,
                                          FLAGS.sample_name)))

    # images = sorted(images, key=lambda x: x.mean(), reverse=True)
    save_image(
        torch.tensor(images[:256]),
        os.path.join(FLAGS.logdir, 'visual_ema_{}_{}_{}.png'.format(
                                    FLAGS.sample_method, FLAGS.omega, FLAGS.sample_name)),
        nrow=8)

    # (IS, IS_std), FID, prd_score, ipr = get_inception_and_fid_score_new(
    #     images, labels, FLAGS.fid_cache, FLAGS.feats_cache, num_images=FLAGS.num_images,
    #     use_torch=FLAGS.fid_use_torch, FLAGS=FLAGS)
    #
    # return (IS, IS_std), FID, prd_score, ipr


def sample():
    # model setup
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout,
        cond=FLAGS.conditional, num_class=FLAGS.num_class)

    sampler = GaussianDiffusionSamplerDDIM(
              model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
              var_type=FLAGS.var_type, omega=FLAGS.omega,
              omega_scheduler=FLAGS.omega_scheduler, gamma=FLAGS.gamma, cond=FLAGS.conditional).to(device)
    if FLAGS.parallel:
        sampler = torch.nn.DataParallel(sampler)
    FLAGS.sample_name = '{}_CATEGORY{}_N{}_STEP{}_Omega{}_OmegaScheduler{}_Gamma{}_SEED{}_DDIM{}'.format(FLAGS.sample_name,
                                                                                   FLAGS.category,
                                                                                   FLAGS.num_images,
                                                                                   FLAGS.ckpt_step,
                                                                                   FLAGS.omega,
                                                                                   FLAGS.omega_scheduler, FLAGS.gamma,
                                                                                   FLAGS.seed, 1000//FLAGS.ddim_skip_step)

    # load model and evaluate
    if FLAGS.ckpt_step >= 0:
        ckpt = torch.load(os.path.join(FLAGS.logdir, f'ckpt_{FLAGS.ckpt_step}.pt'))
    else:
        ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'))

    model.load_state_dict(ckpt['net_model'])
    model.load_state_dict(ckpt['ema_model'])

    if not FLAGS.sampled:
        model.load_state_dict(ckpt['ema_model'])
    else:
        model = None

    sample_ddim(sampler, model, FLAGS.sampled)


    # (IS, IS_std), FID, prd_score, ipr = evaluate_ddim(sampler, model, FLAGS.sampled)

    # print('logdir', FLAGS.logdir)
    # print("Model(EMA): IS: %6.5f(%.5f), FID/%s: %7.5f \n" % (IS, IS_std, FLAGS.data_type.upper(), FID))
    # print("Improved PRD: %6.5f, RECALL: %7.5f \n" % (ipr[0], ipr[1]))
    # print("F_beta: %6.5f, F_beta_inv: %7.5f \n" % (prd_score[0], prd_score[1]))
    #
    # with open(os.path.join(FLAGS.logdir, 'res_ema_{}.txt'.format(FLAGS.sample_name)), 'a+') as f:
    #     f.write(
    #         "Settings: NUM:{} EPOCH:{}, OMEGA:{}, METHOD:{} \n".format(FLAGS.num_images, FLAGS.ckpt_step, FLAGS.omega,
    #                                                                    FLAGS.sample_method))
    #     f.write("Model(EMA): IS: %6.5f(%.5f), FID/%s: %7.5f \n" % (IS, IS_std, FLAGS.data_type.upper(), FID))
    #     f.write("Improved PRD: %6.5f, RECALL: %7.5f \n" % (ipr[0], ipr[1]))
    #     f.write("F_beta: %6.5f, F_beta_inv: %7.5f \n" % (prd_score[0], prd_score[1]))
    #
    #     f.close()


def main(argv):
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action='ignore', category=FutureWarning)

    torch.manual_seed(FLAGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(FLAGS.seed)
        torch.cuda.manual_seed_all(FLAGS.seed)

    if FLAGS.train:
        train()
    if FLAGS.eval:
        raise NotImplementedError("Please use evaluate.py to evaluate the model!")
    if FLAGS.sample:
        print('Sampling...')
        print(f'Image Size: {FLAGS.img_size}')
        sample()

if __name__ == '__main__':
    app.run(main)
