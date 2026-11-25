import math
import numpy as np
from resnet import resnet18
from ccar import ClusterContrastiveRelabeler, ccar_loss
from objectives import (
    auem_regularization_loss,
    ccar_batch_targets,
    dsruc_loss,
    split_uncertainty_regions,
)
import torch.utils.data as data
from torchvision import transforms
import cv2
import pandas as pd
import os, torch
import torch.nn as nn
import image_utils
import argparse, random
import json
from pathlib import Path
torch.cuda.empty_cache()

# from torchsampler import ImbalancedDatasetSampler

CHECKPOINT_FORMAT_VERSION = 2
IMPORTANCE_SOURCE = 'normalized_mu'
UNCERTAINTY_REPRESENTATION = 'pairwise_normalized_vector_mean'
RESUME_MUTABLE_FIELDS = frozenset({
    'checkpoint', 'pretrained', 'save_dir', 'skip_test_eval',
    'target_val_accuracy',
})
WARM_START_EXCLUDED_PREFIXES = ('alpha.', 'log_var.', 'fc.')
BASE_RESUME_KEYS = (
    'implementation_version', 'config', 'iter', 'best_val_accuracy',
    'model_state_dict', 'fc_state_dict', 'ccar_state_dict',
    'optimizer_state_dict', 'scheduler_state_dict', 'ema_decay',
)

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Train the DSRUL RAF-DB implementation.')
    parser.add_argument('--raf_path', type=str, required=True,
                        help='Path to the RAF-DB root directory.')
    parser.add_argument('--train_label_file', type=str, default='0.1noise_train.txt',
                        help='Noisy RAF-DB training labels in EmoLabel (or an absolute path).')
    parser.add_argument('--clean_label_file', type=str, default='list_patition_label.txt',
                        help='Clean RAF-DB labels used for validation/test (or an absolute path).')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Fraction of official training images reserved for clean validation.')
    parser.add_argument('--split_seed', type=int, default=0,
                        help='Seed for the fixed, class-stratified train/validation split.')
    parser.add_argument('--save_dir', type=str, default=None,
                        help="Directory for this run's logs and best checkpoint.")
    parser.add_argument('--target_val_accuracy', type=float, default=0.885,
                        help='Stop after validation accuracy reaches this value.')
    parser.add_argument('--skip_test_eval', action='store_true',
                        help='Skip official-test evaluation during tuning runs.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Full training checkpoint to resume (model, CCAR, optimizer, and epoch).')
    parser.add_argument('--pretrained', type=str, default='',
                        help='Optional model weights for an exact name/shape-compatible warm start.')
    parser.add_argument('--beta', type=float, default=0.2, help='CCAR threshold baseline (equation 11).')
    parser.add_argument('--gamma', type=float, default=0.6, help='CCAR importance adjustment (equation 11).')
    parser.add_argument('--alpha', type=float, default=0.4, help='CCAR geometric adjustment (equation 11).')
    parser.add_argument('--prototype_momentum', type=float, default=0.9, help='CCAR prototype EMA coefficient.')
    parser.add_argument('--ccar_temperature', type=float, default=0.1, help='CCAR distance temperature.')
    parser.add_argument('--relabel_epoch', type=int, default=8,
                        help='Enable batch-local CCAR relabeling starting at this epoch.')
    parser.add_argument('--margin_1', type=float, default=0.5,
                        help='AUEM regularization offset delta.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--optimizer', choices=('adam', 'sgd'), default='adam',
                        help='Optimizer.')
    parser.add_argument('--lr', type=float, default=0.0002, help='Initial learning rate (Adam/SGD).')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='L2 weight decay for the backbone and classifier optimizer.')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.99,
                        help='Per-epoch exponential learning-rate decay. Use 0.9 to restore the previous schedule.')
    parser.add_argument('--lr_scheduler', choices=('plateau', 'exponential', 'cosine'), default='exponential',
                        help='Learning-rate schedule; plateau tracks validation loss, cosine decays smoothly.')
    parser.add_argument('--lr_plateau_factor', type=float, default=0.5,
                        help='Learning-rate multiplier after a validation-loss plateau.')
    parser.add_argument('--lr_plateau_patience', type=int, default=5,
                        help='Epochs without validation-loss improvement before reducing LR.')
    parser.add_argument('--lr_min', type=float, default=1e-6,
                        help='Minimum learning rate for the plateau schedule.')
    parser.add_argument('--train_sampler', choices=('shuffle', 'inverse_sqrt'), default='shuffle',
                        help='Training sampler; inverse_sqrt moderately reduces class-frequency imbalance.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=1, type=int, help='Number of data loading workers (default: 1).')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs.')
    parser.add_argument('--drop_rate', type=float, default=0.2, help='Drop out rate.')
    parser.add_argument('--rotation_degrees', type=float, default=0.0,
                        help='Maximum absolute random rotation for training images; 0 disables it.')
    parser.add_argument('--ema_decay', type=float, default=0.0,
                        help='Epoch-wise EMA decay for validation/checkpoint weights; 0 disables EMA.')
    parser.add_argument('--out_dimension', type=int, default=64, help='feature dimension')

    return parser.parse_args(argv)


def implementation_version():
    return (Path(__file__).resolve().parents[1] / 'VERSION').read_text(
        encoding='utf-8').strip()


def method_contract(args):
    return {
        'importance_source': IMPORTANCE_SOURCE,
        'embedding_dim': args.out_dimension,
        'uncertainty_representation': UNCERTAINTY_REPRESENTATION,
        'backbone': 'resnet18',
    }


def resume_config(config):
    """Return behavior-defining CLI values that must remain fixed on resume."""
    values = vars(config) if not isinstance(config, dict) else config
    return {
        name: value for name, value in sorted(values.items())
        if name not in RESUME_MUTABLE_FIELDS
    }


def required_resume_keys(args):
    keys = list(BASE_RESUME_KEYS)
    if args.ema_decay > 0.0:
        keys.extend(('resume_model_state_dict', 'resume_fc_state_dict'))
    return tuple(keys)


def validate_resume_checkpoint(checkpoint, args):
    if not isinstance(checkpoint, dict):
        raise ValueError('Resume checkpoint must be a dictionary.')
    version = checkpoint.get('checkpoint_format_version')
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            'This is not a V2 checkpoint and cannot be resumed. V2 changes '
            'the AUEM importance input and restores vector uncertainty; use '
            '--pretrained for a shape-compatible warm start instead.')

    expected = method_contract(args)
    observed = checkpoint.get('method_contract')
    if observed != expected:
        raise ValueError(
            'Checkpoint method contract mismatch: expected {}, got {}'.format(
                expected, observed))

    saved_config = checkpoint.get('config')
    if not isinstance(saved_config, dict):
        raise ValueError('Resume checkpoint is missing its resolved CLI config.')
    expected_config = resume_config(args)
    observed_config = resume_config(saved_config)
    if observed_config != expected_config:
        changed = {
            name: {
                'checkpoint': observed_config.get(name, '<missing>'),
                'current': expected_config.get(name, '<missing>'),
            }
            for name in sorted(set(observed_config) | set(expected_config))
            if observed_config.get(name, '<missing>')
            != expected_config.get(name, '<missing>')
        }
        raise ValueError(
            'Resume configuration mismatch for behavior-defining values: {}'.format(
                changed))

    missing_keys = [
        key for key in required_resume_keys(args) if key not in checkpoint
    ]
    if missing_keys:
        raise ValueError(
            'Resume checkpoint is missing required keys: {}'.format(missing_keys))
    if checkpoint['ema_decay'] != args.ema_decay:
        raise ValueError(
            'Checkpoint EMA decay mismatch: expected {}, got {}'.format(
                args.ema_decay, checkpoint['ema_decay']))


def checkpoint_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError('Pretrained checkpoint must contain a state dictionary.')
    for container_key in ('state_dict', 'model_state_dict'):
        candidate = checkpoint.get(container_key)
        if isinstance(candidate, dict):
            return candidate
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise ValueError(
        'Pretrained checkpoint must be a raw state dict or contain state_dict/model_state_dict.')


def load_shape_compatible_state(model, source_state):
    """Warm-start only exact name/shape matches and report every skipped key."""
    target_state = model.state_dict()
    compatible = {}
    skipped = {}

    for source_key, value in source_state.items():
        candidates = [source_key]
        if source_key.startswith('module.'):
            candidates.append(source_key[len('module.'):])
        else:
            candidates.append('module.' + source_key)
        target_key = next((key for key in candidates if key in target_state), None)

        if target_key is None:
            skipped[source_key] = 'missing target key'
            continue
        unwrapped_key = target_key.removeprefix('module.')
        excluded_prefix = next(
            (prefix for prefix in WARM_START_EXCLUDED_PREFIXES
             if unwrapped_key.startswith(prefix)), None)
        if excluded_prefix is not None:
            skipped[source_key] = (
                '{} tensors are not warm-start targets'.format(
                    excluded_prefix.rstrip('.')))
            continue
        if not torch.is_tensor(value):
            skipped[source_key] = 'value is not a tensor'
            continue
        if target_state[target_key].shape != value.shape:
            skipped[source_key] = 'shape {} does not match {}'.format(
                tuple(value.shape), tuple(target_state[target_key].shape))
            continue
        compatible[target_key] = value

    if not compatible:
        raise ValueError('No shape-compatible pretrained tensors were found.')
    incompatible = model.load_state_dict(compatible, strict=False)
    return {
        'loaded': sorted(compatible),
        'skipped': skipped,
        'missing_after_load': sorted(incompatible.missing_keys),
        'unexpected_after_load': sorted(incompatible.unexpected_keys),
    }


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def clone_state_dict_to_cpu(state_dict):
    return {name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()}


def restore_ema_states(checkpoint, ema_decay):
    if checkpoint is None or ema_decay <= 0.0:
        return None, None
    return (
        clone_state_dict_to_cpu(checkpoint['model_state_dict']),
        clone_state_dict_to_cpu(checkpoint['fc_state_dict']),
    )


def update_ema_state(ema_state, current_state, decay):
    with torch.no_grad():
        for name, current_tensor in current_state.items():
            current_tensor = current_tensor.detach().cpu()
            if ema_state[name].is_floating_point():
                ema_state[name].mul_(decay).add_(current_tensor, alpha=1.0 - decay)
            else:
                ema_state[name].copy_(current_tensor)


def load_label_table(raf_path, label_file):
    path = label_file if os.path.isabs(label_file) else os.path.join(raf_path, 'EmoLabel', label_file)
    labels = pd.read_csv(path, sep=r'\s+', header=None, names=['name', 'label'])
    if labels.empty or labels['name'].isna().any() or labels['label'].isna().any():
        raise ValueError('Missing or empty labels in {}'.format(path))
    if labels['name'].duplicated().any():
        raise ValueError('Duplicate image names in {}'.format(path))
    if not labels['label'].isin(range(1, 8)).all():
        raise ValueError('RAF-DB labels must be integers from 1 to 7: {}'.format(path))
    return labels.set_index('name')


def make_raf_splits(raf_path, train_label_file, clean_label_file, val_ratio, split_seed):
    if not 0 < val_ratio < 1:
        raise ValueError('--val_ratio must be between 0 and 1')

    clean = load_label_table(raf_path, clean_label_file)
    noisy = load_label_table(raf_path, train_label_file)
    clean_train = clean.loc[clean.index.str.startswith('train')]
    clean_test = clean.loc[clean.index.str.startswith('test')]
    noisy_train = noisy.loc[noisy.index.str.startswith('train')]
    if clean_train.empty or clean_test.empty:
        raise ValueError('Clean labels must contain both train* and test* images')
    if set(noisy_train.index) != set(clean_train.index):
        raise ValueError('Noisy and clean label files must contain the same train* images')

    changed = (noisy_train.loc[clean_train.index, 'label'].to_numpy()
               != clean_train['label'].to_numpy())
    print('Observed RAF-DB training-label noise: {}/{} ({:.2%})'.format(
        int(changed.sum()), len(changed), float(changed.mean())))

    rng = np.random.default_rng(split_seed)
    validation_names = set()
    for label in sorted(clean_train['label'].unique()):
        class_names = clean_train.index[clean_train['label'] == label].to_numpy(copy=True)
        rng.shuffle(class_names)
        val_count = max(1, round(len(class_names) * val_ratio))
        if val_count >= len(class_names):
            raise ValueError('Class {} has too few training images for validation'.format(label))
        validation_names.update(class_names[:val_count].tolist())

    train_names = [name for name in clean_train.index if name not in validation_names]
    val_names = [name for name in clean_train.index if name in validation_names]
    train_records = noisy_train.loc[train_names].reset_index()
    val_records = clean_train.loc[val_names].reset_index()
    test_records = clean_test.reset_index()
    return train_records, val_records, test_records


class RafDataSet(data.Dataset):
    def __init__(self, raf_path, records, phase, transform=None, basic_aug=False):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path

        file_names = records['name'].to_numpy()
        self.label = records['label'].to_numpy(dtype=np.int64, copy=True) - 1

        self.file_paths = []
        # use raf aligned images for training/testing
        for f in file_names:
            f = f.split(".")[0]
            f = f + "_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image', 'aligned', f)
            self.file_paths.append(path)

        self.basic_aug = basic_aug
        self.aug_func = [image_utils.flip_image, image_utils.add_gaussian_noise]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1]  # BGR to RGB
        label = self.label[idx]

        # augmentation
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label, idx


class MinimumSizeBatchSampler(data.Sampler):
    """Merge a too-small tail into the preceding batch without dropping data."""
    def __init__(self, sampler, batch_size, minimum_batch_size):
        if batch_size < minimum_batch_size:
            raise ValueError(
                'batch_size must be at least {}'.format(minimum_batch_size))
        if len(sampler) < minimum_batch_size:
            raise ValueError(
                'training set must contain at least {} samples'.format(
                    minimum_batch_size))
        self.sampler = sampler
        self.batch_size = batch_size
        self.minimum_batch_size = minimum_batch_size

    def __iter__(self):
        pending = None
        current = []
        for index in self.sampler:
            current.append(index)
            if len(current) == self.batch_size:
                if pending is not None:
                    yield pending
                pending = current
                current = []

        if pending is None:
            yield current
        elif current and len(current) < self.minimum_batch_size:
            yield pending + current
        else:
            yield pending
            if current:
                yield current

    def __len__(self):
        full_batches, remainder = divmod(len(self.sampler), self.batch_size)
        if full_batches == 0:
            return 1
        if remainder == 0 or remainder < self.minimum_batch_size:
            return full_batches
        return full_batches + 1


def initialize_weight_goog(m, n=''):
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)  # fan-out
        fan_in = 0
        if 'routing_fn' in n:
            fan_in = m.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        m.weight.data.uniform_(-init_range, init_range)
        m.bias.data.zero_()

def run_training():
    setup_seed(0)

    args = parse_args()
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError('--ema_decay must be in [0, 1).')
    if args.checkpoint and args.pretrained:
        raise ValueError('--checkpoint and --pretrained cannot be used together.')
    print(args)
    resume_checkpoint = None
    if args.checkpoint:
        resume_checkpoint = torch.load(
            args.checkpoint, map_location='cpu', weights_only=False)
        validate_resume_checkpoint(resume_checkpoint, args)

    default_save_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'runs', 'rafdb_noise0.1')
    save_folder = args.save_dir if args.save_dir else default_save_folder
    os.makedirs(save_folder, exist_ok=True)

    imagenet_pretrained = False
    cuda_device_count = torch.cuda.device_count()
    if cuda_device_count < 1:
        raise RuntimeError('The training entrypoint requires at least one CUDA device.')
    minimum_training_batch_size = 2 * cuda_device_count

    res18 = resnet18(
        pretrained=imagenet_pretrained, drop_rate=args.drop_rate, args=args,
        out_dim=args.out_dimension)
    res18 = torch.nn.DataParallel(res18).cuda()
    fc = nn.Linear(args.out_dimension, 7).cuda()

    if not imagenet_pretrained:
        for m in res18.modules():
            initialize_weight_goog(m)

    if args.pretrained:
        print("Loading pretrained weights...", args.pretrained)
        pretrained = torch.load(
            args.pretrained, map_location='cpu', weights_only=False)
        warm_start_report = load_shape_compatible_state(
            res18, checkpoint_state_dict(pretrained))
        print('Warm-start loaded tensors:', len(warm_start_report['loaded']))
        print('Warm-start skipped tensors:', len(warm_start_report['skipped']))
        for key, reason in sorted(warm_start_report['skipped'].items()):
            print('  skipped {}: {}'.format(key, reason))

    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomRotation(args.rotation_degrees),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        # transforms.RandomHorizontalFlip(p=1),
        # transforms.RandomRotation(15),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.20))])


    train_records, val_records, test_records = make_raf_splits(
        args.raf_path, args.train_label_file, args.clean_label_file,
        args.val_ratio, args.split_seed)
    with open(os.path.join(save_folder, 'split_manifest.json'), 'w', encoding='utf-8') as manifest:
        json.dump({
            'train_label_file': args.train_label_file,
            'clean_label_file': args.clean_label_file,
            'val_ratio': args.val_ratio,
            'split_seed': args.split_seed,
            'train_names': train_records['name'].tolist(),
            'validation_names': val_records['name'].tolist(),
            'test_names': test_records['name'].tolist(),
        }, manifest, ensure_ascii=False, indent=2)

    train_dataset = RafDataSet(args.raf_path, train_records, phase='train',
                               transform=data_transforms, basic_aug=True)

    print('Train set size:', train_dataset.__len__())
    if args.train_sampler == 'inverse_sqrt':
        class_counts = np.bincount(train_dataset.label, minlength=7).astype(np.float64)
        if (class_counts == 0).any():
            raise ValueError('Cannot use inverse_sqrt sampling when a class has no training examples.')
        sample_weights = torch.as_tensor(
            np.power(class_counts[train_dataset.label], -0.5), dtype=torch.double)
        train_sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, num_samples=len(train_dataset), replacement=True,
            generator=torch.Generator().manual_seed(0))
        print('Inverse-sqrt sampler class counts:', class_counts.astype(int).tolist())
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)

    train_batch_sampler = MinimumSizeBatchSampler(
        train_sampler, args.batch_size, minimum_training_batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               num_workers=args.workers,
                                               batch_sampler=train_batch_sampler,
                                               pin_memory=True,
                                               persistent_workers=args.workers > 0
                                               )

    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    val_dataset = RafDataSet(args.raf_path, val_records, phase='val', transform=data_transforms_val)
    print('Validation set size:', val_dataset.__len__())

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             # sampler=ImbalancedDatasetSampler(val_dataset),
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True,
                                             persistent_workers=args.workers > 0,
                                             drop_last=False
                                             )

    test_dataset = RafDataSet(args.raf_path, test_records, phase='test',
                              transform=data_transforms_val)
    print('Test set size:', test_dataset.__len__())
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.workers,
                                              shuffle=False,
                                              pin_memory=True,
                                              persistent_workers=args.workers > 0,
                                              drop_last=False)

    params = list(res18.parameters()) + list(fc.parameters())
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        raise ValueError("Optimizer not supported.")

    if args.lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.lr_plateau_factor,
            patience=args.lr_plateau_patience, threshold=1e-3,
            threshold_mode='abs', min_lr=args.lr_min)
    elif args.lr_scheduler == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.lr_decay_gamma)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr_min)
    res18 = res18.cuda()
    ccar = ClusterContrastiveRelabeler(
        num_classes=7, feature_dim=args.out_dimension,
        momentum=args.prototype_momentum, temperature=args.ccar_temperature,
        beta=args.beta, gamma=args.gamma, alpha=args.alpha).cuda()

    # 交叉熵损失函数Lwce
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = -1.0
    best_epoch = 0
    checkpoint_path = os.path.join(save_folder, 'best.pth')
    start_epoch = 1
    ema_res18_state, ema_fc_state = restore_ema_states(
        resume_checkpoint, args.ema_decay)
    if resume_checkpoint is not None:
        res18.load_state_dict(resume_checkpoint.get(
            'resume_model_state_dict', resume_checkpoint['model_state_dict']))
        fc.load_state_dict(resume_checkpoint.get(
            'resume_fc_state_dict', resume_checkpoint['fc_state_dict']))
        ccar.load_state_dict(resume_checkpoint['ccar_state_dict'])
        optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
        start_epoch = int(resume_checkpoint['iter']) + 1
        best_epoch = int(resume_checkpoint['iter'])
        best_acc = float(resume_checkpoint['best_val_accuracy'])
        torch.save(resume_checkpoint, checkpoint_path)
        print('Resumed from epoch {}. Continuing at epoch {} from {}.'.format(
            best_epoch, start_epoch, args.checkpoint))

    def evaluate(loader):
        res18.eval()
        fc.eval()
        loss_sum = 0.0
        correct = 0
        sample_count = 0
        with torch.no_grad():
            for imgs, targets, _ in loader:
                imgs = imgs.cuda()
                targets = targets.cuda()
                mu, _ = res18(imgs)
                outputs = fc(mu)
                batch_size = targets.size(0)
                loss_sum += criterion(outputs, targets).item() * batch_size
                correct += (outputs.argmax(dim=1) == targets).sum().item()
                sample_count += batch_size
        return correct / sample_count, loss_sum / sample_count

    for i in range(start_epoch, args.epochs + 1):
        epoch_lr = optimizer.param_groups[0]['lr']
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        train_sample_count = 0
        res18.train()
        fc.train()
        for batch_i, (imgs, targets, _) in enumerate(train_loader):
            batch_sz = imgs.size(0)
            train_sample_count += batch_sz
            iter_cnt += 1
            optimizer.zero_grad()

            imgs = imgs.cuda()
            targets = targets.cuda()
            mu, uncertainties, importance = res18(imgs, phase='train', return_importance=True)
            importance = importance.squeeze(1)
            high_uncertainty_idx, low_uncertainty_idx = split_uncertainty_regions(
                importance)
            auem_loss = auem_regularization_loss(
                importance, high_uncertainty_idx, low_uncertainty_idx,
                delta=args.margin_1)

            outputs = fc(mu)
            dsruc_loss_value, _ = dsruc_loss(
                mu, uncertainties, targets, fc,
                high_uncertainty_idx, low_uncertainty_idx)
            ccar_targets = ccar_batch_targets(
                ccar, mu, targets, importance, outputs,
                relabel_enabled=i >= args.relabel_epoch)

            # Equations (13)-(14): scaled-logit CCAR plus AUEM and DSRUC.
            loss = (ccar_loss(outputs, ccar_targets, importance)
                    + auem_loss + dsruc_loss_value)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicts = torch.max(outputs, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num.item()

        acc = correct_sum / float(train_sample_count)
        running_loss = running_loss / iter_cnt
        print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. LR: %.8g' % (
            i, acc, running_loss, epoch_lr))

        raw_res18_state = clone_state_dict_to_cpu(res18.state_dict())
        raw_fc_state = clone_state_dict_to_cpu(fc.state_dict())
        if args.ema_decay > 0.0:
            if ema_res18_state is None:
                ema_res18_state = {name: tensor.clone()
                                   for name, tensor in raw_res18_state.items()}
                ema_fc_state = {name: tensor.clone()
                                for name, tensor in raw_fc_state.items()}
            else:
                update_ema_state(ema_res18_state, raw_res18_state, args.ema_decay)
                update_ema_state(ema_fc_state, raw_fc_state, args.ema_decay)
            res18.load_state_dict(ema_res18_state)
            fc.load_state_dict(ema_fc_state)
            val_model_state = ema_res18_state
            val_fc_state = ema_fc_state
        else:
            val_model_state = raw_res18_state
            val_fc_state = raw_fc_state

        val_acc, val_loss = evaluate(val_loader)
        if args.ema_decay > 0.0:
            res18.load_state_dict(raw_res18_state)
            fc.load_state_dict(raw_fc_state)
            validation_label = 'Validation (EMA)'
        else:
            validation_label = 'Validation'
        print('[Epoch %d] %s accuracy:%.4f. Loss:%.3f' % (
            i, validation_label, val_acc, val_loss))
        if args.lr_scheduler == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = i
            checkpoint_payload = {
                'checkpoint_format_version': CHECKPOINT_FORMAT_VERSION,
                'implementation_version': implementation_version(),
                'method_contract': method_contract(args),
                'config': vars(args).copy(),
                'iter': i,
                'best_val_accuracy': best_acc,
                'model_state_dict': val_model_state,
                'fc_state_dict': val_fc_state,
                'ccar_state_dict': ccar.state_dict(),
                'train_labels': train_loader.dataset.label.copy(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'ema_decay': args.ema_decay,
            }
            if args.ema_decay > 0.0:
                checkpoint_payload['resume_model_state_dict'] = raw_res18_state
                checkpoint_payload['resume_fc_state_dict'] = raw_fc_state
            torch.save(checkpoint_payload, checkpoint_path)
            print('Best validation checkpoint saved:', checkpoint_path)

        if val_acc >= args.target_val_accuracy:
            print('Target validation accuracy {:.4f} reached at epoch {}.'.format(
                args.target_val_accuracy, i))
            break

    if not os.path.isfile(checkpoint_path):
        raise RuntimeError('Training ended without producing a best validation checkpoint.')
    if args.skip_test_eval:
        print('Official test evaluation skipped for this tuning run. Selected epoch: {}. Best validation accuracy: {:.4f}.'.format(
            best_epoch, best_acc))
    else:
        best_checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        res18.load_state_dict(best_checkpoint['model_state_dict'])
        fc.load_state_dict(best_checkpoint['fc_state_dict'])
        test_acc, test_loss = evaluate(test_loader)
        print('Final official-test accuracy: {:.4f}. Loss: {:.3f}. Selected epoch: {}. Best validation accuracy: {:.4f}.'.format(
            test_acc, test_loss, best_epoch, best_acc))


if __name__ == "__main__":
    run_training()
