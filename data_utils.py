from os import listdir
from os.path import join, isfile
import numbers, random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataset import Dataset, ConcatDataset
from torch.utils.data.sampler import BatchSampler, Sampler
from torchvision.transforms import (Compose, RandomHorizontalFlip, RandomVerticalFlip, Resize,
                                    ToTensor, Normalize)
from torchvision.transforms import functional as F
from utils import get_config

datasets = get_config('./configs/datasets.yml')

channel_stats = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])     

def trainset(selected_datasets, output_img_size, add_noise=False):
    l_datasets = []
    for dataset_name in selected_datasets:
        dataset_config = datasets[dataset_name]
        l_transforms = []
        l_transforms += [MyRandomCrop(output_img_size)]
        l_transforms += [Resize(output_img_size, interpolation=Image.BICUBIC)]
        if add_noise:
            l_transforms += [AddDynamicGaussianNoise(std=2)]
        # random flips
        l_transforms += [RandomHorizontalFlip()]
        # to tensor
        l_transforms += [ToTensor()]
        l_transforms += [Normalize(**channel_stats)]

        dataset = ImagePairFromFolders(dataset_config['root_dirs'], Compose(l_transforms))
        dataset.name = dataset_name
        l_datasets += [dataset]

    return ConcatDataset(l_datasets)


def testset(selected_datasets, output_img_size, concat=True):
    l_datasets = []
    for dataset_name in selected_datasets:
        dataset_config = datasets[dataset_name]
        l_transforms = []
        # crop first with largest possible size by keeping ratio
        l_transforms += [MyRandomCrop(output_img_size, center_crop=True)]
        # then resize to
        l_transforms += [Resize(output_img_size, interpolation=Image.BICUBIC)]
        # to tensor
        l_transforms += [ToTensor()]
        l_transforms += [Normalize(**channel_stats)]

        dataset = ImagePairFromFolders(dataset_config['root_dirs'], Compose(l_transforms))
        dataset.name = dataset_name
        l_datasets += [dataset]
    if concat:
        return ConcatDataset(l_datasets)
    else:
        return l_datasets


def trainloader(dataset, num_samples, batch_size, num_workers):
    sampler = SubsetSampler(range(len(dataset)), num_samples=num_samples, randperm=True, replacement=True)
    batch_sampler = BatchSampler(sampler, batch_size, drop_last=True)
    return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers, pin_memory=True)


def testloader(dataset, num_samples, batch_size, num_workers, watch_only=False):
    max_num_test_img = 50
    if isinstance(dataset, list):
        l_loader = []
        for _subset in dataset:
            _subset_name = _subset.name
            dataset_config = datasets[_subset_name]
            watch_list = dataset_config['watch'] if 'watch' in dataset_config else []
            if watch_only:
                _subset = Subset(_subset, watch_list)
            else:
                _subset = Subset(_subset, range(min(len(_subset), max_num_test_img)))
            _loader = DataLoader(_subset, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True, drop_last=False)
            _loader.watch = watch_list
            _loader.name = _subset_name
            l_loader += [_loader]
        return l_loader
    else:
        sampler = SubsetSampler(range(len(dataset)), num_samples=num_samples, randperm=False)
        batch_sampler = BatchSampler(sampler, batch_size, drop_last=False)
        return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers, pin_memory=True)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.jpeg', '.bmp'])

class ImagePairFromFolders(Dataset):
    def __init__(self, dataset_dirs, transform):
        super(ImagePairFromFolders, self).__init__()
        assert len(dataset_dirs) > 0
        self.image_filenames = [sorted([join(root, x) for x in listdir(root) if is_image_file(x)]) for root in dataset_dirs]

        # check file exists
        for l in self.image_filenames:
            for f in l:
                assert isfile(f)

        self.transform = transform

    def __getitem__(self, index):
        images = [Image.open(filenames[index]) for filenames in self.image_filenames]
        if len(images) == 1:
            return self.transform(images[0])
        else:
            seed = random.randint(0, 2**32)
            l_image = []
            for x in images:
                random.seed(seed)
                l_image += [self.transform(x)]
            return l_image

    def __len__(self):
        return len(self.image_filenames[0])


class UnNormalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        out = tensor.new(*tensor.size())
        for z in range(out.shape[1]):
            out[:,z,:,:] = tensor[:,z,:,:] * self.std[z] + self.mean[z]
        return out

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


unnormalize = UnNormalize(**channel_stats)


class SubsetSampler(Sampler):

    def __init__(self, indices, num_samples=None, randperm=False, replacement=False, weights=None):

        self.indices = indices
        self.num_samples = len(indices) if num_samples is None else num_samples

        self.randperm = randperm
        self.replacement = replacement
        if weights is None:
            self.weights = torch.ones((len(self.indices),), dtype=torch.double)
        else:
            self.weights = torch.tensor(weights, dtype=torch.double)

    def __iter__(self):
        if (self.num_samples == len(self.indices)) and (not self.randperm):
            # SubsetSequentialSampler
            return (self.indices[i] for i in range(len(self.indices)))
        elif self.randperm and (not self.replacement):
            # SubsetRandomSampler
            return (self.indices[i] for i in torch.randperm(len(self.indices)))
        elif self.randperm and self.replacement:
            return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, self.replacement))
        else:
            raise NotImplementedError

    def __len__(self):
        return self.num_samples


class MyRandomCrop(object):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size, center_crop=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.center_crop = center_crop

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        crop_ratio = min(float(h)/output_size[0], float(w)/output_size[1])
        th, tw = round(crop_ratio*output_size[0]), round(crop_ratio*output_size[1])
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        i, j, h, w = self.get_params(img, self.size)
        if self.center_crop:
            return F.center_crop(img, (h, w))
        else:
            return F.crop(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class AddDynamicGaussianNoise(object):
    def __init__(self, std=5):
        self.std = float(std)

    def __call__(self, img):
        np_img = np.array(img, dtype=np.float32)
        np_img += np.random.normal(0., self.std, np_img.shape)
        np_img = np.clip(np_img, 0., 255.).astype('uint8')
        img = Image.fromarray(np_img, 'RGB')
        return img

