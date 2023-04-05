import os, cv2, torch
import random
import shutil
import torch
import torchvision
import yaml
import ramps
from scipy import misc
import numpy as np

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def write_grid_grid(list_of_tensor, grid_batch_size=None, filename=None,
                    nocurrent=False, unnormalize=None, nrows=[8, 8], **kwargs):
    batch_size = list_of_tensor[0].shape[0]
    grid_batch_size = min(grid_batch_size, batch_size)
    list_grid = []
    for t in list_of_tensor:
        if grid_batch_size is not None:
            t = t[:grid_batch_size]
        if unnormalize is not None:
            t = unnormalize(t)
        g = torchvision.utils.make_grid(t, nrow=nrows[0], **kwargs)
        list_grid.append(torch.unsqueeze(g, 0).cpu())
    batch_grid = torch.cat(list_grid, 0)
    if filename is None:
        return torchvision.utils.make_grid(batch_grid, nrow=nrows[1], **kwargs)
    else:
        torchvision.utils.save_image(batch_grid, filename, nrow=nrows[1], **kwargs)
        if not nocurrent:
            current_path = os.path.join(os.path.split(filename)[0], 'current' + os.path.splitext(filename)[-1])
            torchvision.utils.save_image(batch_grid, current_path, nrow=nrows[1], **kwargs)

def save_checkpoint(state, dirpath, epoch, is_best=True, current_only=False):
    if current_only:
        filename = 'checkpoint.current.ckpt'
    else:
        filename = 'checkpoint.{}.ckpt'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    best_path = os.path.join(dirpath, 'best.ckpt')
    torch.save(state, checkpoint_path)
    if is_best and not current_only:
        shutil.copyfile(checkpoint_path, best_path)

class AverageMeterSet:
    def __init__(self, display_metrics=None, stateful_metrics=None):
        self.meters = {}
        self.display_metrics = set(display_metrics) if display_metrics else set()
        self.stateful_metrics = set(stateful_metrics) if stateful_metrics else set()

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=''):
        return {name + postfix: meter.val for name, meter in self.meters.items()}

    def averages(self, postfix='/avg'):
        return {name + postfix: meter.avg for name, meter in self.meters.items()}

    def sums(self, postfix='/sum'):
        return {name + postfix: meter.sum for name, meter in self.meters.items()}

    def counts(self, postfix='/count'):
        return {name + postfix: meter.count for name, meter in self.meters.items()}

    def display(self):
        meters_to_disp = {k: self.meters[k] for k in self.display_metrics}
        return {k: v.val if k in self.stateful_metrics else v.avg for k, v in meters_to_disp.items()}

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        from_cuda = False
        if images.is_cuda:
            from_cuda = True
            images = images.cpu()

        if self.pool_size == 0:
            return images

        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)

        if from_cuda:
            return return_images.cuda()
        else:
            return return_images

def get_dyn_weight(current_epoch, config):
    if isinstance(config, float):
        return config
    elif isinstance(config, int):
        return float(config)
    elif isinstance(config, list):
        assert len(config) == 3
        start_from, close_to, at_epoch = config
        if start_from > close_to:
            return (start_from - close_to) * ramps.cosine_rampdown(current_epoch, at_epoch) + close_to
        elif start_from < close_to:
            return (close_to - start_from) * ramps.sigmoid_rampup(current_epoch, at_epoch) + start_from
        else:
            return start_from
    else:
        raise NotImplementedError('Unknown config type')

def load_test_data(image_path, size=256):
    img = misc.imread(image_path, mode='RGB')
    img = misc.imresize(img, [size, size])
    img = np.expand_dims(img, axis=0)
    img = preprocessing(img)

    return img

def preprocessing(x):
    x = x/127.5 - 1 # -1 ~ 1
    return x

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
    return (images+1.) / 2

def imsave(images, size, path):
    return misc.imsave(path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def str2bool(x):
    return x.lower() in ('true')

def cam(x, size = 256):
    x = x - np.min(x)
    cam_img = x / np.max(x)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, (size, size))
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    return cam_img / 255.0

def imagenet_norm(x):
    mean = [0.485, 0.456, 0.406]
    std = [0.299, 0.224, 0.225]
    mean = torch.FloatTensor(mean).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)
    std = torch.FloatTensor(std).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)
    return (x - mean) / std

def denorm(x):
    return x * 0.5 + 0.5

def tensor2numpy(x):
    return x.detach().cpu().numpy().transpose(1,2,0)

def RGB2BGR(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)