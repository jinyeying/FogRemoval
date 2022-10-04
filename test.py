import argparse
import os
from os import makedirs, listdir
from os.path import join, isfile, basename, exists
from math import ceil
from PIL import Image
import PIL
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from networks import GenerativeModel
from utils import get_config
from modules import GaussianBlur
import random
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def increase_saturation(img):
    converter = PIL.ImageEnhance.Color(img)
    img2 = converter.enhance(1)
    return img2

parser = argparse.ArgumentParser(description='Dehazing using GAN')
parser.add_argument('--eval_dir', type=str, default='./baseline_sup_adv_hz0.01_ft_nomask_Kres_ref_gray_moredata', help='evaluate from this')
epoch = 1000

args = parser.parse_args()

config = get_config(args.eval_dir + '/config.yml')
datasets = get_config('./configs/datasets.yml')

result_dir = './results/'
if not exists(result_dir):
    makedirs(result_dir, exist_ok=True)

step_size = 4
args.resize_to = 512                    
save_dir = 'proposed_' + os.path.basename(args.eval_dir) + '_' + str(args.resize_to) 

print(save_dir)

testset_names = [
    'test_haze',
    #'Smokemachine',
]

"""
Models
"""
print("===> Creating models...")

netGen = GenerativeModel(config['gen'])
if epoch: 
    ckpt_file = join(args.eval_dir, './models/checkpoint.'+str(epoch)+'.ckpt')
    save_dir = save_dir + '_epoch' + str(epoch) + '_dir3'
else:
# optionally resume from a checkpoint
    ckpt_file = join(args.eval_dir, './models/checkpoint.current.ckpt')
assert isfile(ckpt_file), "=> no checkpoint found at '{}'".format(ckpt_file)
checkpoint = torch.load(ckpt_file)
netGen.load_state_dict(checkpoint['state_dict_netGen'])

"""
Testing
"""
print('===> Start testing...')

netGen.train()
gBlur = GaussianBlur(sigma=step_size/4).cuda()

def is_image_file(filename):
    return any(filename.lower().endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.bmp'])


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]
)


for dname in testset_names:

    input_dir = datasets[dname]['root_dirs'][0]
    out_dir = join(result_dir, dname, save_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    in_filenames = sorted([join(input_dir, x) for x in listdir(input_dir) if is_image_file(x)])
    pbar = tqdm(total=len(in_filenames), desc=dname)

    for in_filename in in_filenames:
        assert isfile(in_filename)
        out_filename = join(out_dir, basename(in_filename)[:-3]+'png')

        img = Image.open(in_filename).convert('RGB')

        w, h = img.size
        shortest = min([w, h])
        new_w = int(ceil(float(w) / shortest * args.resize_to))
        new_h = int(ceil(float(h) / shortest * args.resize_to))

        new_w = round(new_w/step_size) * step_size
        new_h = round(new_h/step_size) * step_size

        img = img.resize([new_w, new_h], Image.LANCZOS)   

        with torch.no_grad():
            imgIn = transform(img).unsqueeze_(0)
            imgIn = imgIn.cuda()

            divider = torch.zeros_like(imgIn).cuda()
            prediction = torch.zeros_like(imgIn).cuda()

            pred , _ = netGen(imgIn)
            prediction = (pred + 1.)/2.

        prediction = prediction.data[0, :, :, :]
        prediction = prediction.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()

        result = Image.fromarray(prediction)
        if 0 < args.resize_to < shortest:
            result = result.resize([new_w, new_h], Image.LANCZOS)
        else:
            result = result.resize([w, h], Image.LANCZOS)
        result.save(out_filename)
        pbar.update()

    pbar.close()


