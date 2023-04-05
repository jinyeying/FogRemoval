from SRDefog_test import SRDefog
import argparse
from utils import *

def parse_args():
    desc = "Implementation of SRDefog"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='test', help='[train / test]')
    parser.add_argument('--dataset', type=str, default='NH-HAZE', help='dataset_name')
    parser.add_argument('--datasetpath', type=str, default='/diskc/yeying/dayfog/ACCV_code_release2/NH-HAZE/', help='dataset_path')
    parser.add_argument('--iteration', type=int, default=1000000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image print freq')
    parser.add_argument('--save_freq', type=int, default=50000, help='The number of model save freq')
    #parser.add_argument('--decay_flag', type=str2bool, default=True, help='The decay_flag')
    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='The weight decay')
    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_res', type=int, default=4, help='The number of resblock')
    parser.add_argument('--n_dis', type=int, default=6, help='The number of discriminator layer')

    parser.add_argument('--img_size', type=int, default=512, help='The size of image')
    parser.add_argument('--img_h', type=int, default=512, help='The org size of image')
    parser.add_argument('--img_w', type=int, default=512, help='The org size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
    parser.add_argument('--result_dir', type=str, default='./results', help='Directory name to save the results')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Set gpu mode; [cpu, cuda]')
    parser.add_argument('--im_suf_A', type=str, default='.png', help='The suffix of test images [.png / .jpg / .JPG]')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --result_dir
    check_folder(os.path.join(args.result_dir, args.dataset, 'model'))

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    gan = SRDefog(args)

    # build graph
    gan.build_model()

    if args.phase == 'test' :
        gan.test()
        print(" [*] Test finished!")

if __name__ == '__main__':
    main()