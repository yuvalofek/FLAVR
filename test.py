import os
import sys
import time
import copy
import shutil
import random
import pdb

import torch
import numpy as np
from tqdm import tqdm

import config
import myutils

from torch.utils.data import DataLoader

##### Parse CmdLine Arguments #####
#os.environ["CUDA_VISIBLE_DEVICES"]='7'
args, unparsed = config.get_args()
cwd = os.getcwd()

print(f'Using cuda: {args.cuda}')
device = torch.device('cuda' if args.cuda else 'cpu')

torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)

if args.dataset == "vimeo90K_septuplet":
    from dataset.vimeo90k_septuplet import get_loader
    test_loader = get_loader('test', args.data_root, args.test_batch_size, shuffle=False, num_workers=args.num_workers)
elif args.dataset == "ucf101":
    from dataset.ucf101_test import get_loader
    test_loader = get_loader(args.data_root, args.test_batch_size, shuffle=False, num_workers=args.num_workers)
elif args.dataset == "gopro":
    from dataset.GoPro import get_loader
    test_loader = get_loader(args.data_root, args.test_batch_size, shuffle=False, num_workers=args.num_workers, test_mode=True, interFrames=args.n_outputs)    
elif args.dataset == "satellite":
    from dataset.Satellite import get_loader, get_train_test
    set_length = (args.nbr_frame-1)*(args.n_outputs+1)+1
    paths, tr_idx, test_idx = get_train_test(args.data_root, set_length, random_state=214)
    train_loader = get_loader(paths, tr_idx, args.batch_size, shuffle=True, num_workers=args.num_workers, is_training=True, inter_frames=args.n_outputs, n_inputs=args.nbr_frame, channels=args.channels)
    test_loader = get_loader(paths, test_idx, args.batch_size, shuffle=False, num_workers=args.num_workers, is_training=False, inter_frames=args.n_outputs, n_inputs=args.nbr_frame, channels=args.channels)
else:
    raise NotImplementedError


from model.FLAVR_arch import UNet_3D_3D
print("Building model: %s"%args.model.lower())
model = UNet_3D_3D(args.model.lower() , n_inputs=args.nbr_frame, n_outputs=args.n_outputs, joinType=args.joinType, upmode=args.upmode, channels=args.channels)

model = torch.nn.DataParallel(model).to(device)
print("#params" , sum([p.numel() for p in model.parameters()]))


def test(args):
    time_taken = []
    losses, psnrs, ssims = myutils.init_meters(args.loss)
    model.eval()

    psnr_list = []
    with torch.no_grad():
        for i, (images, gt_image ) in enumerate(tqdm(test_loader)):

            images = [img_.cuda() for img_ in images]
            gt = [g_.cuda() for g_ in gt_image]

            torch.cuda.synchronize()
            start_time = time.time()
            out = model(images)

            out = torch.cat(out)
            gt = torch.cat(gt)

            torch.cuda.synchronize()
            time_taken.append(time.time() - start_time)

            myutils.eval_metrics(out, gt, psnrs, ssims)

    print("PSNR: %f, SSIM: %fn" %
          (psnrs.avg, ssims.avg))
    print("Time , " , sum(time_taken)/len(time_taken))

    return psnrs.avg


""" Entry Point """
def main(args):
    
    assert args.load_from is not None

    model_dict = model.state_dict()
    model.load_state_dict(torch.load(args.load_from)["state_dict"] , strict=True)
    test(args)


if __name__ == "__main__":
    main(args)
