import os
import numpy as np
import time
import sys
import argparse
import errno
from collections import OrderedDict
import tensorboardX
from tqdm import tqdm
import random
import imageio
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from lib.utils.tools import *
from lib.utils.learning import *
from lib.model.loss import *
from lib.data.dataset_action import NTURGBD,WildActionDataset,get_action_names
from lib.model.model_action import ActionNet
from lib.data.dataset_wild import WildDetDataset


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-j', '--json_path', type=str, help='alphapose detection result json path')
    parser.add_argument('-p', '--pretrained', default='checkpoint', type=str, metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-freq', '--print_freq', default=100)
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('--clip_len', type=int, default=243, help='clip length for network input')
    parser.add_argument('-v', '--vid_path', type=str, help='video path')
    parser.add_argument('-kpn', '--keypoints_normalized',action="store_true", help='chose among using keypoints or normalized keypoints')



    opts = parser.parse_args()
    return opts

def infer_with_config(args, opts):

    OKGREEN = '\033[92m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

    # Load wild dataset

    print('Loading sample')

    vid = imageio.get_reader(opts.vid_path, 'ffmpeg')
    fps_in = vid.get_meta_data()['fps']
    vid_size = vid.get_meta_data()['size'] 
    vid_shape=(vid_size[1],vid_size[0])
    vid.close()
    wild_dataset = WildActionDataset(opts.json_path, n_frames=opts.clip_len, vid_shape=vid_shape ,scale_range=args.scale_range_test,keypoints_normalized=opts.keypoints_normalized)
    
    # Load model

    print('Loading model')
    
    model_backbone = load_backbone(args)
    model = ActionNet(backbone=model_backbone, dim_rep=args.dim_rep, num_classes=args.action_classes, dropout_ratio=args.dropout_ratio, version=args.model_version, hidden_dim=args.hidden_dim, num_joints=args.num_joints)
    criterion = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()

    # Load checkpoint 
    
    chk_filename = opts.evaluate 
    print('Loading checkpoint', chk_filename)
        
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    if not torch.cuda.is_available():
      for k, v in checkpoint['model'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    else:
      new_state_dict=checkpoint['model']

    model.load_state_dict(new_state_dict, strict=True)

    # Inference

    print('Starting infrerence')

    sample=wild_dataset[0]
    sample=sample[np.newaxis, :]
    sample=torch.tensor(sample)
    
    model.eval()
    with torch.no_grad():        
        if torch.cuda.is_available():
            sample = sample.cuda()
        logits = model(sample)    # (N, num_classes)
        probabilities = F.softmax(logits, dim=1)
        names=get_action_names()
        print(f'\n{BOLD}Predicted class:{ENDC} {OKGREEN}{names[np.argmax(probabilities)]}{ENDC}')
        

   
if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    infer_with_config(args, opts)