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
import shap

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def shap_with_config(args, opts):

    num_samples=100
    class_names=get_action_names()


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Loading sample')
    sample=load_sample(opts,args)
    print('Loading background')
    background = load_background(num_samples)
    print('Loading model')
    model=load_model(opts,args)
    
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
    model.eval()
    model.to(device)

    def predict(sample):    
        logits = model(sample)    # (N, num_classes)
        return logits
           
    

    sample = torch.tensor(sample).to(device)
    background = torch.tensor(background).to(device)

    topk = 4
    batch_size = 50
    n_evals = 1000

    # define a masker that is used to mask out partitions of the input image.
    masker_blur = shap.maskers.Image("blur(128,8)", background[0].shape)

    # create an explainer with model and image masker
    explainer = shap.Explainer(predict, masker_blur, output_names=class_names)

    # feed only one image
    # here we explain two images using 100 evaluations of the underlying model to estimate the SHAP values
    shap_values = explainer(
        sample,
        max_evals=n_evals,
        batch_size=batch_size,
        outputs=shap.Explanation.argsort.flip[:topk],
    )
    print(shap_values.data.shape, shap_values.values.shape)


class FixedInputAdapterLayer(nn.Module):
    """Layer di adattamento con M=2 fisso."""
    def forward(self, x):
        x=torch.tensor(x)
        B, F, V, FE = x.shape
        zeros = torch.zeros(B, 1, F, V, FE, device=x.device, dtype=x.dtype)
        output = torch.cat([x.unsqueeze(1), zeros], dim=1)
        return output

class SingleMotionActionModel(nn.Module):
    def __init__(self, original_model):
        super(SingleMotionActionModel, self).__init__()
        self.adapter_layer = FixedInputAdapterLayer()
        self.original_model = original_model

    def forward(self, x):
        x = self.adapter_layer(x)  # Adatta l'input
        return self.original_model(x)  # Applica il modello originale


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

def load_sample(opts,args):
  # Load wild dataset

    vid = imageio.get_reader(opts.vid_path, 'ffmpeg')
    fps_in = vid.get_meta_data()['fps']
    vid_size = vid.get_meta_data()['size'] 
    vid_shape=(vid_size[1],vid_size[0])
    vid.close()
    wild_dataset = WildActionDataset(opts.json_path, n_frames=opts.clip_len, vid_shape=vid_shape ,scale_range=args.scale_range_test,keypoints_normalized=opts.keypoints_normalized)
    sample = wild_dataset[0][0]
    return sample[np.newaxis, :]

def load_model(opts,args):
      # Load model    
    model_backbone = load_backbone(args)
    model = ActionNet(backbone=model_backbone, dim_rep=args.dim_rep, num_classes=args.action_classes, dropout_ratio=args.dropout_ratio, version=args.model_version, hidden_dim=args.hidden_dim, num_joints=args.num_joints)

    # Load checkpoint 
    
    chk_filename = opts.evaluate 
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    #else:
    #new_state_dict=checkpoint['model']

    model.load_state_dict(new_state_dict, strict=True)
    return SingleMotionActionModel(model)

def load_background(num_samples):
  with open('data/action/processed/ntu60_xsub_train_processed_150.pkl', 'rb') as file:
    Xt, _ = pickle.load(file)
  X= Xt[:7350, 0, :, :, :]

  return X[np.random.choice(X.shape[0], num_samples, replace=False)]
    
   
if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    shap_with_config(args, opts)