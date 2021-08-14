import json
import os
import numpy as np
import timm
import torch
from paths import *
import src

def get_domain_extractor(img_size=224, model_type='So_vit_7', domain='imagenet'):

    print(f'Loading {domain} feature extractor')
    checkpoint_path = os.path.join(CHECKPOINT_DIR, domain, 'model_best.pth.tar')
    model = timm.create_model('So_vit_7', checkpoint_path=checkpoint_path, num_classes=0, img_size=img_size)
    return model

def get_base_model(img_size=224, model_type='ViT-B_16', pretrained_ckpt='checkpoints/pretrained_ckpts/imagenet/model_best.pth.tar', from_timm=False):

    print(f'Loading base timm model')
    model = timm.create_model('So_vit_7', checkpoint_path=pretrained_ckpt, num_classes=0, img_size=img_size)
    return model


def extract_features(extractors, images):
  all_features = []
  with torch.no_grad():
    for name, extractor in extractors.items():
      features = extractor.forward_features(images)
      all_features.append(features)
  return torch.stack(all_features, dim=1) # batch x #extractors x #features