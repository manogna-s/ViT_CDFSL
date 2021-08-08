import json
import os
import numpy as np
import timm
import torch
from models.model_vit import VisionTransformer, CONFIGS
from paths import *


def get_base_model(img_size=224, model_type='ViT-B_16', pretrained_ckpt='checkpoints/pretrained_ckpts/ViT-B_16.npz', from_timm=False):
    if from_timm:
        print(f'Loading base timm model')
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0, img_size=img_size)
    else:
        config = CONFIGS[model_type]
        model = VisionTransformer(config, img_size, zero_head=True, num_classes=1000)
        model.load_from(np.load(pretrained_ckpt))
    return model


def get_model(img_size=224, dataset='dtd', from_timm=False):
    if from_timm:
        model_dir = os.path.join(CHECKPOINT_DIR, f'timm_{dataset}_img{img_size}')
        config_file = os.path.join(model_dir, 'model_config.json')
        with open(config_file) as cfg:
            config = json.load(cfg)
        print('Getting model:')
        print(config)
        print(f'Loading timm model trained on {dataset} on image size {img_size}')
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=10, img_size=img_size)
    else:
        model_dir = os.path.join(CHECKPOINT_DIR, f'{dataset}_img{img_size}')
        config_file = os.path.join(model_dir, 'model_config.json')
        with open(config_file) as cfg:
            config = json.load(cfg)
        print('Getting model:')
        print(config)
        model = VisionTransformer(CONFIGS[config['model_type']], num_classes=config['num_classes'],
                                  img_size=config['img_size'], zero_head=True)
    checkpoint_file = os.path.join(model_dir, 'checkpoint.bin')
    model.load_state_dict(torch.load(checkpoint_file, map_location=torch.device('cpu')))
    return model
