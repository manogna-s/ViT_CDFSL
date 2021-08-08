# Vision Transformers for Cross-domain Few-shot Learning

This code was adapted from the following repositories:
1. [Meta-Dataset](https://github.com/google-research/meta-dataset)
2. [Vision Transformers](https://github.com/jeonsworld/ViT-pytorch)
3. [Selecting Universal Representations](https://github.com/dvornikita/SUR)

### Install requirements 
```commandline
setup.sh
```

### Get pretrained ViT-B16 model
```commandline
cd checkpoints/pretrained_ckpts/
!wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
```

### Prepare meta-dataset following these [instructions](https://github.com/google-research/meta-dataset#user-instructions)

### Fine tune ViT on new dataset:
-Extract datasets to ./data folder. Set model configs and fine-tune. 

-Currently supports fine tuning on CIFAR-10, CIFAR-100, Omniglot, Aircraft, Textures dataset. 

-To fine-tune on other datasets add Dataloaders in utils/data_utils.py
```commandline
python train.py --model_config=vit_configs/dtd_224.gin 
```

### Evaluate single feature extractor on tasks from Meta-Dataset:
```commandline
python eval_vit.py --testsets dtd  --trainset 'dtd' --img_size 224
```

### Evaluate ViT with Selecting Universal Representations on Meta-Dataset:
```commandline
python eval_vit_sur.py --testsets cu_birds  --trainset 'imagenet dtd' --img_size 224 --num_tasks 100
```

### Visualize attention maps on image
```commandline
python visualize_attention.py --test_img data/test_imgs/dog.jpg
```