mkdir checkpoints
mkdir checkpoints/pretrained_ckpts
cd checkpoints/pretrained_ckpts/
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
cd ../..
mkdir checkpoints/output_ckpts
pip install numpy tqdm torch torchvision scipy pillow matplotlib tabulate ml-collections tensorboard tensorflow gin-config
pip install timm
cd cdfsl_dataset
python3 setup.py install
cd ..
