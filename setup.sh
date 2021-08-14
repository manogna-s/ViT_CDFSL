mkdir checkpoints
mkdir checkpoints/pretrained_ckpts
mkdir checkpoints/output_ckpts
pip install numpy tqdm torch torchvision scipy pillow matplotlib tabulate ml-collections tensorboard tensorflow gin-config
pip install timm
cd dataset
python setup.py install
cd ..
