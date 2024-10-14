Check out [repo](https://github.com/facebookresearch/segment-anything)
```bash
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything
# install torch deps with conda
conda create -n sam -y python=3.11 pip
conda activate sam
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

# install SAM repo and some libraries
pip install -e .
pip install numpy matplotlib opencv-python torchvision

# download model checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```