```bash
# clone the repo
git clone git@github.com:facebookresearch/detr.git
cd detr
# setup fresh conda env
conda create -n detr -y python=3.11 pip
conda activate detr
# install dependencies
conda install -c pytorch pytorch torchvision
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install panopticapi
# get coco
mkdir dataset
mkdir dataset/coco
cd dataset/coco
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
for u in `ls *zip`; do unzip $u; done
cd ../..
# get coco paniptic annotations
cd dataset
mkdir coco_panoptic
cd coco_panoptic
wget http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip
unzip panoptic_annotations_trainval2017.zip
unzip annotations/panoptic_train2017.zip
unzip annotations/panoptic_val2017.zip

# train obj detector
python main.py --coco_path dataset/coco
# train stuff obj detector
python main.py --coco_path dataset/coco  --coco_panoptic_path dataset/coco_panoptic --dataset_file coco_panoptic --output_dir ./output/path/box_model
# train stuff obj detector multi gpu
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path dataset/coco  --coco_panoptic_path dataset/coco_panoptic --dataset_file coco_panoptic --output_dir ./output/box_model2

# --dataset_file coco_panoptic means that panoptic annotations will be used for training


# train segmentation model
python main.py --masks --epochs 25 --lr_drop 15 --coco_path dataset/coco  --coco_panoptic_path dataset/coco_panoptic  --dataset_file coco_panoptic --frozen_weights ./output/box_model2/checkpoint.pth --output_dir ./output/path/segm_model
```