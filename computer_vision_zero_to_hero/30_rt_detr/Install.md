To install, clone the repo: https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch   
Navigate to RT-DETR/rtdetr_pytorch  
Install the dependencies using conda:  
```bash
conda create -n rtdetr -y python=3.10 pip
conda activate rtdetr
# install pytorch of the required version using conda
# makes sure correct version for your cuda version is installed
conda install pytorch==2.0.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
pip install transformers
```
Download the dataset:
```bash
mkdir dataset
mkdir dataset/coco
cd dataset/coco
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
for u in `ls *zip`; do unzip $u; done
cd ../..
```
Now training command should work:  
```bash
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml
```