Clone the repository and cd into rtdetr v2, pytorch version, working directory:  
```bash
git clone git@github.com:lyuwenyu/RT-DETR.git
cd RT-DETR/rtdetrv2_pytorch/
```
Create environment using conda, install some deps:   
```bash
conda create -n rtdetrv2 -y python=3.10 pip
conda activate rtdetrv2
conda install pytorch==2.4 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
pip install scipy
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
Now, model training should work like this:  
```bash
python tools/train.py -c configs/rtdetrv2/rtdetrv2_r18vd_sp1_120e_coco.yml --use-amp --seed=0
```