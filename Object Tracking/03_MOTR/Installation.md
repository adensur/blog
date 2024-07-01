1. Set up new virtual env
```bash
conda create -n motr python=3.10 pip
conda activate motr
```
2. Clone the repo
```bash
git clone git@github.com:megvii-research/MOTR.git
cd MOTR
```
3. Setup the repo
```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
# needed for dataset conversion scripts
pip install numpy

cd ./models/ops
sh ./make.sh
cd ../..
```
4. Download datasets  
- [MOT2017](https://motchallenge.net/data/MOT17/)
```bash
wget https://motchallenge.net/data/MOT17.zip
unzip MOT17.zip
```
- [CrowdHuman](https://www.crowdhuman.org/download.html)   
    Unfortunately, download links on google drive are behind auth so no command line solution is possible.   
    Recommended recipe to avoid downloading to a laptop and downloading directly to the remote vm:   
        - Open developer tools in browser    
        - Click on "download file"   
        - In the network section of developer tools click on "copy as curl".   
        - Paste curl into remove vm cmd line. It will download the file   
5. Convert datasets to proper format    
Instructions in MOTR Github [repo](https://github.com/megvii-research/MOTR?tab=readme-ov-file#dataset-preparation) are not entirely clear. It refers to FairMot [repo](https://github.com/ifzhang/FairMOT/tree/master) which have some conversion scripts.   
- MOT17
    After unpacking the .zip file, the directory structure will look something like this:
```
MOT17
├── test
│   ├── MOT17-01-DPM
│   ├── MOT17-01-FRCNN
│   ├── MOT17-01-SDP
│   ├── MOT17-03-DPM
│   ├── MOT17-03-FRCNN
│   ├── MOT17-03-SDP
│   ├── MOT17-06-DPM
│   ├── MOT17-06-FRCNN
│   ├── MOT17-06-SDP
│   ├── MOT17-07-DPM
│   ├── MOT17-07-FRCNN
│   ├── MOT17-07-SDP
│   ├── MOT17-08-DPM
│   ├── MOT17-08-FRCNN
│   ├── MOT17-08-SDP
│   ├── MOT17-12-DPM
│   ├── MOT17-12-FRCNN
│   ├── MOT17-12-SDP
│   ├── MOT17-14-DPM
│   ├── MOT17-14-FRCNN
│   └── MOT17-14-SDP
└── train
    ├── MOT17-02-DPM
    ├── MOT17-02-FRCNN
    ├── MOT17-02-SDP
    ├── MOT17-04-DPM
    ├── MOT17-04-FRCNN
    ├── MOT17-04-SDP
    ├── MOT17-05-DPM
    ├── MOT17-05-FRCNN
    ├── MOT17-05-SDP
    ├── MOT17-09-DPM
    ├── MOT17-09-FRCNN
    ├── MOT17-09-SDP
    ├── MOT17-10-DPM
    ├── MOT17-10-FRCNN
    ├── MOT17-10-SDP
    ├── MOT17-11-DPM
    ├── MOT17-11-FRCNN
    ├── MOT17-11-SDP
    ├── MOT17-13-DPM
    ├── MOT17-13-FRCNN
    └── MOT17-13-SDP
```
Conversion scripts in MOTR are not really compatible with other models, so I just copy all the images to MOTR repo in a slightly different format:
```bash
# in MOTR repo root
mkdir data
mkdir data/MOT17
mkdir data/MOT17/images
mkdir data/MOT17/images/train
mkdir data/MOT17/images/test
cp -r path_to_downloaded_mot17/train/* data/MOT17/images/train
cp -r path_to_downloaded_mot17/test/* data/MOT17/images/test/
```
After this, `data` directory structure should look like this:
```
data/MOT17
└── images
    ├── test
    │   ├── MOT17-01-DPM
    │   ├── MOT17-01-FRCNN
    │   ├── MOT17-01-SDP
    │   ├── MOT17-03-DPM
    │   ├── MOT17-03-FRCNN
    │   ├── MOT17-03-SDP
    │   ├── MOT17-06-DPM
    │   ├── MOT17-06-FRCNN
    │   ├── MOT17-06-SDP
    │   ├── MOT17-07-DPM
    │   ├── MOT17-07-FRCNN
    │   ├── MOT17-07-SDP
    │   ├── MOT17-08-DPM
    │   ├── MOT17-08-FRCNN
    │   ├── MOT17-08-SDP
    │   ├── MOT17-12-DPM
    │   ├── MOT17-12-FRCNN
    │   ├── MOT17-12-SDP
    │   ├── MOT17-14-DPM
    │   ├── MOT17-14-FRCNN
    │   └── MOT17-14-SDP
    └── train
        ├── MOT17-02-DPM
        ├── MOT17-02-FRCNN
        ├── MOT17-02-SDP
        ├── MOT17-04-DPM
        ├── MOT17-04-FRCNN
        ├── MOT17-04-SDP
        ├── MOT17-05-DPM
        ├── MOT17-05-FRCNN
        ├── MOT17-05-SDP
        ├── MOT17-09-DPM
        ├── MOT17-09-FRCNN
        ├── MOT17-09-SDP
        ├── MOT17-10-DPM
        ├── MOT17-10-FRCNN
        ├── MOT17-10-SDP
        ├── MOT17-11-DPM
        ├── MOT17-11-FRCNN
        ├── MOT17-11-SDP
        ├── MOT17-13-DPM
        ├── MOT17-13-FRCNN
        └── MOT17-13-SDP
```
Now, place the [gen_labels_17.py](gen_labels_17.py) script to the root of the repository and launch it:
```bash
python gen_labels_17.py
```
This is how `data` directory should look like now:   
```
data/MOT17
├── images
│   ├── test
│   │   ├── MOT17-01-DPM
│   │   ├── MOT17-01-FRCNN
│   │   ├── MOT17-01-SDP
│   │   ├── MOT17-03-DPM
│   │   ├── MOT17-03-FRCNN
│   │   ├── MOT17-03-SDP
│   │   ├── MOT17-06-DPM
│   │   ├── MOT17-06-FRCNN
│   │   ├── MOT17-06-SDP
│   │   ├── MOT17-07-DPM
│   │   ├── MOT17-07-FRCNN
│   │   ├── MOT17-07-SDP
│   │   ├── MOT17-08-DPM
│   │   ├── MOT17-08-FRCNN
│   │   ├── MOT17-08-SDP
│   │   ├── MOT17-12-DPM
│   │   ├── MOT17-12-FRCNN
│   │   ├── MOT17-12-SDP
│   │   ├── MOT17-14-DPM
│   │   ├── MOT17-14-FRCNN
│   │   └── MOT17-14-SDP
│   └── train
│       ├── MOT17-02-DPM
│       ├── MOT17-02-FRCNN
│       ├── MOT17-02-SDP
│       ├── MOT17-04-DPM
│       ├── MOT17-04-FRCNN
│       ├── MOT17-04-SDP
│       ├── MOT17-05-DPM
│       ├── MOT17-05-FRCNN
│       ├── MOT17-05-SDP
│       ├── MOT17-09-DPM
│       ├── MOT17-09-FRCNN
│       ├── MOT17-09-SDP
│       ├── MOT17-10-DPM
│       ├── MOT17-10-FRCNN
│       ├── MOT17-10-SDP
│       ├── MOT17-11-DPM
│       ├── MOT17-11-FRCNN
│       ├── MOT17-11-SDP
│       ├── MOT17-13-DPM
│       ├── MOT17-13-FRCNN
│       └── MOT17-13-SDP
└── labels_with_ids
    └── train
        ├── MOT17-02-DPM
        ├── MOT17-02-FRCNN
        ├── MOT17-02-SDP
        ├── MOT17-04-DPM
        ├── MOT17-04-FRCNN
        ├── MOT17-04-SDP
        ├── MOT17-05-DPM
        ├── MOT17-05-FRCNN
        ├── MOT17-05-SDP
        ├── MOT17-09-DPM
        ├── MOT17-09-FRCNN
        ├── MOT17-09-SDP
        ├── MOT17-10-DPM
        ├── MOT17-10-FRCNN
        ├── MOT17-10-SDP
        ├── MOT17-11-DPM
        ├── MOT17-11-FRCNN
        ├── MOT17-11-SDP
        ├── MOT17-13-DPM
        ├── MOT17-13-FRCNN
        └── MOT17-13-SDP
```
- CrowdHuman   
CrowdHuman comes in 6 files: `annotation_train.odgt`, `annotation_val.odgt`, `CrowdHuman_train01.zip`, `CrowdHuman_train02.zip`, `CrowdHuman_train03.zip`, `CrowdHuman_val.zip`   
Upon unzipping, images from both `CrowdHuman_train0*.zip` and `CrowdHuman_val.zip` will put images into Images folder, so it's best to be careful about that:   
```bash
# in the downloaded CrowdHuman directory
unzip CrowdHuman_train01.zip
unzip CrowdHuman_train02.zip
unzip CrowdHuman_train03.zip
mv Images train
unzip CrowdHuman_val.zip
mv Images val
```
Then, copy images into MOTR repository in desired format:   
```bash
# root of MOTR repo
mkdir data/crowdhuman
mkdir data/crowdhuman/images
cp -r path_to_downloaded_crowdhuman/train data/crowdhuman/images/train
cp -r path_to_downloaded_crowdhuman/val data/crowdhuman/images/val
cp -r path_to_downloaded_crowdhuman/annotation_train.odgt data/crowdhuman/
cp -r path_to_downloaded_crowdhuman/annotation_val.odgt data/crowdhuman/
```
This is how `data/crowdhuman` directory should look now:
```
data/crowdhuman
├── annotation_train.odgt
├── annotation_val.odgt
└── images
    ├── train
    │   ├── 273271,1017c000ac1360b7.jpg
        ...
    ├── val
        ├── 273271,104ec00067d5b782.jpg.jpg
        ...
```
Now, place the [gen_labels_crowd_id.py](gen_labels_crowd_id.py) script to the root of the repository and launch it:
```bash
python gen_labels_crowd_id.py
```
6. Download pretrained weights for the detection model  
MOTR authors used "iterative bounding box refinement" from [Deformable Detr](https://github.com/fundamentalvision/Deformable-DETR?tab=readme-ov-file#main-results)
7. Launch training script  
Script is located under `configs/r50_motr_train.sh`  
Remove comments from the file; change the pretrained weights path to the correct one: `coco_model_final.pth` -> `r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth` in my case.  
Add `--mot_path data` command to point it to the correct dataset path that we've created above.  
Modify `--nproc_per_node` according to the number of GPUs on your host (like 1)  
Here is how the final script looks for me:  
```bash
# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------


# for MOT17

PRETRAIN=r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth
EXP_DIR=exps/e2e_motr_r50_joint
python3 -m torch.distributed.launch --nproc_per_node=1 \
    --use_env main.py \
    --meta_arch motr \
    --use_checkpoint \
    --dataset_file e2e_joint \
    --epoch 200 \
    --with_box_refine \
    --lr_drop 100 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --pretrained ${PRETRAIN} \
    --output_dir ${EXP_DIR} \
    --batch_size 1 \
    --sample_mode 'random_interval' \
    --sample_interval 10 \
    --sampler_steps 50 90 150 \
    --sampler_lengths 2 3 4 5 \
    --update_query_pos \
    --merger_dropout 0 \
    --dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --query_interaction_layer 'QIM' \
    --extra_track_attn \
    --data_txt_path_train ./datasets/data_path/joint.train \
    --data_txt_path_val ./datasets/data_path/mot17.train \
    --mot_path data
```  
Now launch the script.   
```bash
bash configs/r50_motr_train.sh
```
8. Fixing `"ImportError: cannot import name '_NewEmptyTensorOp' from 'torchvision.ops.misc'"` error  
The error originates from incorrect version parsing in [utils/misc.py](https://github.com/megvii-research/MOTR/blob/main/util/misc.py#L32-L61) for newer torchvision versions. Simply remove lines 32-61 if your torchvision version is 0.8 or above.
9. Fixing `"rank0]: FileNotFoundError: [Errno 2] No such file or directory: 'data/crowdhuman/images/val/273278,41ba000090737c94.png'"`    
This error originates from the fact that MOTR code expects CrowdHuman dataset to have .png files, even though they are currently in .jpg (at least when downloaded from google drive). To fix, simply comment [these lines](https://github.com/megvii-research/MOTR/blob/main/datasets/joint.py#L101-L102) (`datasets/joint.py:101-102)
