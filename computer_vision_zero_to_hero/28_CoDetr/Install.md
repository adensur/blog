This guide shows how to install and run Co-DETR on a system with nvidia gpu and cuda 11.* drivers.

Co-DETR uses copy-pasted, old version of `mmdet` library, which in turn depends on old version of pytorch, which is only available on older versions of cuda (11.* or lower), unless you are willing to build all of that from source. I've spent quite some time trying to make that all work with different cuda and library versions, and this is the only way I could make it work. This is unfortunate because you cannot, for example, create a dedicated cuda 11.* virtual environment - these drivers will have to be installed on the entire host, potentially messing with your other projects.  

I also couldn't install cpu-only version of Co-DETR, though that should be possible in theory - they even have "deformable attention" cpu implementation.  
- Clone the repo  
```bash
git clone git@github.com:Sense-X/Co-DETR.git
cd Co-DETR
```
- Set up virtual environment and setup some packages
To proceed, make sure that cuda 11.* is installed (11.4 in my case).  
```bash
# check that cuda version is correct
nvidia-smi
# should show 11.*

# create venv with required package versions
conda create -n codetr -y python=3.10 pip
conda activate codetr
# official codetr instructions follow pytorch=1.11.0, cuda=11.3
conda install -y pytorch=1.11 torchvision pytorch-cuda=11 -c pytorch -c nvidia
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```
The last line checks that the versions are correct, in my case they print:   
```
1.11.0
11.3
```
- Install some dependencies from the file I provide: [my_requirements.txt](./my_requirements.txt). Simply copy the contents to `my_requirements.txt` file in the root of the repo, and then:   
```bash
pip install -r my_requirements.txt
```
- Install mmcv - core dependency of Co-DETR
```bash
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
pip install -e .
python -c "import mmcv; print(mmcv.__version__)"
```
- Download the data - coco and lvis dataset
```bash
mkdir data
cd data
mkdir coco
cd coco
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
for u in `ls *zip`; do unzip $u; done
rm -rf *zip
cd ..
mkdir lvis_v1
cd lvis_v1
# lvis_v1 uses the same images as coco, so symlinks should suffice
ln -s ../coco/train2017/ .
ln -s ../coco/val2017/ 
wget https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip
wget https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip
for u in `ls *zip`; do unzip $u; done
rm -rf *zip
mkdir annotations
mv lvis_v1_* annotations/
cd ../..
```
- Launch the training
[Readme](https://github.com/Sense-X/Co-DETR/blob/main/README.md) has some instructions on how to launch scripts for distributed training. I prefer to launch using bare python, which allows you to reproduce the same setup in vscode debugger, for example:   
```bash
# set up some env var to make code work
export LOCAL_RANK=0
export RANK=0
export GROUP_RANK=0
export ROLE_RANK=0
export ROLE_NAME=default
export LOCAL_WORLD_SIZE=1
export WORLD_SIZE=1
export GROUP_WORLD_SIZE=1
export ROLE_WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
python tools/train.py projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py --launcher pytorch --work-dir my_exp1
```
- To launch the same code in vscode debugger
Add this code to `launch.json` in vscode:
```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
    

        {
            "name": "train",
            "type": "debugpy",
            "request": "launch",
            "program": "tools/train.py",
            "console": "integratedTerminal",
            "justMyCode": false,
            "args": "projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py --launcher pytorch --work-dir my_exp1",
            "env": {
                "LOCAL_RANK": "0",
                "RANK": "0",
                "GROUP_RANK": "0",
                "ROLE_RANK": "0",
                "ROLE_NAME": "default",
                "LOCAL_WORLD_SIZE": "1",
                "WORLD_SIZE": "1",
                "GROUP_WORLD_SIZE": "1",
                "ROLE_WORLD_SIZE": "1",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "29500",
            }
        }
    ]
}
```
This sets the same arguments and env variables that we used for launch above.  
Then, choose the same interpreter as in conda environment: cmd-shift-p, select interpreter, and choose `codetr` virtual env. This makes sure that environment and packages will be exactly the same when launching debugger.   

It is also helpful to add this line in the beginning of the training script in train.py:
```python
print("Arguments: ", args)
```
Which will come in handy later.

After this, I prefer to "disassemble" the train and model code by copy-pasting stuff into my own `sandbox.ipynb` located in the root of the repository, which allows me to plot the actual images, execute code interactively and so on. [Here](./sandbox.ipynb) is the final version of such notebook.