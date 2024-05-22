1. Clone the repository: https://github.com/timmeinhardt/trackformer/tree/main
2. Create virtual env using miniconda. Install some standard dependencies:
```bash
conda create -n trackformer python=3.10 pip
conda activate trackformer
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```
3. Install requirements
```bash
pip install -r requirements.txt
```
This file contains specific library versions that worked back when the model was being actively developed. Although in theory it should be possible to still install everything and run the code with the same library versions; in practice, it might not work out of the box for specific os/processor architecture you have at the moment.   

In my case, some old versions wouldn't install, but simply removing specific version requirement worked. I'm putting the up-to-date version of requirements.txt [here](./requirements.txt)    

4. Install modules from the repo itself
```bash
python src/trackformer/models/ops/setup.py build --build-base=src/trackformer/models/ops/ install
```
Mostly needed to build custom "Deformable Attention" cuda layer. The code for deformable attention is only written for cuda gpu; it will not work on cpu or Apple's GPUs aka "Metal Performane Shaders" (mps).   

5. Go over instructions from the repository [here](https://github.com/timmeinhardt/trackformer/blob/main/docs/INSTALL.md) starting from #3 - downloading datasets and pretrained model weights.   

6. Install pycocotools
```bash
pip install pycocotools
```
It used to be necessary to do some custom install steps. Right now it is freely available via pip, you can put it in requirements.txt as well.    

7. Fix `RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu)`
By adding the following line:
```python
prev_out_ind = prev_out_ind.to(device)
```
Before line 94 of `detr_tracking.py` file.   

8. Launch the training 
```bash
# crowdhuman
python src/train.py with \
    crowdhuman \
    deformable \
    multi_frame \
    tracking \
    output_dir=models/crowdhuman_deformable_multi_frame

# mot
python src/train.py with \
    mot17 \
    deformable \
    multi_frame \
    tracking \
    output_dir=models/mot17_crowdhuman_deformable_multi_frame
```

Crowdhuman pretraining uses object detection data (aka just immages annotated with detected objects and bounding boxes) to "bootstrap" into object tracking by using image augmentation.   

MOT dataset contains actual video sequences with annotated 

9. In case of "cuda out of memory" errors
You can go to `cfgs/train.yaml` and modify some parameters to reduce memory footprint:
- batch_size
- enc_layers/dec_layers
- hidden_dim
None of this is recommended for a production tracking system, but might work for you if you just want to read the code for educational purposes.