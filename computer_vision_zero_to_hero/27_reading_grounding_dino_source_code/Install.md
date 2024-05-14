To install, clone the repo with the source code:   
```bash
git clone git@github.com:IDEA-Research/GroundingDINO.git
```
Then, copy `sandbox.ipynb` file to the root of the repository.  

It is recommended to create a fresh virtual environment before installing so that different projects do not mess one another. Using Miniconda:   

```bash
conda create -n grounding_dino python=3.10 pip
conda activate grounding_dino
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -e .
pip install opencv-python
pip install -r requirements.txt
```

Getting model weights:   

```bash
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```

Getting a sample image from COCO (or choose your own):   
```bash
wget "https://farm9.staticflickr.com/8437/7910770924_e51c726cc7_z.jpg" -O coco.jpg
```