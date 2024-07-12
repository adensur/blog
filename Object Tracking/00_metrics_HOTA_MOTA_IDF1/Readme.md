## Installation
The official code for metrics calculation for [MOT Challenge](https://motchallenge.net/data/MOT17/) is located [here](https://github.com/JonathonLuiten/TrackEval/tree/master). [This](https://github.com/JonathonLuiten/TrackEval/tree/master/docs/MOTChallenge-Official) page contains some instructions.

### Installing TrackEval dependencies
- Clone the repo and navigate to the root folder of that repo.  
- Create a fresh conda environment  
```bash
conda create -n hota python=3.10 pip
conda activate hota
```
- Install dependencies  
    `requirements.txt` will contain exact package versions that were used by authors of the repo. You can either try to install those, which probably will require you to build some wheels from source, and it might require extra dependencies to work. I went on an alternative root - updating all the packages to the freshest versions, and then modifying the code slightly to make it work:   
```
# my requirements.txt
numpy
scipy
pycocotools
matplotlib
opencv_python
scikit_image
pytest
Pillow
tqdm
tabulate
```
```bash
pip install -r requirements.txt
```
- Download sample data
```bash
wget https://omnomnom.vision.rwth-aachen.de/data/TrackEval/data.zip
unzip data.zip
```
It will contain ground truth data for various challenges, including MOT17 - just the text files, no images; and sample detections for a detector called MPNTrack
- Modify the code
Since we use updated version of numpy, we will have to replace all entries with `np.float` to `float`, and `np.int` to `int`!
- Run the eval script
```bash
python scripts/run_mot_challenge.py --BENCHMARK MOT17 --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL MPNTrack --METRICS HOTA CLEAR Identity VACE --USE_PARALLEL False --NUM_PARALLEL_CORES 1
```
If everything was done correctly, this command should work and output metric values for MPNTrack tracker.