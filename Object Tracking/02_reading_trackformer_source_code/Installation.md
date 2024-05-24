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

6. Fix `RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu)`
By adding the following line:
```python
prev_out_ind = prev_out_ind.to(device)
```
Before line 94 of `detr_tracking.py` file.   

7. Launch the training 
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

8. In case of "cuda out of memory" errors
You can go to `cfgs/train.yaml` and modify some parameters to reduce memory footprint:
- batch_size
- enc_layers/dec_layers
- hidden_dim
None of this is recommended for a production tracking system, but might work for you if you just want to read the code for educational purposes.

9. Install the repo globally on the system
```bash
pip install -e .
```
This will make it importable

10. Install ipykernel package for debugging
```bash
pip install -U ipykernel
```

11. Create a jupyter notebook at the root of the repository
Add the following convenience stubs:
```python
import torch, torchvision
import matplotlib.pyplot as plt
# COCO classes
CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

coco_idx_to_label = {idx: label for idx, label in enumerate(CLASSES)}

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        output_tensor = []
        for t, m, s in zip(tensor, self.mean, self.std):
            output_tensor.append(t.mul(s).add(m))
            # t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return torch.stack(output_tensor, dim=0)

unnorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(img, labels, boxes, mask=None):
    h, w = img.shape[1:]
    if mask != None:
        # width
        if torch.where(mask[0])[0].shape[0] > 0:
            mask_w = torch.where(mask[0])[0][0]
            w = min(w, mask_w)
        if torch.where(mask[:, 0])[0].shape[0]:
            mask_h = torch.where(mask[:, 0])[0][0]
            h = min(h, mask_h)
            
    boxes = rescale_bboxes(boxes, (w, h))
    plt.figure(figsize=(16,10))
    unimage = unnorm(img)
    #image = (unimage*256).to(torch.uint8)
    image = unimage
    pil_img = torchvision.transforms.functional.to_pil_image(image)
    plt.imshow(pil_img)
    
    ax = plt.gca()
    colors = COLORS * 100
    for label, (xmin, ymin, xmax, ymax), c in zip(labels, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{CLASSES[label]}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

%matplotlib inline
```
They allow actually plotting input image tensors and bounding boxes.   

Having a jupyter notebook in the root of the repository allows copy-pasting code from main scripts (train.py), executing and debugging interactively.