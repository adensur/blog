The last few years were exceptionally bountiful for Computer Vision. From simple models that do basic image classification, to desribing a scene in real time in human language; generating exceptionally detailed and beautiful images based on text prompt; "morphing" your photo into anime character; editing an image, replacing one object with another; translating a video of someone talking into another language with original actor's voice and modifying lip movements in such a way as to make it look like he was the one talking in this language; creating a "deepfake" video of a politician saying things they never did. It's like a pandora box that was opened at some point, and now gives more and more surprises!  

And it doesn't stop at deepfakes and fun. Computer Vision is used in lots of industries - self driving cars, industrial damage sniffing drones, cancer recognition from MRI images and so on.  

Just think about how complicated these tasks are. In the example of editing, we first perform *segmentation* - identify pixels of the object to be replaced using *text prompt* - which can be something like "third person from the left"; generate another object using stable diffusion, and make sure it "blends in" with the rest of the image - our eye is quite good at spotting inconsistencies with colours and shadows, so this is no trivial task; generating the "continuation" of the background.  
There are plenty of free mobile apps for that, plenty of online tutorials that teach you how to download a model and do some generating yourself. However, if you try to dig deeper and actually understand how all of that works under the hood, it becomes quite complicated really fast. Trying to read original paper (like [this one](https://arxiv.org/pdf/2303.05499.pdf)) instantly bombards you with words like *contrastive loss*, *grounding*, *multi-modality feature comparison* and so on. Trying to dig into just the vocabulary leads you to even more papers, with even more unknown words...  

The purpose of this blog series is to bridge that gap. To get started with something really simple from the "introduction to machine learning" tutorials, and to work our way all the way up to the most exciting concepts in modern computer vision; to be able to build, train and understand the fundamentals for models similar to current state-of-the-art (SOTA); be able to read the papers and understand everything that is going on. I don't assume that you know anything about computer vision so far, but I do assume knowing some machine learning fundamentals - backprop, gradient descent, dense layers and so on.  

Here is roughly what we will go through:
- Pattern recognition with dense layers
- Convolutions
- Transformers
- Object detection and segmentation
- Moving from closed-set object detection to open-set by fusing text and image representations
- GANs
- Stable Diffusion  

We will be writing code, training our own models, reading the actual papers and explaining the concepts from them. Stay tuned!