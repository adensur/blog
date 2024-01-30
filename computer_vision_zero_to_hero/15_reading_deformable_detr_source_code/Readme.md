This post is also available as [video](https://youtu.be/3M9mS_3eiaw).  

This post is a second post about [Deformable DETR](https://arxiv.org/pdf/2010.04159.pdf). Previous [post](https://github.com/adensur/blog/blob/main/computer_vision_zero_to_hero/14_deformable_detr/Readme.md) explained the algorithm itself. This post will talk more about the [source code](https://github.com/fundamentalvision/Deformable-DETR): how to run it on your machine and how to read it.   

The source code for Deformable DETR is really close to that of the original DETR. I have a detailed [post](https://github.com/adensur/blog/blob/main/computer_vision_zero_to_hero/13_reading_detr_source_code/Readme.md) about that. I will skip lot of stuff that is similar between original DETR and Deformable DETR, including debugger setup, data loading, loss function and train loop.   

[Here](https://github.com/adensur/blog/blob/main/computer_vision_zero_to_hero/15_reading_deformable_detr_source_code/sandbox.ipynb) is the jupyter notebook used in this post.    

This is the plan for today:   

- How to set up training on your own machine
- Multiscale backbone output
- Deformable self attention
- Deformable cross attention
## How to set up training on your own machine
Deformable DETR uses one custom [module](https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/src/ms_deform_attn.h#L31) for the actual deformable attention that is only implemented on CUDA. This means that you need a machine with actual GPU in it to train or run inference for Deformable DETR. It will not work on mac's "mps" or on cpu.   

To start, clone the [repository](https://github.com/fundamentalvision/Deformable-DETR), download and unpack the [CoCo dataset](https://cocodataset.org/#download). You will need Train and Validation images for 2017, as well as trainval annotations for 2017.  

The official [repo](https://github.com/fundamentalvision/Deformable-DETR/tree/main?tab=readme-ov-file#installation) has some instructions on how to set up. They didn't work for me because my GPU was too modern for the version of cudatoolkit used. Here are the instructions that I used that worked for me:  
```bash
conda create -n deformable_detr python=3.10 pip
conda activate deformable_detr
conda install pytorch=2.1.0 torchvision cudatoolkit=11.8 -c pytorch
cd /home/mak/repos/Deformable-DETR
pip install -r requirements.txt
cd ./models/ops
sh ./make.sh
```
It sets up a fresh environment using anaconda (google *miniconda* or *anaconda* for installation instructions if `conda` command is unavailable for you), installs some libraries, and compiles cpp and CUDA binaries used by the model.  

After installation, you can run the test to make sure everything was compiled correctly:  
```bash
# at ./models/ops
python test.py
```
To run the training, I use:  
```bash
# at the root of the repository
python -u main.py --output_dir exps/r50_deformable_detr --coco_path ~/repos/datasets/coco_original
```
This assumes that unpacked `annotations`, `train2017` and `val2017` folders are located at `~/repos/datasets/coco_original`.  

When running the script, I got the following error:  
```
ImportError: cannot import name '_NewEmptyTensorOp' from 'torchvision.ops.misc' (/home/mak/.local/lib/python3.10/site-packages/torchvision/ops/misc.py)
```
It happens because [this](https://github.com/fundamentalvision/Deformable-DETR/blob/main/util/misc.py#L30) version check compares the package version incorrectly; torchvision version `0.16`, for example, will be incorrectly classified as "earlier than 0.5". To fix this, I simply remove the checks on [these](https://github.com/fundamentalvision/Deformable-DETR/blob/main/util/misc.py#L30-L59) lines.  

After removing the package version checks and installing everything, the script above should work. You should see iterations starting to happen in the console. 
## Multiscale backbone output
Beyond this point, the easiest way to read the code I found is to set up the debugger in VSCode, and then run the [jupyter notebook](./sandbox.ipynb). You can execute all cells except last, and then select the "Debug Cell" option for the last cell, after setting some breakpoints. This allows to interactively call the inference of the model, and step through the execution line by line, printing variables and so on.   
```python
# juyter notebook, inal cell to debug model inference
preds = model(X.to(device))
print(preds.keys())
preds["pred_logits"].shape, preds["pred_boxes"].shape,
```
The model itself looks like several modules wrapped within each other. The outermost module, called [DeformableDETR](https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_detr.py#L35), calls backbone and the DeformableTransformer. The [DeformableTransformer](https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py#L23) calls `DeformableTransformerEncoder` and `DeformableTransformerDecoder`, defined in the same file. Both Encoder and Decoder themselves call `DeformableTransformerEncoderLayer` and `DeformableTransformerDecoderLayer`, 6 times each. All of these modules inherit from `nn.Module`, and have implemented `__init__` for construction and `forward` method for inference. We will start reading from the outermost module, and keep digging deeper.     

Below is the part of the `forward` pass for the outermost layer, `DeformableDETR`, that calls the backbone:  
```python
# source code, deformable_detr.py:114
 def forward(self, samples: NestedTensor):
    if not isinstance(samples, NestedTensor):
        samples = nested_tensor_from_tensor_list(samples)
    features, pos = self.backbone(samples)
```
```python
# vscode debugger
samples.tensors.shape
# torch.Size([2, 3, 1063, 813])
type(features), len(features)
# (<class 'list'>, 3)
features[0].tensors.shape, features[1].tensors.shape, features[2].tensors.shape
# (torch.Size([2, 512, 133, 102]), torch.Size([2, 1024, 67, 51]), torch.Size([2, 2048, 34, 26]))
```
We can see that the input to our `forward` method is a batch of size `2` with images padded to 1063x813. As in the original DETR, all images in the batch are padded to the max dimensions within the batch. The resulting size of every batch will be slightly different.  

First difference to the original detr is that we get 3 "scales" or "levels" as the output of the backbone: 133x102, 67x51, and 34x26. They correspond to processed versions of the image downsampled to 1/8, 1/16 and 1/32 the size. In addition to that, we also compute the fourth level, projecting the smallest, 34x26 backbone output with our own convolutions:  
```python
# source code, deformable_detr.py:60
# DeformableDETR __init__ method
num_backbone_outs = len(backbone.strides)
input_proj_list = []
for _ in range(num_backbone_outs):
    in_channels = backbone.num_channels[_]
    input_proj_list.append(nn.Sequential(
        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
        nn.GroupNorm(32, hidden_dim),
    ))
for _ in range(num_feature_levels - num_backbone_outs):
    input_proj_list.append(nn.Sequential(
        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
        nn.GroupNorm(32, hidden_dim),
    ))
    in_channels = hidden_dim
self.input_proj = nn.ModuleList(input_proj_list)
```
```python
# source code, deformable_detr.py:140
# DeformableDETR forward pass
if self.num_feature_levels > len(srcs):
    _len_srcs = len(srcs)
    for l in range(_len_srcs, self.num_feature_levels):
        if l == _len_srcs:
            src = self.input_proj[l](features[-1].tensors)
        else:
            src = self.input_proj[l](srcs[-1])
        m = samples.mask
        mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
        pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
        srcs.append(src)
        masks.append(mask)
        pos.append(pos_l)
```
We get the fourth level by projecting the third level from the backbone output with `input_proj`. `input_proj` is defined as Conv2d followed by [nn.GroupNorm](https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html) for normalization. We use GroupNorm here instead of BatchNorm because batch size is extremely small. GroupNorm is a better alternative than LayerNorm because it has several normalization groups, thus allowing the network to have a few different parameter groups with different normalizations; it can be useful because not all feature maps are equal.  


```python
# source code, deformable_detr.py:58
# DeformableDETR __init__ method
self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
```
```python
# source code, deformable_detr.py:154
# DeformableDETR forward pass
query_embeds = None
if not self.two_stage:
    query_embeds = self.query_embed.weight
```
```python
# vscode debugger
query_embeds.shape
# torch.Size([300, 512])
```

For query embeddings (used in the decoder) we use [nn.Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html) - randomly initialized, trainable tensor of parameters. Our embedding size within the transformer will be `256`; query embedding has twice the size of that. First half of this "query embedding" will be used as "target" for the decoder layer; second half - as "query positional embedding". For the first decoder layer, `target` and `query positional embedding` are just summed together to get the `Query` input to cross attention; in further levels, previous decoder layer output is used as `target`, while `query positional embedding` is added to that in every layer.  

```python
# source code, deformable_detr.py:157
# DeformableDETR forward pass
hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)
```
We then call the DeformableTransformer, passing in our 4 levels, masks, pisitional embeddings and query embeddings.  
## Deformable self attention
Next, we dig into the modules responsible for self-attention, starting with DeformableTransformer - one module below DeformableDETR.  
### DeformableTransformer code
```python
# source code, deformable_transformer.py:46
# DeformableTransformer __init__ method
self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
```
```python
# source code, deformable_transformer.py:157
# DeformableTransformer forward pass
def forward(self, srcs, masks, pos_embeds, query_embed=None):
    assert self.two_stage or query_embed is not None

    # prepare input for encoder
    src_flatten = []
    mask_flatten = []
    lvl_pos_embed_flatten = []
    spatial_shapes = []
    for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
        bs, c, h, w = src.shape
        spatial_shape = (h, w)
        spatial_shapes.append(spatial_shape)
        src = src.flatten(2).transpose(1, 2)
        mask = mask.flatten(1)
        pos_embed = pos_embed.flatten(2).transpose(1, 2)
        lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
        lvl_pos_embed_flatten.append(lvl_pos_embed)
        src_flatten.append(src)
        mask_flatten.append(mask)
    src_flatten = torch.cat(src_flatten, 1)
    mask_flatten = torch.cat(mask_flatten, 1)
    lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
    spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
    valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

    # encoder
    memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
```
```python
# debugger
src_flatten.shape
# torch.Size([2, 18088, 256])
spatial_shapes
#tensor([[133, 102],
#        [ 67,  51],
#        [ 34,  26],
#        [ 17,  13]], device='cuda:0')
level_start_index
# tensor([    0, 13566, 16983, 17867], device='cuda:0')
valid_ratios
# tensor([[[0.9118, 1.0000],
#         [0.9216, 1.0000],
#         [0.9231, 1.0000],
#         [0.9231, 1.0000]],
#        [[1.0000, 0.5188],
#         [1.0000, 0.5224],
#         [1.0000, 0.5294],
#         [1.0000, 0.5294]]], device='cuda:0')
lvl_pos_embed_flatten.shape
# torch.Size([2, 18088, 256])
mask_flatten.shape
# torch.Size([2, 18088])
```
Withing the `forward` call off DeformableTransformer, we have to flatten the input. By flattening 4 levels with various resolutions, we get a huge sequence of length 18088.   

To make sure that the model has information about which level each sequence element comes from, apart from positional embedding we also have `level embedding` - [nn.Parameter](https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html) - randomly initialized, trainable vector of parameters. 

As we will see further, to compute DeformableAttention, for a certain output sequence element, we need to predict offsets, and then compute a `sampling point` within the image as `offsets` + `reference point`. `Reference point` is just the location within the image that corresponds to the current output sequence element. However, we need to `project` that location on to all the levels, because we compute attention between all the levels. This task is futher complicated by the fact that we pass around padded images, and the padding ratios will be slightly different on every level because of how the interpolation for the mask is done during downsampling: when projecting the mask down from 1063x813 resolution to 1/8, 1/16 and 1/32 version, precise pixel ratios of mask to not-mask will be slightly different on every level. To deal with that, we use `valid_ratios` - the propotion of real image to overall image height and width, computed for each level.   

We also pass in `spatial_shapes` - original shapes of the images on each level before flattening, and `level_start_index` - index within the sequence where each level starts.  

### DeformableTransformerEncoder
Next, we dig into DeformableTransformerEncoder.  
```python
# source code, deformable_transformer.py:157
# DeformableTransformerEncoder forward pass
def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
    output = src
    reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
    for _, layer in enumerate(self.layers):
        output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

    return output
```
```python
# debugger
len(self.layers)
# 6
reference_points.shape
# torch.Size([2, 18088, 4, 2])
src.shape
# torch.Size([2, 18088, 256])
reference_points[0][0]
# tensor([[0.0049, 0.0038],
#        [0.0050, 0.0038],
#        [0.0050, 0.0038],
#        [0.0050, 0.0038]], device='cuda:0')
```
Knowing `spatial_shapes` and `valid_ratios`, we compute `reference_points` - a sequence with the same length as the input. i'th element in that sequence has 4 pairs of x, y coordinates - relative pixel position that corresponds to that pixel elements within each of the 4 different image levels.   

We then apply encoder layer 6 times to obtain a more refined image representation. Output of the encoder is referred to as `memory`.   
### DeformableTransformerEncoderLayer
Next, we dig into the actual encoder layer.
```python
# source code, deformable_transformer.pu:190
# DeformableTransformerEncoderLayer __init__ method
def __init__(self,
                d_model=256, d_ffn=1024,
                dropout=0.1, activation="relu",
                n_levels=4, n_heads=8, n_points=4):
    super().__init__()

    # self attention
    self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
    self.dropout1 = nn.Dropout(dropout)
    self.norm1 = nn.LayerNorm(d_model)

    # ffn
    self.linear1 = nn.Linear(d_model, d_ffn)
    self.activation = _get_activation_fn(activation)
    self.dropout2 = nn.Dropout(dropout)
    self.linear2 = nn.Linear(d_ffn, d_model)
    self.dropout3 = nn.Dropout(dropout)
    self.norm2 = nn.LayerNorm(d_model)
```
```python
# source code, deformable_transformer.pu:219
# DeformableTransformerEncoderLayer forward pass
def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
    # self attention
    src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
    src = src + self.dropout1(src2)
    src = self.norm1(src)

    # ffn
    src = self.forward_ffn(src)

    return src
```
Main deformable attention logic is defined in `MSDeformAttn` custom module (we will read its code later). As is typical with DETRs, we pass in `src` mixed with positional embeddings as `Query` attention input, and plain `src` as `Value` attention input. Apart from attention, we also have residuals, dropout, LayerNorms and dense layers.  
### MSDeformAttn
Next, let's read the MSDeformAttn - the python part of our custom deformable attention module
```python
# source code, ms_deform_attn.py:55
# MSDeformAttn __init__ method
self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
self.value_proj = nn.Linear(d_model, d_model)
self.output_proj = nn.Linear(d_model, d_model)
```
```python
# source code, ms_deform_attn.py:78
# MSDeformAttn forward pass
def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
    N, Len_q, _ = query.shape
    N, Len_in, _ = input_flatten.shape
    assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

    value = self.value_proj(input_flatten)
    if input_padding_mask is not None:
        value = value.masked_fill(input_padding_mask[..., None], float(0))
    value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
    sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
    attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
    attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
    # N, Len_q, n_heads, n_levels, n_points, 2
    if reference_points.shape[-1] == 2:
        offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
        sampling_locations = reference_points[:, :, None, :, None, :] \
                                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
    elif reference_points.shape[-1] == 4:
        sampling_locations = reference_points[:, :, None, :, None, :2] \
                                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
    else:
        raise ValueError(
            'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
    output = MSDeformAttnFunction.apply(
        value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
    output = self.output_proj(output)
    return output
```
Using `Query` input of self attention, we get `sampling_offsets` and `attention_weights` by applying a simple dense layer to every sequence element.   

```python
# debugger
sampling_offsets.shape
# torch.Size([2, 18088, 8, 4, 4, 2]), batch_size * seq_length * attn_heads * level_count * attn_points_count * 2
attention_weights.shape
# torch.Size([2, 18088, 8, 4, 4]), batch_size * seq_length * attn_heads * level_count * attn_points_count
```
We use multihead attention with N=8 attention heads, so for every sequence element, we will have 8 attention heads, 4 levels, 4 attention points to which to attend to; 2 coordinates - x and y - to compute offsets, and 1 attention weight. Note that Deformable Attention doesn't use keys, like normal attention; instead of computing dot product between pairs of Query and Key sequence elements, we just project Query sequence element to arrive at both offsets and attention weights. So in this case, attention weights do not depend on *what we attend to*, only on current sequence element.  

Having reference points and sampling offsets, we then compute `samplig_locations` - fractional pixel coordinates that we need to attend to. If they were integer, we could use native pytorch syntax to *index into* the `value` tensor using these coordinates to select the sequence elements we are interested in, and then multiply them by attention weights tensor. In our case, the pixel coordinates can be fractional, so we use our custom `MSDeformAttnFunction` cpp module to compute bilinear interpolation of these fractional coordinates effectively, on gpu.  

`MSDeformAttnFunction` is defined [here](https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/functions/ms_deform_attn_func.py#L23). It forwards the call to [ms_deform_attn_forward](https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/src/ms_deform_attn.h#L21) cpp function, which, in turn, forwards it to [ms_deform_attn_cuda_forward](https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/src/cuda/ms_deform_attn_cuda.cu#L20), because only cuda version is implemented.  

```c++

at::Tensor ms_deform_attn_cuda_forward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step)
```
Now we have bridged from Python code into c++ code. Thanks to Pytorch's apis, we did it without doing any copying; both c++ and python versions of tensors can be seen as shallow objects that contain a pointer to the data, which might be located on cpu or gpu.   

This function is still written in relatively high-level, Torch Lib code: we receive Tensors as objects, so we can get their shapes, types and so on.  

Next, we bridge into actual cuda code in [ms_deformable_im2col_cuda](https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/src/cuda/ms_deform_im2col_cuda.cuh#L924):  
```c++
template <typename scalar_t>
void ms_deformable_im2col_cuda(cudaStream_t stream,
                              const scalar_t* data_value,
                              const int64_t* data_spatial_shapes, 
                              const int64_t* data_level_start_index, 
                              const scalar_t* data_sampling_loc,
                              const scalar_t* data_attn_weight,
                              const int batch_size,
                              const int spatial_size, 
                              const int num_heads, 
                              const int channels, 
                              const int num_levels, 
                              const int num_query,
                              const int num_point,
                              scalar_t* data_col)
{
  const int num_kernels = batch_size * num_query * num_heads * channels;
  const int num_actual_kernels = batch_size * num_query * num_heads * channels;
  const int num_threads = CUDA_NUM_THREADS;
  ms_deformable_im2col_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
          0, stream>>>(
      num_kernels, data_value, data_spatial_shapes, data_level_start_index, data_sampling_loc, data_attn_weight, 
      batch_size, spatial_size, num_heads, channels, num_levels, num_query, num_point, data_col);
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in ms_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }

}
```
This is significantly more low-level: all the Tensors are gone, and we just get raw points to some data of stuff located on GPU. We then launch the actual gpu kernel - [ms_deformable_im2col_gpu_kernel](https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/src/cuda/ms_deform_im2col_cuda.cuh#L238) - function that will be called on thousands of gpu cores in parallel.   

`ms_deformable_im2col_gpu_kernel` function receives raw pointers to the data, and a bunch of numbers corresponding to all relevant sizes we need to know: number of heads, levels, sequence elements and so on. Each kernel knows its own id, and it will iterate over only those portions of data that were supposed to be assigned to that id (here, it's masked under `CUDA_KERNEL_LOOP` macro that uses `blockIdx`, `blockDim` and  `threadIdx` magical variables that are not defined anywhere, and are filled in by CUDA engine).   

95% of the cuda code is just to deal with the pointers. We receive a bunch of fractional `x` and `y` coordinates in range if [0, 1]. Then we lookup the actual `spatial_shapes` of the images on every level, and convert the sampling locations to coordinates within that pixel space (say, 34x26 resolution). Then we call [ms_deform_attn_im2col_bilinear](https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/src/cuda/ms_deform_im2col_cuda.cuh#L238) that deals with actual bilinear approximation: computes `floor` of the fractional height and width sampling locations to get neighbouring pixels; converts `x` and `y` pixel indices into pointer addresses to lookup data in our huge `bottom_data` array; and does some simple multiplications to get the actual bilinear approximation.
## Deformable cross attention
Now let's come back a couple of modules up, to DeformableTransformer, and discuss the decoder part of it and self attention.
```python
# source code, deformable_transformer.py:54
# DeformableTransformer __init__ method
self.reference_points = nn.Linear(d_model, 2)
```
```python
# source code, deformable_transformer.py:152
# DeformableTransformer forward pass
# encoder, we've read that before
memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

# prepare input for decoder
bs, _, c = memory.shape
if self.two_stage:
    # not used
else:
    query_embed, tgt = torch.split(query_embed, c, dim=1)
    query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
    tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
    reference_points = self.reference_points(query_embed).sigmoid()
    init_reference_out = reference_points

# decoder
hs, inter_references = self.decoder(tgt, reference_points, memory,
                                    spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)
```
As already mentioned above, we split query embeddings with 512 embedding dim into two halves: the one used as query positional embedding, image-independent embedding that is being added to input of every decoder layer; and `tgt` - only used as input for the first decoder layer. Deeper layers will use outputs from previous layer as `tgt`.   

```python
# debugger
reference_points.shape
# torch.Size([2, 300, 2])
```
In self-attention, input and output both represented the image itself. As reference points, we used the position of the output pixel element in the picture, and computed the projection of that on all the levels of the input. In decoder, the "Query" input is not located in image space, so we don't have these a-priori reference points to speak of. Instead, we obtain reference points by passing query positional embedding through a dense layer. These query embeddings are image-independant, and they represent a learnable prior of the queries: they allow queries to specialize and focus on attending to specific parts of the image only.   

```python
# source code, deformable_transformer.py:262
# DeformableTransformerDecoderLayer __init__ method
def __init__(self, d_model=256, d_ffn=1024,
                dropout=0.1, activation="relu",
                n_levels=4, n_heads=8, n_points=4):
    super().__init__()

    # cross attention
    self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
    self.dropout1 = nn.Dropout(dropout)
    self.norm1 = nn.LayerNorm(d_model)

    # self attention
    self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
    self.dropout2 = nn.Dropout(dropout)
    self.norm2 = nn.LayerNorm(d_model)

    # ffn
    self.linear1 = nn.Linear(d_model, d_ffn)
    self.activation = _get_activation_fn(activation)
    self.dropout3 = nn.Dropout(dropout)
    self.linear2 = nn.Linear(d_ffn, d_model)
    self.dropout4 = nn.Dropout(dropout)
    self.norm3 = nn.LayerNorm(d_model)
```
```python
# source code, deformable_transformer.py:297
# DeformableTransformerDecoderLayer forward call
def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
    # self attention
    q = k = self.with_pos_embed(tgt, query_pos)
    tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
    tgt = tgt + self.dropout2(tgt2)
    tgt = self.norm2(tgt)

    # cross attention
    tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                            reference_points,
                            src, src_spatial_shapes, level_start_index, src_padding_mask)
    tgt = tgt + self.dropout1(tgt2)
    tgt = self.norm1(tgt)

    # ffn
    tgt = self.forward_ffn(tgt)

    return tgt
```
In decoder, we use full attention, [nn.MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) native pytorch module for self attention (because query sequence length is just 300), and our custom deformable attention layer as cross attention. As usual, we have a bunch of residuals, dense layers, dropouts and layernorms on top of that.  

## Next up
In the next post, I will talk about DAB-Detr, or Dynamic Anchor Boxes Detr - another addition that helped improve Detr-like models and another step on our road to current SOTA model. 