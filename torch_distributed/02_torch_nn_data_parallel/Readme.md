In the [previous chapter](./01_torch_distributed_basics/Readme.md) we've learned that we can use "data parallel" approach to split input data across multiple executors. We will need our executors to communicate after the end of each batch, to exchange gradients and update model weights.  

One easy way of how to implement this in practice is [torch.nn.DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html).   

Let's assume that we have a pandas dataframe with queries and passages, and we want to compute some sort of similarity score between them using [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) model from Huggingface. This is how we can do it: 
```python
# init model and tokenizer
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
model = AutoModel.from_pretrained("BAAI/bge-m3")
model.eval()

# pre-tokenize everything - to make sure we measure only inference time
tokenized_query = tokenizer(df["query"].tolist(), padding=True, truncation=True, return_tensors="pt", max_length=512)
tokenized_passage = tokenizer(df["passage"].tolist(), padding=True, truncation=True, return_tensors="pt", max_length=512)

# put model and inputs on gpu
DEVICE = "cuda"
model = model.to(DEVICE)
tokenized_query = tokenized_query.to(DEVICE)
tokenized_passage = tokenized_passage.to(DEVICE)


# Apply model to data in batches
from tqdm import tqdm
import time

batch_size = 256
num_samples = len(df)
embeddings_query = []
embeddings_passage = []

start_time = time.time()

with torch.no_grad():
    # Process queries in batches
    for i in tqdm(range(0, num_samples, batch_size), desc="Processing queries"):
        batch_query = {k: v[i:i+batch_size] for k,v in tokenized_query.items()}
        model_output_query = model(**batch_query)
        embeddings_query.append(model_output_query[0][:, 0])
        
    # Process passages in batches  
    for i in tqdm(range(0, num_samples, batch_size), desc="Processing passages"):
        batch_passage = {k: v[i:i+batch_size] for k,v in tokenized_passage.items()}
        model_output_passage = model(**batch_passage)
        embeddings_passage.append(model_output_passage[0][:, 0])

# Concatenate batches
embeddings_query = torch.cat(embeddings_query)
embeddings_passage = torch.cat(embeddings_passage)

dot = (embeddings_query * embeddings_passage).sum(axis=1)

df["score"] = dot.cpu().numpy()

end_time = time.time()
print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")
```
Refer to [sandbox.ipynb](./sandbox.ipynb) for full example.   

We can check gpu consumption using `nvidia-smi`. One thing we'll notice is that only the first gpu is being used. To fix this:   
```python
model = torch.nn.DataParallel(model)
```
Remaining code will be exactly the same, but we might want to increase batch size by a factor of num_gpus.   

If we launch our measuring code now, we will see that all gpus are being used, and that inference time is reduced by a factor of num_gpus!   

The speed-up part is rather obvious. It's the simplicity of the approach that I find interesting. And it stems from the fact that in data parallel approach, we don't need to "communicate" inside of the model. We simply perform forward call as usual, compute gradients, and then exchange gradients between executors. This simple fact allows us to create a simple "wrapper" around any torch.nn.Module class that will do roughly the following:   
- Copy the model across all available gpus
- During forward call, split input data into subbatches and scatter to gpus
- Gather results from all gpus
- (if we are doing training) during `optimizer.step()` call, update model weights on all gpus.   

We call model.forward() as is, which means that we can call arbitrary user code, even if the user was not aware of "data parallel" approach when writing the code.  

Some caveats:  
- torch.nn.DataParallel uses threads for setting up parallel inference processes. In python context, that means that we might be bumping into GIL limitations, especially if gpus are very fast. Pytorch itself recommends not to use torch.nn.DataParallel for training.
- For the same reason, it only works for "one machine, many gpus" scenarios.   

With that in mind, torch.nn.DataParallel is a simple but limited thing (that's why we went through it first!). More solid approach is to launch multiple processes and let them communicate over the network/localhost, which is exactly what happens in [torch.distributed](https://pytorch.org/docs/stable/distributed.html), and most major industry approaches to distributed training inherit from it (huggingface trainer, pytorch lightning, Nvidia's NeMo etc). We will cover those over the course of next chapters.
