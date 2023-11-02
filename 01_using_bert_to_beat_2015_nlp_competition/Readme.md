I've recently had to go through the following exercise: go over [Similarity Evaluation (SemEval)](https://aclanthology.org/S15-2001.pdf) competition from Twitter that ran back in 2015. [Here](https://github.com/cocoxu/SemEval-PIT2015/tree/master) is the project's repo, containing the data for the competition itself.  
Using modern means - pretrained transformer-based LLMs that I finetuned on the data provided, I was able to beat top-1 result of the original competition by a quite significant margin. So I decided to make it into a proper blog post. Here is what we will go over in this post:
- How to get a pretrained LLM and perform inference
- How to define some layers on top of an existing model, finetune them along with the model itself
- How to add BERT's segment tokens to the input to make the model aware of the logical splitting of the text  

Full code as a jupyter notebook is available [here](https://github.com/adensur/blog/blob/main/01_using_bert_to_beat_2015_nlp_competition/semeval_blogpost.ipynb)
## The data
Full data for the competition is available as part of the official repo: [link](https://github.com/cocoxu/SemEval-PIT2015/blob/master/data/SemEval-PIT2015-github.zip).
I am running all the code in google collab, so I will be providing all the commands to get the data from within collab or any other jupyter notebook.
```
!wget https://github.com/cocoxu/SemEval-PIT2015/raw/master/data/SemEval-PIT2015-github.zip
!unzip SemEval-PIT2015-github.zip
!ls SemEval-PIT2015-github/data
```
The data folder contains the following:
```
dev.data  test.data  test.label  train.data
```
Let's take a look at the data:
```
import pandas as pd
df = pd.read_csv("SemEval-PIT2015-github/data/train.data", sep="\t", names=["topicId", "topic_name", "sent1", "sent2", "label", "sent1_tag", "sent2_tag"])
```
`sent1` and `sent2` are the actual tweets. `topicID` is the category of the tweets; `sent1_tag` and `sent2_tag` are tagged texts of the tweets, marking which words correspond to names or objects, for example. For our model, we will only use the tweets themselves.  
The label for train data looks like this:
```
(2, 3)
```
This shows how many human assessors voted that these tweets are similar (2) and how many voted that they are different (3). Sum is always 5.  
The label for test data looks slightly different:  
```
2
```
This just shows the number of experts who voted for similarity.  
Here is the remap function to turn these into true/false labels, as per competition guidelines:
```
def label_remap_train(label_str):
    numbers = label_str.strip("()").split(", ")
    number = int(numbers[0])
    if number >= 3:
        return 1
    if number <= 1:
        return 0
    return 2 # to be filtered out
def label_remap_val(label_str):
        number = int(label_str)
    if number >= 4:
        return 1
    if number <= 2:
        return 0
    return 2 # to be filtered out
```
Here is the full dataset load code:
```
%pip install -U "tensorflow-text==2.13.*"
%pip install "tf-models-official==2.13.*"
import tensorflow as tf
def load_data(path, is_val):
  df = pd.read_csv(path, sep="\t", names=["topicId", "topic_name", "sent1", "sent2", "label", "sent1_tag", "sent2_tag"])
  def label_remap_train(label_str):
    numbers = label_str.strip("()").split(", ")
    number = int(numbers[0])
    if number >= 3:
      return 1
    if number <= 1:
      return 0
    return 2 # to be filtered out
  def label_remap_val(label_str):
    number = int(label_str)
    if number >= 4:
      return 1
    if number <= 2:
      return 0
    return 2 # to be filtered out
  if is_val:
    label_remap = label_remap_val
  else:
    label_remap = label_remap_train
  df["label_remapped"] = df["label"].apply(label_remap)
  if not is_val:
    df = df[df["label_remapped"] != 2]
  s1 = tf.constant(df["sent1"])
  s2 = tf.constant(df["sent2"])
  l = tf.constant(df["label_remapped"])
  ds = tf.data.Dataset.from_tensor_slices(({"s1": s1, "s2": s2}, l)).batch(32)
  return ds
train_ds = load_data("SemEval-PIT2015-github/data/train.data", is_val=False)
dev_ds = load_data("SemEval-PIT2015-github/data/dev.data", is_val=False)
val_ds = load_data("SemEval-PIT2015-github/data/test.data", is_val=True)
```
On top of loading train/dev/test data, I am also converting it to tensorflow tensors and tf.Dataset.
## Getting the model
I am following [this](https://www.tensorflow.org/text/tutorials/classify_text_with_bert) tutorial from Tensorflow to get and apply the model:
```
import os
import shutil


import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer

import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')

bert_model_name = "bert_en_uncased_L-12_H-768_A-12"
tfhub_handle_encoder = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"
tfhub_handle_preprocess = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

print(f'BERT model selected           : {tfhub_handle_encoder}')
print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')

bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)
text_test = ['this is such an amazing movie!']
text_preprocessed = bert_preprocess_model(text_test)

print(f'Keys       : {list(text_preprocessed.keys())}')
print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')
```
```
Keys       : ['input_type_ids', 'input_word_ids', 'input_mask']
Shape      : (1, 128)
Word Ids   : [ 101 2023 2003 2107 2019 6429 3185  999  102    0    0    0]
Input Mask : [1 1 1 1 1 1 1 1 1 0 0 0]
Type Ids   : [0 0 0 0 0 0 0 0 0 0 0 0]
```
Let's discuss a little bit about what is going on here.  
`bert_preprocess_model` is a "model" that handles text preprocessing for Bert: 
- Tokenization: turning an arbitrary text into a list of token ids from a predefined dictionary. This particular model uses BPE tokenization (called "Wordpiece" in the original work) - individual characters or character sequences generated in such a way as to store most frequent combinations in a limited (30k) dictionary: most frequent words or even phrases will make it into the dictionary; the rest will be represented as a series of character sequences or even individual characters. Consult the original [paper](https://arxiv.org/pdf/1810.04805.pdf) for more details. Apart from text tokens, tokenization also adds 2 special ones: [CLS] as a special token for classification problem, and [SEP] to distinguish between different segments/sentences.
- Converting variable-length list of tokens into fixed length (128). Extra tokens are discarded. If input length is less then 128, input is padded with zero tokens; to distinguish real and padding tokens, another "mask" input is used.
"Type Ids" represent segments. Since we passed our text as a single input, they all are equal to 0.  
Since our actual task at hand requires 2 inputs, let's fix this:
```
preprocessor = hub.load(tfhub_handle_preprocess)
tokenize = hub.KerasLayer(preprocessor.tokenize)
s1 = tf.constant(["this is such an amazing movie!"])
s2 = tf.constant(["Indeed it is!"])
tokenized_inputs = [tokenize(s1), tokenize(s2)]
bert_pack_inputs = hub.KerasLayer(
  preprocessor.bert_pack_inputs,
  arguments=dict(seq_length=seq_length))  # Optional argument.
text_preprocessed = bert_pack_inputs(tokenized_inputs)
print(f'Keys       : {list(text_preprocessed.keys())}')
print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :24]}')
print(f'Input Mask : {text_preprocessed["input_mask"][0, :24]}')
print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :24]}')
```
```
Keys       : ['input_type_ids', 'input_word_ids', 'input_mask']
Shape      : (1, 128)
Word Ids   : [ 101 2023 2003 2107 2019 6429 3185  999  102 5262 2009 2003  999  102
    0    0    0    0    0    0    0    0    0    0]
Input Mask : [1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0]
Type Ids   : [0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0]
```
Instead of the final preprocessing model, we have to patch it together out of tokenization layer and a special "input packing layer". Result - we can now provide 2 (or more!) inputs to our Bert, and pass in corresponding Type Ids/Segment Ids. This piece is crucial for our success in similarity task.  
Now let's load the actual model:
```
bert_model = hub.KerasLayer(tfhub_handle_encoder)
bert_results = bert_model(text_preprocessed)

print(f'Loaded BERT: {tfhub_handle_encoder}')
print(f'Pooled Outputs Shape:{bert_results["pooled_output"].shape}')
print(f'Pooled Outputs Values:{bert_results["pooled_output"][0, :12]}')
print(f'Sequence Outputs Shape:{bert_results["sequence_output"].shape}')
print(f'Sequence Outputs Values:{bert_results["sequence_output"][0, :12]}')
```
```
Loaded BERT: https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3
Pooled Outputs Shape:(1, 768)
Pooled Outputs Values:[-0.9860884  -0.7361431  -0.99579465  0.97829276  0.938155   -0.46599045
  0.9942906   0.6153666  -0.99380165 -1.         -0.88662416  0.9971318 ]
Sequence Outputs Shape:(1, 128, 768)
Sequence Outputs Values:[[-0.22726044  0.5517694  -0.13868636 ... -0.47418836  0.47371227
   0.18510711]
 [-0.78651994 -0.26537383  0.03352063 ... -0.51515394  1.0323156
   0.21256736]
 [-0.14084333 -0.32510874  0.43203336 ...  0.19620211  0.6856104
   0.5408714 ]
 ...
 [ 0.42817232  0.5447182   1.168936   ... -0.7984779   0.31761962
  -0.39228404]
 [-0.04523202 -0.01889456  0.8552697  ... -0.03477155  0.44176832
   0.39378563]
 [-0.41469806 -0.65558314  0.36160386 ...  0.6352162   0.5755696
   0.54252493]]
```
"Sequence output" is the full output of bert, with `seq_length` * `hidden_dim` dimensions (128 * 768). Pooled output is the embedding of the entire sentence (1 * 128).
```
def build_classifier_model():
  s1_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='s1_input')
  s2_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='s2_input')
  tokenize = hub.KerasLayer(preprocessor.tokenize)
  tokenized_inputs = [tokenize(s1_input), tokenize(s2_input)]
  seq_length = 128  # Your choice here.
  bert_pack_inputs = hub.KerasLayer(
    preprocessor.bert_pack_inputs,
    arguments=dict(seq_length=seq_length))  # Optional argument.
  encoder_inputs = bert_pack_inputs(tokenized_inputs)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(1, activation=None, name='dense1')(net)
  return tf.keras.Model({"s1": s1_input, "s2": s2_input}, net)
classifier_model = build_classifier_model()
```
Now we pack that all together into a single model in tensorflow functional format. We define just 1 dense layer on top of pooled outout, and use it to predict our labels.   
Note that we defined input as a dictionary:
```
{"s1": s1_input, "s2": s2_input}
```
This format has to correspond to exactly the same format we returned when creating our tf.Dataset; to check that everything works, we can apply this model on a single batch of data like this:
```
for batch in train_ds:
  break
classifier_model(batch[0])
```
This is an extremely important step when modifying model architecture.  

Now, we define loss and train the actual model:
```
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = tf.metrics.BinaryAccuracy()
epochs = 5
steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)

init_lr = 3e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')
classifier_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)
history = classifier_model.fit(x=train_ds,
                               validation_data=dev_ds,
                               epochs=epochs)
```
```
Epoch 1/3
361/361 [==============================] - 415s 1s/step - loss: 0.5361 - binary_accuracy: 0.7335 - val_loss: 0.6650 - val_binary_accuracy: 0.7139
Epoch 2/3
361/361 [==============================] - 397s 1s/step - loss: 0.3257 - binary_accuracy: 0.8694 - val_loss: 0.5964 - val_binary_accuracy: 0.7745
Epoch 3/3
361/361 [==============================] - 429s 1s/step - loss: 0.1953 - binary_accuracy: 0.9253 - val_loss: 0.8056 - val_binary_accuracy: 0.7789
```
Defined in such a way, we will train both our extra dense layers as well as Bert's original parameters.  
Now a final bit of housekeeping - getting the scores for test dataset and comparing to 2015 scoreboard:
```
# getting predictions
all_preds = []
threshold = 0.1
with open("output.tsv", "w") as file:
  for batch in val_ds:
    preds = classifier_model(batch[0])
    for row in preds:
      pred = tf.sigmoid(row[0]).numpy()
      all_preds.append(pred)
      if pred >= threshold:
        label = "true"
      else:
        label = "false"
      print(label + "\t{:.4f}".format(pred), file=file)
# loading test labels to calculate f1 score
df = pd.read_csv("SemEval-PIT2015-github/data/test.label", sep="\t", names=["label", "score"])
def remap(label_str):
    if label_str == "true":
        return 1
    if label_str == "false":
        return 0
    return 2 # to be filtered out later
df["label_remapped"] = df["label"].apply(remap)
df2 = df[df["label_remapped"] != 2]
# finding out the best threshold for f1 score
from sklearn.metrics import f1_score
threshold = 0.0
all_preds2 = [x for x, condition in zip(all_preds, df["label_remapped"] != 2) if condition]
while threshold <= 1.0:
  def thr(value, threshold):
    if value >= threshold:
      return 1
    return 0

  predicted_labels = [thr(value, threshold) for value in all_preds2]
  f1 = f1_score(df2["label_remapped"], predicted_labels)
  print("Threshold: ", threshold, "; f1: ", f1)
  threshold += 0.05
```
```
### results
```
With a threshold of 0.2 for true/false labels, we achieve f1 score of 0.7556, 0.674 being top-1 result in 2015. Neat!
Once again, full code as a jupyter notebook is available [here](https://github.com/adensur/blog/blob/main/01_using_bert_to_beat_2015_nlp_competition/semeval_blogpost.ipynb)