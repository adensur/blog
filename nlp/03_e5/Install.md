Clone the repo and install the dependencies  
```bash
git clone git@github.com:microsoft/unilm.git
cd unilm/simlm
conda create -n e5 -y python=3.11 pip
conda activate e5
conda install -y pytorch -c pytorch
pip install -r requirements.txt
pip install accelerate
```
Eval & train:   
```bash
# download msmarco dataset
bash scripts/download_msmarco_data.sh

# eval finetuned retriever
export DATA_DIR=./data/msmarco_bm25_official/
export OUTPUT_DIR=./tmp/
bash scripts/encode_marco.sh intfloat/simlm-base-msmarco-finetuned
# Perform nearest-neighbor search for queries
bash scripts/search_marco.sh intfloat/simlm-base-msmarco-finetuned dev

bash scripts/search_marco.sh intfloat/simlm-base-msmarco-finetuned trec_dl2019
bash scripts/search_marco.sh intfloat/simlm-base-msmarco-finetuned trec_dl2020

# train retriever
export DATA_DIR=./data/msmarco_bm25_official/
export OUTPUT_DIR=./checkpoint/biencoder/
bash scripts/train_biencoder_marco.sh

# encode & eval scripts
bash scripts/search_marco.sh $OUTPUT_DIR dev

# train biencoder with kd
export DATA_DIR=./data/msmarco_distillation/
export OUTPUT_DIR=./checkpoint/distilled_biencoder/

bash scripts/train_kd_biencoder.sh

# crossencoder
export DATA_DIR=./data/msmarco_reranker/
export OUTPUT_DIR=./checkpoint/cross_encoder_reranker/

# Train cross-encoder re-ranker
bash scripts/train_reranker_marco.sh
```