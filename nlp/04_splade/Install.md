```bash
# getting the code
git clone git@github.com:naver/splade.git
cd splade

# creating the environment
conda create -n splade -y python=3.11 pip
conda activate splade
conda install -y pytorch -c pytorch
pip install hydra-core transformers numba h5py pytrec-eval tensorboard matplotlib


# running train + eval
# toy example
export PYTHONPATH=$PYTHONPATH:$(pwd)
export SPLADE_CONFIG_NAME="config_default.yaml"
python3 -m splade.all \
  config.checkpoint_dir=experiments/debug/checkpoint \
  config.index_dir=experiments/debug/index \
  config.out_dir=experiments/debug/out
  
# getting full data
# simple splade
wget https://download.europe.naverlabs.com/splade/sigir22/triplets.tar.gz
tar -zxvf triplets.tar.gz

# running train + eval
# actual splade
export PYTHONPATH=$PYTHONPATH:$(pwd)
export SPLADE_CONFIG_NAME="config_splade"
python3 -m splade.all \
  config.checkpoint_dir=experiments/splade/checkpoint \
  config.index_dir=experiments/splade/index \
  config.out_dir=experiments/splade/out
```