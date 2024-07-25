## Installation and debugging setup
- Set up new env and install some libraries
```bash
conda create -n gpt2 -y python=3.10 pip
conda activate gpt2
# install pytorch
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
# execute from the root of transformers repo
pip install -e .
```
- Place the [notebook](./sandbox.ipynb) at the root of the "transformers" repository
- Open the folder with the repository with vscode
- Run the jupyter notebook with vscode
- Make sure to select "gpt2" conda environment that we've created above for your kernel source
