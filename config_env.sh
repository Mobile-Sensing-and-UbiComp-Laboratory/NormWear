# create conda env (customize env name and python version if needed)
conda create --name normwear-env python=3.11.5
conda activate normwear-env

# instal basic libs
pip install pandas
pip install tabulate
pip install scipy
pip install scikit-learn

# install pytorch (change cuda version if needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# # if cpu only
# conda install pytorch torchvision torchaudio cpuonly -c pytorch

# install other modules for modeling
pip install timm
pip install transformers
pip install pytorch_pretrained_vit