tmux

# Checkout code
echo 
git clone https://github.com/baohq1595/meta-deep-binning.git
git clone https://github.com/baohq1595/data-storage.git

# Install bio, gensim, networkx, ggdrive downloader
pip install biopython &> /dev/null
git clone https://github.com/networkx/networkx-metis.git &> /dev/null
%cd networkx-metis
python setup.py build &> /dev/null
python setup.py install &> /dev/null
pip install gensim
pip install gdown

# Setup data
cd meta-deep-binning
git checkout encode_hash_2

mkdir data
mkdir data/hmp
mkdir data/hmp/raw
python metadec/utils/gg_download.py --link "https://drive.google.com/u/2/uc?id=1ZS5n-XPWSczz65x93S0yMYE0bNHdfSdl&export=download" --name hmp.fna
mv hmp.fna data/hmp/raw/hmp.fna
