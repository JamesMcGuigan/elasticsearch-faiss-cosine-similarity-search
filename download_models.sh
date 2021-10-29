#!/usr/bin/env bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
set -x

# Create Directories
mkdir -p models
mkdir -p models/laser
mkdir -p models/LaBSE_1_tensorflow
mkdir -p models/fasttext
mkdir -p models/punkt

# NLTK Data
# DOCS: https://www.nltk.org/data.html
python3 -m nltk.downloader -d ./models/nltk_data punkt
python3 -m nltk.downloader -d ./models/nltk_data europarl_raw
python3 -m nltk.downloader -d ./models/nltk_data all


# LaBSE Pytorch Model
# DOCS: https://github.com/bojone/labse
if [[ ! -d models/labse ]]; then
  gdown https://drive.google.com/uc?id=14Zaq8RE9NMyJb_9B-lkgFZQ9H1K-U-Nf -O models/labse.zip
  unzip models/labse.zip -d models/
fi


# LaBSE Tensorflow Model
# DOCS: https://tfhub.dev/google/LaBSE/1
wget -c https://tfhub.dev/google/LaBSE/1?tf-hub-format=compressed -O models/LaBSE_1_tensorflow.tar.gz
tar -xvzf models/LaBSE_1.tar.gz -C models/LaBSE_1


# LASER models
# DOCS: https://github.com/facebookresearch/LASER/blob/master/install_models.sh
for FILE in bilstm.eparl21.2018-11-19.pt \
            eparl21.fcodes \
            eparl21.fvocab \
            bilstm.93langs.2018-12-26.pt \
            93langs.fcodes \
            93langs.fvocab; do
    wget -c https://dl.fbaipublicfiles.com/laser/models/$FILE -O models/laser/$FILE
done


# FastText
# DOCS: https://amitness.com/2019/07/identify-text-language-python/
wget -c https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -P models/fasttext/


## PunktSentenceTokenizer - use NLTK_data version insread
# kaggle datasets download nltkdata/punkt --path models/
# wget -c https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip -P models/
# unzip models/punkt.zip -d models/
