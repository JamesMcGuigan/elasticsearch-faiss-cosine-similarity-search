#!/usr/bin/env bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
set -x

##### Manual Downloads #####

### CEF Dataset - Manual Download ###

if [[ ! -d datasets/cef/CEF-DM-Multilingual-Benchmark/ ]]; then
    echo
    echo -------------------------------------------------------------
    echo
    echo Download: https://www.elrc-share.eu/repository/download/365a8b821aa011eb913100155d02670611118e05e423402bb729137ecf6ac864/
    echo Unzip archive.zip to create following directory structure:
    echo   datasets/cef/CEF-DM-Multilingual-Benchmark/
    echo
    echo -------------------------------------------------------------
    echo
fi;


### TMX Dataset - Manual Download ###

if [[ ! -d datasets/TMX/ ]]; then
    echo
    echo -------------------------------------------------------------
    echo
    echo Download: https://drive.google.com/drive/folders/1PYrCPwjzaXhdELNd00tpqJRdCsNjUKmF
    echo Unzip file to create following directory structure:
    echo    datasets/TMX/alignment-tool
    echo    datasets/TMX/human-translation-tmx
    echo    datasets/TMX/public-corpora
    echo
    echo -------------------------------------------------------------
    echo
else
    find ./datasets/TMX/alignment-tool/ -name '*.zip' | xargs -tr -L1 -I{} unzip  -f -d ./datasets/TMX/alignment-tool/ "{}"
    find ./datasets/TMX/public-corpora/ -name '*.gz'  | xargs -tr -L1 -I{} gunzip -v -f "{}"
fi;


### "Europarl samples" dataset - Manual Download ###

if [[ ! -f datasets/europarl_samples*.zip ]]; then
    unzip -d datasets/ datasets/europarl_samples*.zip
    # rm datasets/europarl_samples*.zip
fi
if [[ ! -d datasets/europarl_samples/ ]]; then
    echo
    echo -------------------------------------------------------------
    echo
    echo Download: https://drive.google.com/drive/u/2/folders/1wsEDrB9ma6TZIgKWGV4RlAlCia-wlLI5
    echo Unzip file to create following directory structure:
    echo    datasets/TMX/alignment-tool
    echo    datasets/TMX/human-translation-tmx
    echo    datasets/TMX/public-corpora
    echo
    echo -------------------------------------------------------------
    echo
fi



##### Automatic Downloads #####

### Universal Declaration of Human Rights (UDHR) corpus - 372 languages ###
# Source: http://research.ics.aalto.fi/cog/data/udhr/
wget -c http://research.ics.aalto.fi/cog/data/udhr/udhr_txt_20100325.tar.gz -P datasets/
tar -xzf datasets/udhr_txt_20100325.tar.gz -C datasets/ --skip-old-files


### Euro-parliament sentence-aligned corpus (English to 20 European Languages) ###
# Source: https://www.statmt.org/
mkdir -p datasets/europarl
curl -s https://www.statmt.org/europarl/v7/ |
  perl -ne 'if ( /href="([^"]+\.tgz)"/ ) { print "$1\n" }' |
  grep -v 'europarl.tgz\|tools.tgz' |
  xargs -t -L1 -I{} wget -c https://www.statmt.org/europarl/v7/{} -P datasets/europarl/
find datasets/europarl/*.tgz | grep -v 'europarl.tgz\|tools.tgz' |
  xargs -t -L1 -I{} tar -xzf {} -C datasets/europarl/ --one-top-level --skip-old-files



### Euro-parliament Raw Dataset - date aligned, but not sentence aligned ###
mkdir -p datasets/europarl_raw
if [[ -d datasets/europarl/europarl/ ]]; then
  mv -f datasets/europarl/europarl.tgz datasets/europarl_raw/
  mv -f datasets/europarl/tools.tgz    datasets/europarl_raw/
  mv -f datasets/europarl/europarl/*   datasets/europarl_raw/
  mv -f datasets/europarl/tools        datasets/europarl_raw/
fi
wget -c https://www.statmt.org/europarl/v7/europarl.tgz  -P datasets/europarl_raw/
wget -c https://www.statmt.org/europarl/v7/tools.tgz     -P datasets/europarl_raw/
tar -xzf datasets/europarl_raw/europarl.tgz --one-top-level=datasets/europarl_raw/  --skip-old-files
tar -xzf datasets/europarl_raw/tools.tgz --one-top-level -C datasets/europarl_raw/  --skip-old-files
