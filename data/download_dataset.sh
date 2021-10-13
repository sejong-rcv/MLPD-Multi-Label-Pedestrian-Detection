#!/bin/bash

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

dataset_root=${1}

if [ "$#" -ne 1 ]; then
    echo "Usage: ./download_dataset.sh DATASET_DOWNLOAD_PATH"
    exit 2
fi

# Check if directory exists
if [[ ! -d "${dataset_root}" ]]
then	
	echo "Invalid destination path: ${dataset_root}."	
	exit 2
fi

echo ""
echo "Downloading dataset to ${dataset_root}"
echo ""

url="https://kaist-cvpr15-dataset.s3.ap-northeast-2.amazonaws.com/kaist-cvpr15.tar.gz"
wget --directory-prefix=${dataset_root} ${url}

echo ""
echo "Extract dataset (takes > 10 mins)"
echo ""
tar zxvf ${dataset_root}/${filename} >/dev/null 2>&1 && rm ${dataset_root}/${filename}

echo ""
echo "Add a symbolic link"
echo ""
ln -s ${dataset_root} kaist-rgbt

echo ""
echo "Done."
echo ""