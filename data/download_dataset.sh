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

ggID='1GRz6dX7CVZe_66vkyBGNziC0NFYIjRws'
ggURL='https://drive.google.com/uc?export=download'

filename="$(curl -sc /tmp/gcache "${ggURL}&id=${ggID}" | grep -o '="uc-name.*</span>' | sed 's/.*">//;s/<.a> .*//')"
getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcache)"

echo "Filename: ${filename}"
curl -Lb ${dataset_root} "${ggURL}&confirm=${getcode}&id=${ggID}" -o "${filename}"

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