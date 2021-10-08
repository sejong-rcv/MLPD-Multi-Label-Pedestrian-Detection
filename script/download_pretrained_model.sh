#!/bin/bash

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo ""
echo "Downloading checkpoint..."
echo ""

ggID='1smXP4xpSDYC8cL_bbT9-E2aywROLlC2v'
ggURL='https://drive.google.com/uc?export=download'

filename="$(curl -sc /tmp/gcokie "${ggURL}&id=${ggID}" | grep -o '="uc-name.*</span>' | sed 's/.*">//;s/<.a> .*//')"
getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"

mkdir ../pretrained
curl -Lb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}" -o "../pretrained/${filename}"

echo ""
echo "Done. Please try 'python src/eval.py --model pretrained/${filename}'."
echo ""