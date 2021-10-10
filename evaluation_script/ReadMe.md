## Evalutation_script

You can evaluate the model with the code.
For annotations file, only json is supported, and for result files, json and txt formats are supported.
(multiple `--rstFiles` are supported)

Example)

```bash
$ python evaluation_script.py \
	--annFile evaluation_script/KAIST_annotation.json \
	--rstFile evaluation_script/MLPD_result.txt \
			  evaluation_script/ARCNN_result.txt \
			  evaluation_script/CIAN_result.txt \
			  evaluation_script/MSDS-RCNN_result.txt \
			  evaluation_script/MBNet_result.txt
```

