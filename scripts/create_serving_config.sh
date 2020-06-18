#!/usr/bin/env bash
# $NF表示最后一列
base = 'model_config_list {{\n{}\n}}'
config_template = "  config {{\n    name: \'{}\',\n    base_path: \'/models/{}/\',\n\tmodel_platform: \'tensorflow\'\n  }}"
d=$(ls -l $1|awk '/^d/ {print $NF}')
for i in $d
do
 echo $i
done