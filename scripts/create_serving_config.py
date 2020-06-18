#!usr/bin/env python3
# -*- coding: UTF-8 -*-


import os

path = os.getcwd()
models = os.listdir(path)
model_names = [i for i in models if os.path.isdir(f"{path}/{i}")]
base = """model_config_list {{\n{}\n}}"""
config_template = "  config {{\n    name: '{}',\n    " \
                  "base_path: '/models/{}/',\n\tmodel_platform: 'tensorflow'\n  }}"
config_str = base.format(",\n".join(map(lambda x: config_template.format(x, x), model_names)))
with open(f"{path}/models.config", "w", encoding="utf-8") as f:
    f.write(config_str)
