from xl_tensorflow.utils.params_dict import ParamsDict
config = ParamsDict()
config.override({"fuck":"you"},is_strict=False)
print(config.as_dict())