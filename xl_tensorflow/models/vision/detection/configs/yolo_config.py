from .common import Config
from ..body.common import DarknetConv2D_BN_Leaky
from tensorflow.keras import layers


def default_yolo_config():
    """default yolo config """
    config = Config()
    config.name = "yolov3"
    config.image_size = 608
    # ssp module
    config.spp_block = []
    # feature aggregation config
    config.agg_method = "fpn"
    config.agg_inputs_ops = []
    config.backward_ops = []
    config.forward_ops = []
    config.inlevel_backward_ops = []
    config.inlevel_forward_ops = []


    return config


yolo_model_param_dict = {
    "yolov4-608":
        dict(
            name="yolov4-608",
            agg_method="panet",
            # all config order as from p7——>p1 direction
            agg_inputs_ops=[
                [DarknetConv2D_BN_Leaky(filters=128, kernel_size=1, strides=1)],
                [DarknetConv2D_BN_Leaky(filters=256, kernel_size=1, strides=1)],
                [DarknetConv2D_BN_Leaky(filters=512, kernel_size=1, strides=1)]
            ],

            # all config order as from p7——>p1 direction
            backward_ops=[
                [DarknetConv2D_BN_Leaky(filters=256, kernel_size=1, strides=1),
                 layers.UpSampling2D()],
                [DarknetConv2D_BN_Leaky(filters=128, kernel_size=1, strides=1),
                 layers.UpSampling2D()]
            ],
            # all config order as from p1——>p7 direction
            forward_ops=[
                [DarknetConv2D_BN_Leaky(filters=256, kernel_size=3, strides=2), ],
                [DarknetConv2D_BN_Leaky(filters=512, kernel_size=3, strides=2), ]
            ],
            # all config order as from p7——>p1 direction
            inlevel_backward_ops=[
                [],
                [
                    DarknetConv2D_BN_Leaky(filters=256, kernel_size=1, strides=1),
                    DarknetConv2D_BN_Leaky(filters=512, kernel_size=3, strides=1),
                    DarknetConv2D_BN_Leaky(filters=256, kernel_size=1, strides=1),
                    DarknetConv2D_BN_Leaky(filters=512, kernel_size=3, strides=1),
                    DarknetConv2D_BN_Leaky(filters=256, kernel_size=1, strides=1),
                ],
                [
                    DarknetConv2D_BN_Leaky(filters=128, kernel_size=1, strides=1),
                    DarknetConv2D_BN_Leaky(filters=256, kernel_size=3, strides=1),
                    DarknetConv2D_BN_Leaky(filters=128, kernel_size=1, strides=1),
                    DarknetConv2D_BN_Leaky(filters=256, kernel_size=3, strides=1),
                    DarknetConv2D_BN_Leaky(filters=128, kernel_size=1, strides=1),
                ],
            ],
            # all config order as from p1——>p7 direction
            inlevel_forward_ops=[
                [],
                [
                    DarknetConv2D_BN_Leaky(filters=256, kernel_size=1, strides=1),
                    DarknetConv2D_BN_Leaky(filters=512, kernel_size=3, strides=1),
                    DarknetConv2D_BN_Leaky(filters=256, kernel_size=1, strides=1),
                    DarknetConv2D_BN_Leaky(filters=512, kernel_size=3, strides=1),
                    DarknetConv2D_BN_Leaky(filters=256, kernel_size=1, strides=1),
                ],
                [
                    DarknetConv2D_BN_Leaky(filters=512, kernel_size=1, strides=1),
                    DarknetConv2D_BN_Leaky(filters=1024, kernel_size=3, strides=1),
                    DarknetConv2D_BN_Leaky(filters=512, kernel_size=1, strides=1),
                    DarknetConv2D_BN_Leaky(filters=1024, kernel_size=3, strides=1),
                    DarknetConv2D_BN_Leaky(filters=512, kernel_size=1, strides=1),
                ],
            ]
        )
}


def get_yolo_config(model_name='yolov4-608'):
    """Get the default config for yolo based on model name."""
    h = default_yolo_config()
    h.override(yolo_model_param_dict[model_name])
    return h
