from .common import Config
from xl_tensorflow.models.vision.classification.darknet import DarknetConv2D_BN_Leaky, DarknetConv2D
from tensorflow.keras import layers


def default_yolo_config():
    """default yolo config """
    config = Config()
    config.name = "yolov3"
    config.image_size = 608
    config.anchors_per_grid = 3
    config.num_classes = 80
    # ssp module
    config.spp = False
    # feature aggregation config
    config.agg_method = "fpn"
    config.agg_inputs_ops = []
    config.backward_ops = []
    config.forward_ops = []
    config.inlevel_backward_ops = []
    config.inlevel_forward_ops = []
    config.agg_out_ops = []

    return config


def get_model_param(model_name='yolov4', num_anchors=3, num_classes=80):
    yolo_out_size = (num_classes + 5) * num_anchors
    yolo_model_param_dict = {
        "yolov4":
            dict(
                name="yolov4",
                agg_method="panet",
                spp=True,
                # all config order as from  p1——>p7 direction
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
                ],
                # all config order as from  p1——>p7 direction
                agg_out_ops=[
                    [DarknetConv2D_BN_Leaky(filters=256, kernel_size=3, strides=1),
                     DarknetConv2D(filters=yolo_out_size, kernel_size=1, strides=1)],
                    [DarknetConv2D_BN_Leaky(filters=512, kernel_size=3, strides=1),
                     DarknetConv2D(filters=yolo_out_size, kernel_size=1, strides=1)],
                    [DarknetConv2D_BN_Leaky(filters=1024, kernel_size=3, strides=1),
                     DarknetConv2D(filters=yolo_out_size, kernel_size=1, strides=1)]
                ]
            ),
        "yolov3": dict(
            name="yolov3",
            agg_method="fpn",
            spp=False,
            agg_inputs_ops=[
                [],
                [],
                []
            ],
            backward_ops=[
                [DarknetConv2D_BN_Leaky(filters=256, kernel_size=1, strides=1),
                 layers.UpSampling2D()],
                [DarknetConv2D_BN_Leaky(filters=128, kernel_size=1, strides=1),
                 layers.UpSampling2D()]
            ],
            inlevel_backward_ops=[
                [DarknetConv2D_BN_Leaky(filters=512, kernel_size=1, strides=1),
                 DarknetConv2D_BN_Leaky(filters=1024, kernel_size=3, strides=1),
                 DarknetConv2D_BN_Leaky(filters=512, kernel_size=1, strides=1),
                 DarknetConv2D_BN_Leaky(filters=1024, kernel_size=3, strides=1),
                 DarknetConv2D_BN_Leaky(filters=512, kernel_size=1, strides=1), ],
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
            # all config order as from  p1——>p7 direction
            agg_out_ops=[
                [DarknetConv2D_BN_Leaky(filters=256, kernel_size=3, strides=1),
                 DarknetConv2D(filters=yolo_out_size, kernel_size=1, strides=1)],
                [DarknetConv2D_BN_Leaky(filters=512, kernel_size=3, strides=1),
                 DarknetConv2D(filters=yolo_out_size, kernel_size=1, strides=1)],
                [DarknetConv2D_BN_Leaky(filters=1024, kernel_size=3, strides=1),
                 DarknetConv2D(filters=yolo_out_size, kernel_size=1, strides=1)]
            ]
        )
    }
    return yolo_model_param_dict[model_name]


def get_yolo_config(model_name='yolov4', num_anchors=3, num_classes=80):
    """Get the default config for yolo based on model name."""
    h = default_yolo_config()
    h.override(get_model_param(model_name=model_name,
                               num_anchors=num_anchors, num_classes=num_classes))
    return h
