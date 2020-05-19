#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import tensorflow as tf
from xl_tensorflow.models.vision.detection.dataloader.common.anchors_yolo import YOLOV3_ANCHORS
import tensorflow.keras.backend as K
from ..body.yolo import yolo_head, box_iou


class YoloLoss(tf.keras.losses.Loss):
    """yolo损失函数
    定义模型为标准输出，把yolo head写入模型里面（即还原成相对坐标形式）
    不把损失函数写入模型里面
    """
    defalt_anchors = YOLOV3_ANCHORS

    def __init__(self,
                 scale_stage,
                 input_shape,
                 num_class,
                 iou_loss="",
                 anchors=None,
                 ignore_thresh=.4,
                 print_loss=False,
                 trunc_inf=False,
                 iou_scale=4.0,
                 avg_loss_object=False,
                 name='yolo_loss'):
        """
        计算每个stage的损失
        Args:
            scale_stage: ie 1: 13X13 2:26X26 3:52X52
            anchors: anchors for yolo
            ignore_thresh: float,0-1, the iou threshold whether to ignore object confidence loss
        """
        super(YoloLoss, self).__init__(reduction=tf.losses.Reduction.NONE, name=name)
        anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if (anchors and (len(anchors) // 3) == 3) or (
            not anchors) else [[3, 4, 5], [1, 2, 3]]
        self.scale_stage = scale_stage
        self.ignore_thresh = ignore_thresh
        self.anchor = anchors[anchor_masks[scale_stage]] if \
            anchors else self.defalt_anchors[anchor_masks[scale_stage]]
        self.input_shape = input_shape
        self.num_class = num_class
        self.iou_loss = iou_loss
        self.iou_scale = iou_scale
        self.avg_loss_object = avg_loss_object
        self.print_loss = print_loss
        self.grid_shape = ((input_shape[0] // 32) * (scale_stage + 1), (input_shape[1] // 32) * (scale_stage + 1))
        self.trunc_inf = trunc_inf

    def call(self, y_true, y_pred):
        """
        y_pred:  shape like (batch, gridx,gridy,3,(5+class))
        y_true:  shape like  (batch,gridx,gridy,3,(5+class))
        anchors: array, shape=(N, 2), wh, default value:
        num_classes: integer
        ignore_thresh: float, the iou threshold whether to ignore object confidence loss
        Returns
        loss: tensor, shape=(1,)

        """
        loss = 0

        batch_tensor = tf.cast(tf.shape(y_pred)[0], K.dtype(y_true[0]))
        grid_shape = K.cast(K.shape(y_pred)[1:3], K.dtype(y_true))
        # 真实值掩码，无目标的对应位置为0 ,shape like gridx,gridy,3,1
        object_mask = y_true[..., 4:5]
        object_count = tf.reduce_sum(object_mask) + 1.0
        true_class_probs = y_true[..., 5:]
        input_shape = tf.shape(y_pred)[1:3]
        grid, raw_pred, pred_xy, pred_wh = yolo_head(y_pred,
                                                     self.anchor,
                                                     input_shape,
                                                     calc_loss=True)

        pred_box = K.concatenate([pred_xy, pred_wh])
        # relative to specified gird
        raw_true_xy = y_true[..., :2] * grid_shape[::-1] - grid
        # wh 还原到与yolobody对应的值即原文中的tw和th
        raw_true_wh = K.log(y_true[..., 2:4] / self.anchor * self.input_shape[::-1] + 1e-10)
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))  # avoid log(0)=-inf
        # box_loss_scale used for scale imbalance large value for small object and small value for large object
        box_loss_scale = 2 - y_true[..., 2:3] * y_true[..., 3:4]
        # Find ignore mask, iterate over each of batch.

        object_mask_bool = K.cast(object_mask, 'bool')

        def iou_best(elems):
            pred_box_one, y_true_one, object_mask_bool_one = elems
            true_box_one = tf.boolean_mask(y_true_one[..., 0:4], object_mask_bool_one[..., 0])
            iou = box_iou(pred_box_one, true_box_one)
            best_iou = K.cast(K.max(iou, axis=-1) < self.ignore_thresh, tf.float32)
            return best_iou

        ignore_mask = tf.map_fn(iou_best, (pred_box, y_true, object_mask_bool), tf.float32)
        ignore_mask = K.expand_dims(ignore_mask, -1)
        # ---------------------------旧版写法
        # ignore_mask = tf.TensorArray(K.dtype(y_true), size=1, dynamic_size=True)
        # def loop_body(b, ignore_mask):
        #     # 每张图片的真实box,应为二维tensor
        #     true_box = tf.boolean_mask(y_true[b, ..., 0:4], object_mask_bool[b, ..., 0])
        #     # pred_box batch,26,26,3,4，iou shape like  (26,26,3,real_box_num)
        #     iou = box_iou(pred_box[b], true_box)
        #     # 26,26,3，所有grid的预测值与真实box最好的iou值
        #     best_iou = K.max(iou, axis=-1)
        #     # 小于阙值的为真，即确定当前值是否参与计算
        #     ignore_mask = ignore_mask.write(b, K.cast(best_iou < self.ignore_thresh, K.dtype(true_box)))
        #     return b + 1, ignore_mask
        #
        # # ignoremask  即所有与真实框iou小于指定值的位置为1，其他位置为0，即看作负样本
        # _, ignore_mask = tf.while_loop(lambda b, *args: b < batch, loop_body, [0, ignore_mask])
        # ignore_mask = ignore_mask.stack()
        # ignore_mask = K.expand_dims(ignore_mask, -1)
        # ---------------------------旧版写法
        """
        损失函数组成：
            1、中心定位误差，采用交叉熵，只计算有真实目标位置的损失
            2、宽度高度误差，使用L2损失，只计算有真实目标位置的损失
            3、是否包含目标的置信度计算，包含两部分，一个是真实目标位置的binary crossentropy, 
                一个是其他位置的binary crossentropy，不包含与真实目标iou大于0.5的位置，即负样本损失
            4、各个类别的损失函数，只计算有真实目标位置的损失
        """
        # K.binary_crossentropy is helpful to avoid exp overflow.     相等时为极值点
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + \
                          (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                    from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)
        confidence_loss = tf.reduce_sum(confidence_loss) / batch_tensor
        class_loss = tf.reduce_sum(class_loss) / batch_tensor
        if self.avg_loss_object:
            class_loss = class_loss / object_count
            confidence_loss = confidence_loss / object_count
        class_loss = tf.identity(class_loss, "class_loss")
        confidence_loss = tf.identity(confidence_loss, "confidence_loss")
        if self.iou_loss in ("giou", "ciou", "diou", "iou"):
            iou = box_iou(pred_box, y_true, method=self.iou_loss, as_loss=True, trunc_inf=self.trunc_inf)
            iou_loss = object_mask * (1 - tf.expand_dims(iou, -1))
            # todo  是否除以均值
            iou_loss = (tf.reduce_sum(iou_loss) / batch_tensor) * self.iou_scale
            if self.avg_loss_object:
                iou_loss = iou_loss / object_count
            iou_loss = tf.identity(iou_loss, self.iou_loss + "_loss")
            loss += iou_loss + confidence_loss + class_loss
            if self.print_loss:
                tf.print("\n" + str(
                    self.scale_stage) + ":\tiou:", iou_loss, "\tconfidence:", confidence_loss,
                         "\tclass:", class_loss)
        else:
            xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2],
                                                                           from_logits=True)
            wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])
            xy_loss = tf.reduce_sum(xy_loss) / batch_tensor
            wh_loss = tf.reduce_sum(wh_loss) / batch_tensor
            mse_loss = xy_loss + wh_loss
            if self.avg_loss_object:
                mse_loss = mse_loss / object_count
            loss += mse_loss + confidence_loss + class_loss
            if self.print_loss:
                tf.print("\n" + str(
                    self.scale_stage) + ":\tmse:", mse_loss, "\tconfidence:", confidence_loss,
                         "\tclass:", class_loss)

        return loss
