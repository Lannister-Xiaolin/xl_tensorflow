# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""An executor class for running model on TensorFlow 2.0."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

from absl import logging

import tensorflow as tf
from xl_tensorflow.training import xl_distributed_executor as executor
from ..dataloader.utils import visualization_utils
import gc

class DetectionDistributedExecutor(executor.DistributedExecutor):
    """Detection specific customer training loop executor.

    Subclasses the DistributedExecutor and adds support for numpy based metrics.
    """

    def __init__(self,
                 predict_post_process_fn=None,
                 trainable_variables_filter=None,
                 **kwargs):
        super(DetectionDistributedExecutor, self).__init__(**kwargs)
        if predict_post_process_fn:
            assert callable(predict_post_process_fn)
        if trainable_variables_filter:
            assert callable(trainable_variables_filter)
        self._predict_post_process_fn = predict_post_process_fn
        self._trainable_variables_filter = trainable_variables_filter
        self.eval_steps = tf.Variable(
            0,
            trainable=False,
            dtype=tf.int32,
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            shape=[])

    def _create_replicated_step(self,
                                strategy,
                                model,
                                loss_fn,
                                optimizer,
                                metric=None):
        trainable_variables = model.trainable_variables
        if self._trainable_variables_filter:
            trainable_variables = self._trainable_variables_filter(
                trainable_variables)
        logging.info('Filter trainable variables from %d to %d',
                     len(model.trainable_variables), len(trainable_variables))
        _update_state = lambda labels, outputs: None
        if isinstance(metric, tf.keras.metrics.Metric):
            _update_state = lambda labels, outputs: metric.update_state(
                labels, outputs)
        else:
            logging.error('Detection: train metric is not an instance of '
                          'tf.keras.metrics.Metric.')

        def _replicated_step(inputs):
            """Replicated training step."""
            inputs, labels = inputs

            with tf.GradientTape() as tape:
                outputs = model(inputs, training=True)
                all_losses = loss_fn(labels, outputs)
                losses = {}
                for k, v in all_losses.items():
                    losses[k] = tf.reduce_mean(v)
                # 此处等价于官方文档中的 scale_loss = tf.reduce_sum(loss) * (1. / GLOBAL_BATCH_SIZE)，但最好保证batch size能被gpu数量整除
                per_replica_loss = losses['total_loss'] / strategy.num_replicas_in_sync
                _update_state(labels, outputs)

            grads = tape.gradient(per_replica_loss, trainable_variables)
            optimizer.apply_gradients(zip(grads, trainable_variables))
            return losses

        return _replicated_step

    def _create_test_step(self, strategy, model, metric, loss_fn=None):
        """Creates a distributed test step."""

        @tf.function
        def test_step(iterator, eval_steps):
            """Calculates evaluation metrics on distributed devices."""

            def _test_step_fn(inputs, eval_steps):
                """Replicated accuracy calculation."""
                inputs, labels = inputs
                model_outputs = model(inputs, training=False)
                all_losses = loss_fn(labels, model_outputs)
                # if self._predict_post_process_fn:
                # labels, prediction_outputs = self._predict_post_process_fn(
                #     labels, model_outputs)
                model_outputs.update({
                    'source_id': labels['groundtruths']['source_id'],
                    'image_info': labels['image_info'],
                })
                prediction_outputs = {
                    'source_id': labels['groundtruths']['source_id'],
                    'image_info': labels['image_info'],
                    'num_detections': model_outputs['num_detections'],
                    'detection_boxes': model_outputs['detection_boxes'],
                    'detection_classes': model_outputs['detection_classes'],
                    'detection_scores': model_outputs['detection_scores'],
                }
                labels = {
                    'source_id': labels['groundtruths']['source_id'],
                    'image_info': labels['image_info'],
                    'num_detections': labels['groundtruths']['num_detections'],
                    'boxes': labels['groundtruths']['boxes'],
                    'classes': labels['groundtruths']['classes'],
                    'areas': labels['groundtruths']['areas'],
                    'is_crowds': labels['groundtruths']['is_crowds'],
                    'height': labels['height'],
                    'width': labels['width'],
                }
                losses = {}
                for k, v in all_losses.items():
                    losses[k] = tf.reduce_mean(v)
                return labels, prediction_outputs, losses

            labels, outputs, per_replica_losses = strategy.experimental_run_v2(
                _test_step_fn, args=(
                    next(iterator),
                    eval_steps,
                ))
            outputs = tf.nest.map_structure(strategy.experimental_local_results,
                                            outputs)
            labels = tf.nest.map_structure(strategy.experimental_local_results,
                                           labels)
            losses = tf.nest.map_structure(
                lambda x: strategy.reduce(tf.distribute.ReduceOp.MEAN, x, axis=None),
                per_replica_losses)

            eval_steps.assign_add(self._params.eval.batch_size)
            return labels, outputs, losses

        return test_step

    def _run_evaluation(self, test_step, current_training_step, metric,
                        test_iterator):
        """Runs validation steps and aggregate metrics."""
        self.eval_steps.assign(0)
        if not test_iterator or not metric:
            logging.warning(
                'Both test_iterator (%s) and metrics (%s) must not be None.',
                test_iterator, metric)
            return None
        logging.info('Running evaluation after step: %s.', current_training_step)
        eval_step = 0
        eval_losses = {}
        while True:
            try:
                labels, outputs, losses = test_step(test_iterator, self.eval_steps)
                if metric:
                    metric.update_state(labels, outputs)
                eval_step += 1
                logging.info('----->evaluation  step: %s.', eval_step)
                try:
                    for k, v in losses.items():
                        eval_losses[k].append(losses[k])
                except KeyError:
                    for k, v in losses.items():
                        eval_losses[k] = []
                        eval_losses[k].append(losses[k])
                # 调试
                # break
            except (StopIteration, tf.errors.OutOfRangeError):
                del labels
                del outputs
                del losses
                break
        for k, v in eval_losses.items():
            eval_losses[k] = tf.reduce_mean(tf.stack(eval_losses[k])).numpy().astype(float)
        metric_result = metric.result()
        if isinstance(metric, tf.keras.metrics.Metric):
            metric_result = tf.nest.map_structure(lambda x: x.numpy().astype(float),
                                                  metric_result)
        logging.info('Step: [%d] Validation metric = %s', current_training_step,
                     metric_result)
        metric_result.update(eval_losses)
        del eval_losses
        gc.collect()
        return metric_result
