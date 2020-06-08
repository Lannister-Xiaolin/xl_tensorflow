#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import tensorflow as tf
from xl_tensorflow.models.vision import EfficientNetB1
from xl_tool.xl_io import read_json, save_to_json
import xl_tensorflow.models.vision.detection.configs.factory as config_factory
from xl_tensorflow.models.vision.detection.architecture.fpn import BiFpn
from xl_tensorflow.models.vision import EfficientNetB0
from xl_tensorflow.models.vision.detection.body.efficientdet_model import EfficientDetModel
import numpy as np
from tensorflow.keras import layers, Model
from xl_tensorflow.models.vision.detection.dataloader.input_reader import InputFn
from xl_tensorflow.models.vision.detection.inference.efficientdet_inference import det_post_process_combined, \
    batch_image_preprocess




params = config_factory.config_generator("efficientdet-d1")
params.architecture.num_classes = 21
input_reader = InputFn(r"E:\Temp\test\tfrecord\*.tfrecord", params, "train", 1)
train_dataset = input_reader(batch_size=4)
model_fn = EfficientDetModel(params)

model = model_fn.build_model(params)
loss_fn = model_fn.build_loss_fn()
epochs = 10

optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4)
for epoch in range(epochs):
  print('Start of epoch %d' % (epoch,))

  # Iterate over the batches of the dataset.
  for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

    # Open a GradientTape to record the operations run
    # during the forward pass, which enables autodifferentiation.
    with tf.GradientTape() as tape:

      # Run the forward pass of the layer.
      # The operations that the layer applies
      # to its inputs are going to be recorded
      # on the GradientTape.
      logits = model(x_batch_train, training=True)  # Logits for this minibatch

      # Compute the loss value for this minibatch.
      loss_value = loss_fn(y_batch_train, logits)

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss_value['total_loss'], model.trainable_weights)

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # Log every 200 batches.
    if step % 2 == 0:
        print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value['total_loss'])))
        print('Seen so far: %s samples' % ((step + 1) * 64))