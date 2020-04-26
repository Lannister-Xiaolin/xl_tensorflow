#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import tensorflow as tf
import tensorflow_addons as tfa
import math

def blend(image1, image2, factor):
    """Blend image1 and image2 using 'factor'.

    Factor can be above 0.0.  A value of 0.0 means only image1 is used.
    A value of 1.0 means only image2 is used.  A value between 0.0 and
    1.0 means we linearly interpolate the pixel values between the two
    images.  A value greater than 1.0 "extrapolates" the difference
    between the two pixel values, and we clip the results to values
    between 0 and flip.
    if input value is 0-255,please set flip=255.0
    Args:
      image1: An image Tensor
      image2: An image Tensor
      factor: A floating point value above 0.0.

    Returns:
      A blended image Tensor of type as input
    """
    assert 0.0 <= factor <= 1.0
    image1 = tf.convert_to_tensor(image1)
    image2 = tf.convert_to_tensor(image2)
    dtype = image1.dtype
    if factor == 0.0:
        return image1
    if factor == 1.0:
        return image2

    image1 = tf.cast(image1, tf.float32)
    image2 = tf.cast(image2, tf.float32)
    assert image1.shape == image2.shape
    difference = image2 - image1
    scaled = factor * difference
    temp = image1 + scaled
    flip = 255 if dtype == tf.uint8 else 1.0
    temp = tf.clip_by_value(temp, 0.0, flip)
    return tf.cast(temp, dtype)


def cutout(image, pad_size, replace=0):
    """Apply cutout (https://arxiv.org/abs/1708.04552) to image.

    This operation applies a (2*pad_size x 2*pad_size) mask of zeros to
    a random location within `img`. The pixel values filled in will be of the
    value `replace`. The located where the mask will be applied is randomly
    chosen uniformly over the whole image.
    please confirm replace is 0-255 if image is uint8, 0-1.0 if image is float32
    Args:
      image: An image Tensor of type uint8 or float32.
      pad_size: Specifies how big the zero mask that will be generated is that
        is applied to the image. The mask will be of size
        (2*pad_size x 2*pad_size).
      replace: What pixel value to fill in the image in the area that has
        the cutout mask applied to it.

    Returns:
      An image Tensor that is of type input.
    """
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]
    # Sample the center location in the image where the zero mask will be applied.
    cutout_center_height = tf.keras.backend.random_uniform(
        shape=[], minval=0, maxval=image_height,
        dtype=tf.int32)

    cutout_center_width = tf.keras.backend.random_uniform(
        shape=[], minval=0, maxval=image_width,
        dtype=tf.int32)

    lower_pad = tf.maximum(0, cutout_center_height - pad_size)
    upper_pad = tf.maximum(0, image_height - cutout_center_height - pad_size)
    left_pad = tf.maximum(0, cutout_center_width - pad_size)
    right_pad = tf.maximum(0, image_width - cutout_center_width - pad_size)

    cutout_shape = [image_height - (lower_pad + upper_pad),
                    image_width - (left_pad + right_pad)]
    padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
    mask = tf.pad(
        tf.zeros(cutout_shape, dtype=image.dtype),
        padding_dims, constant_values=1)
    mask = tf.expand_dims(mask, -1)
    mask = tf.tile(mask, [1, 1, 3])
    image = tf.where(
        tf.equal(mask, 0),
        tf.ones_like(image, dtype=image.dtype) * replace,
        image)
    return image


def shear_x(image, level):
    # Shear parallel to x axis is a projective transform
    # with a matrix form of,:
    # level is -1,1,recommend -0.1,0.1
    image = tfa.image.transform(tf.convert_to_tensor(image), tf.constant([1., level, 0., 0., 1., 0., 0., 0.]))
    return image


def shear_y(image, level):
    # Shear parallel to x axis is a projective transform
    # with a matrix form of,:
    # level is -1,1,recommend -0.1,0.1
    image = tfa.image.transform(
        tf.convert_to_tensor(image), tf.constant([1., 0, 0., level, 1., 0., 0., 0.]))
    return image


def equalize(image):
    """Implements Equalize function from PIL using TF ops."""
    image = tf.convert_to_tensor(image)
    dtype = image.dtype
    if dtype not in [tf.uint8, tf.int32, tf.uint16, tf.int64, tf.uint32]:
        image = image * 255.0

    def scale_channel(im, c):
        """Scale the data in the channel to implement equalize."""
        im = tf.cast(im[:, :, c], tf.int32)
        # Compute the histogram of the image channel.
        histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)

        # For the purposes of computing the step, filter out the nonzeros.
        nonzero = tf.where(tf.not_equal(histo, 0))
        nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
        step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (tf.cumsum(histo) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = tf.concat([[0], lut[:-1]], 0)
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return tf.clip_by_value(lut, 0, 255)

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        result = tf.cond(tf.equal(step, 0),
                         lambda: im,
                         lambda: tf.gather(build_lut(histo, step), im))

        return tf.cast(result, tf.uint8)

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image, 0)
    s2 = scale_channel(image, 1)
    s3 = scale_channel(image, 2)
    image = tf.stack([s1, s2, s3], 2)
    if dtype not in [tf.uint8, tf.int32, tf.uint16, tf.int64, tf.uint32]:
        image = tf.cast(image, tf.float32) / 255
    return image


def color(image, factor):
    """Equivalent of PIL Color
    factor 0-1,means  grey color to origin color
    """
    image = tf.convert_to_tensor(image)
    dtype = image.dtype
    if dtype not in [tf.uint8, tf.int32, tf.uint16, tf.int64, tf.uint32]:
        image = tf.cast(image * 255.0, tf.uint8)
    degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
    degenerate = blend(degenerate, image, factor)
    if dtype not in [tf.uint8, tf.int32, tf.uint16, tf.int64, tf.uint32]:
        degenerate = tf.cast(degenerate, tf.float32) / 255
    return degenerate


def rotate(image, degrees):
    """Rotates the image by degrees either clockwise or counterclockwise.

    Args:
      image: An image Tensor of type uint8.
      degrees: Float, a scalar angle in degrees to rotate all images by. If
        degrees is positive the image will be rotated clockwise otherwise it will
        be rotated counterclockwise.
      replace: A one or three value 1D tensor to fill empty pixels caused by
        the rotate operation.

    Returns:
      The rotated version of image.
    """
    # Convert from degrees to radians.
    degrees_to_radians = math.pi / 180.0
    radians = degrees * degrees_to_radians
    return tfa.image.rotate(tf.convert_to_tensor(image), tf.constant(radians))


def translate_x(image, pixels):
    """Equivalent of PIL Translate in X dimension."""
    image = tfa.image.translate(image, tf.constant([-pixels, 0], dtype=tf.float32))
    return image


def translate_y(image, pixels):
    """Equivalent of PIL Translate in X dimension."""
    image = tfa.image.translate(image, tf.constant([0, -pixels], dtype=tf.float32))
    return image









