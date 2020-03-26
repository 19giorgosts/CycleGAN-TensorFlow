import tensorflow as tf
import ops
import utils

class Generator:
  def __init__(self, name, is_training, ngf=64, norm='instance', image_size=128):
    self.name = name
    self.reuse = False
    self.ngf = ngf
    self.norm = norm
    self.is_training = is_training
    self.image_size = image_size

  def __call__(self, input):
    """
    Args:
      input: batch_size x width x height x 3
    Returns:
      output: same size as input
    """
    with tf.variable_scope(self.name):
      # conv layers
      c7s1_32 = ops.c7s1_k(input, self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='c7s1_32')                             # (?, w, h, 32)
      d64 = ops.dk(c7s1_32, 2*self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='d64')                                 # (?, w/2, h/2, 64)
      d128 = ops.dk(d64, 4*self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='d128')                                # (?, w/4, h/4, 128)

      if self.image_size <= 128:
        # use 6 residual blocks for 128x128 images
        res_output = ops.n_res_blocks(d128, reuse=self.reuse, n=6)      # (?, w/4, h/4, 128)
      else:
        # 9 blocks for higher resolution
        res_output = ops.n_res_blocks(d128, reuse=self.reuse, n=9)      # (?, w/4, h/4, 128)

      # fractional-strided convolution
      u64 = ops.uk(res_output, 2*self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='u64')                                 # (?, w/2, h/2, 64)
      u32 = ops.uk(u64, self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='u32', output_size=self.image_size)         # (?, w, h, 32)

      # conv layer
      # Note: the paper said that ReLU and _norm were used
      # but actually tanh was used and no _norm here
      output = ops.c7s1_k(u32, 3, norm=None,
          activation='tanh', reuse=self.reuse, name='output')           # (?, w, h, 3)
    # set reuse=True for next call
    self.reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    return output

  def sample(self, input):
    image = utils.batch_convert2int(self.__call__(input))
    image = tf.image.encode_jpeg(tf.squeeze(image, [0]))
    return image

class UnetGenerator:

  def __init__(self, name, is_training, ngf=64, norm='instance', batch_size=1, image_size=256):
        self.name = name
        self.reuse = False
        self.ngf = ngf
        self.norm = norm
        self.is_training = is_training
        self.image_size = image_size
        self.batch_size = batch_size

  def __call__(self,input):
        def create_weights(shape):
                return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

        def create_biases(size):
            return tf.Variable(tf.constant(0.05, shape=[size]))

        def conv_layer(input, num_input_channels, conv_filter_size, num_filters, padding='SAME', relu=True):
            weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
            #biases = create_biases(num_filters)
            layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding=padding)
            #layer += biases

            if relu:
                layer = tf.nn.relu(layer)
            return layer

        def pool_layer(input, padding='SAME'):
            return tf.nn.max_pool(value=input,
                                  ksize = [1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding=padding)

        def un_conv(input, num_input_channels, conv_filter_size, num_filters, feature_map_size, train=True, padding='SAME',relu=True):

            weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_filters, num_input_channels])
            #biases = create_biases(num_filters)
            if train:
                batch_size_0 = self.batch_size
            else:
                batch_size_0 = 1
            layer = tf.nn.conv2d_transpose(value=input, filter=weights,
                                           output_shape=[batch_size_0, feature_map_size, feature_map_size, num_filters],
                                           strides=[1, 2, 2, 1],
                                           padding=padding)
            #layer += biases

            if relu:
                layer = tf.nn.relu(layer)
            return layer

        with tf.variable_scope(self.name):        
            train=True
            # train is used for un_conv, to determine the batch size
            #print(input.shape)
            conv1 = conv_layer(input, input.get_shape()[3], 4, 64)
            #print(conv1.shape)
            conv2 = conv_layer(conv1, conv1.get_shape()[3], 4, 64)
            #print(conv2.shape)
            pool2 = pool_layer(conv2)
            #print(pool2.shape)
            conv3 = conv_layer(pool2, pool2.get_shape()[3], 4, 128)
            #print(conv3.shape)
            conv4 = conv_layer(conv3, conv3.get_shape()[3], 4, 128)
            #print(conv4.shape)
            pool4 = pool_layer(conv4)
            #print(pool4.shape)
            conv5 = conv_layer(pool4, pool4.get_shape()[3], 4, 256)
            #print(conv5.shape)
            conv6 = conv_layer(conv5, conv5.get_shape()[3], 4, 256)
            #print(conv6.shape)
            pool6 = pool_layer(conv6)
            #print(pool6.shape)
            conv7 = conv_layer(pool6, pool6.get_shape()[3], 4, 512)
            #print(conv7.shape)
            conv8 = conv_layer(conv7, conv7.get_shape()[3], 4, 512)
            #print(conv8.shape)
            pool8 = pool_layer(conv8)
            #print(pool8.shape)

            conv9 = conv_layer(pool8, pool8.get_shape()[3], 4, 1024)
            #print(conv9.shape)
            conv10 = conv_layer(conv9, conv9.get_shape()[3], 4, 1024)
            #print(conv10.shape)

            conv11 = un_conv(conv10, conv10.get_shape()[3], 4, 512, self.image_size // 8, train)
            #print(conv11.shape)
            merge11 = tf.concat(values=[conv8, conv11], axis = -1)
            #print(merge11.shape)

            conv12 = conv_layer(merge11, merge11.get_shape()[3], 4, 512)
            #print(conv12.shape)
            conv13 = conv_layer(conv12, conv12.get_shape()[3], 4, 512)
            #print(conv13.shape)

            conv14 = un_conv(conv13, conv13.get_shape()[3], 4, 256, self.image_size // 4, train)
            #print(conv14.shape)
            merge14 = tf.concat([conv6, conv14], axis=-1)
            #print(merge14.shape)

            conv15 = conv_layer(merge14, merge14.get_shape()[3], 4, 256)
            #print(conv15.shape)
            conv16 = conv_layer(conv15, conv15.get_shape()[3], 4, 256)
            #print(conv16.shape)

            conv17 = un_conv(conv16, conv16.get_shape()[3], 4, 128, self.image_size // 2, train)
            #print(conv17.shape)
            merge17 = tf.concat([conv17, conv4], axis=-1)
            #print(merge17.shape)

            conv18 = conv_layer(merge17, merge17.get_shape()[3], 4, 128)
            #print(conv18.shape)
            conv19 = conv_layer(conv18, conv18.get_shape()[3], 4, 128)
            #print(conv19.shape)

            conv20 = un_conv(conv19, conv19.get_shape()[3], 4, 64, self.image_size, train)
            #print("UP",conv20.shape)
            merge20 = tf.concat([conv20, conv2], axis=-1)
            #print(merge20.shape)

            conv21 = conv_layer(merge20, merge20.get_shape()[3], 4, 64)
            #print(conv21.shape)
            conv22 = conv_layer(conv21, conv21.get_shape()[3], 4, 64)
            #print(conv22.shape)
            conv23 = conv_layer(conv22, conv22.get_shape()[3], 1, input.get_shape()[3])
            #print("OUTPUT",conv23.shape)
            output=conv23
                # set reuse=True for next call
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return output