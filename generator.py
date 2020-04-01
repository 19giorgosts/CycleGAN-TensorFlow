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

        def conv_layer(input, num_input_channels, conv_filter_size, num_filters, acti_func, padding='SAME'):
            weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
            #biases = create_biases(num_filters)
            layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 2, 2, 1], padding=padding)
            #layer += biases

            if acti_func=='relu':
                layer = tf.nn.relu(layer)
            elif acti_func=='leaky_relu':
                layer = tf.nn.leaky_relu(layer)
            return layer

        def decoder_block(input, skip_in, num_input_channels, conv_filter_size, num_filters, feature_map_size, dropout=True, padding='SAME'):
            weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_filters, num_input_channels])
            layer = tf.nn.conv2d_transpose(value=input, filter=weights,
                                           output_shape=[self.batch_size, feature_map_size, feature_map_size, num_filters],
                                           strides=[1, 2, 2, 1],
                                           padding=padding)
              # add batch normalization
            #g = BatchNormalization()(g, training=True)
            # conditionally add dropout
            #if dropout:
            #  g = Dropout(0.5)(g, training=True)
            # merge with skip connection
            layer=tf.concat(values=[layer, skip_in], axis = -1)
            # relu activation
            layer = tf.nn.relu(layer)
            return layer 


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
            layer = tf.nn.relu(layer)
            return layer

        with tf.variable_scope(self.name):        
            train=True
            # train is used for un_conv, to determine the batch size
            print(input.shape)
            conv1 = conv_layer(input, input.get_shape()[3], 4, 64, acti_func='leaky_relu')
            print(conv1.shape)
            conv2 = conv_layer(conv1, conv1.get_shape()[3], 4, 128, acti_func='leaky_relu')
            print(conv2.shape)
            conv3 = conv_layer(conv2, conv2.get_shape()[3], 4, 256, acti_func='leaky_relu')
            print(conv3.shape)
            conv4 = conv_layer(conv3, conv3.get_shape()[3], 4, 512, acti_func='leaky_relu')
            print(conv4.shape)
            conv5 = conv_layer(conv4, conv4.get_shape()[3], 4, 512, acti_func='leaky_relu')
            print(conv5.shape)
            conv6 = conv_layer(conv5, conv5.get_shape()[3], 4, 512, acti_func='leaky_relu')
            print(conv6.shape)
            conv7 = conv_layer(conv6, conv6.get_shape()[3], 4, 512, acti_func='leaky_relu')
            print(conv7.shape)
            conv8 = conv_layer(conv7, conv7.get_shape()[3], 4, 512, acti_func='relu')
            print(conv8.shape)


            d1=decoder_block(conv8, conv7, conv8.get_shape()[3], 4, 512, conv7.get_shape()[2])
            print(d1.shape)
            d2=decoder_block(d1, conv6, conv7.get_shape()[3], 4, 512, conv6.get_shape()[2])
            print(d2.shape)
            d3=decoder_block(d2, conv5, conv6.get_shape()[3], 4, 512, conv5.get_shape()[2])
            print(d3.shape)
            d4=decoder_block(d3, conv4, conv5.get_shape()[3], 4, 512, conv4.get_shape()[2])
            print(d4.shape)
            d5=decoder_block(d4, conv3, conv4.get_shape()[3], 4, 256, conv3.get_shape()[2])
            print(d5.shape)
            d6=decoder_block(d5, conv2, conv3.get_shape()[3], 4, 128, conv2.get_shape()[2])
            print(d6.shape)
            d7=decoder_block(d6, conv1, conv2.get_shape()[3], 4, 64, conv1.get_shape()[2])
            print(d7.shape)
            last = un_conv(d7, d7.get_shape()[3], 4, 3, input.get_shape()[1], train)
            print("output",last.shape)
            output = tf.nn.tanh(last)
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
            return output