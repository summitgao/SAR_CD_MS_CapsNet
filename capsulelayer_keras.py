import keras.backend as K
from keras import initializers, layers, regularizers
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Lambda
from keras.layers import Multiply, multiply
from keras.layers.merge import add
import tensorflow as tf
import tflearn
import numpy as np
from SKNet import SKConv


class Length(layers.Layer):
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


class Mask(layers.Layer):

    def call(self, inputs, **kwargs):
        # use true label to select target capsule, shape=[batch_size, num_capsule]
        if type(inputs) is list:  # true label is provided with shape = [batch_size, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of vectors of capsules
            x = inputs
            # Enlarge the range of values in x to make max(new_x)=1 and others < 0
            x = (x - K.max(x, 1, True)) / K.epsilon() + 1
            mask = K.clip(x, 0, 1)  # the max value in x clipped to 1 and other to 0

        # masked inputs, shape = [batch_size, dim_vector]
        inputs_masked = K.batch_dot(inputs, mask, [1, 1])
        return inputs_masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][-1]])
        else:
            return tuple([None, input_shape[-1]])


def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keep_dims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + K.epsilon())
    return tf.multiply(scale, vectors)


class Conv_Capsule(layers.Layer):

    def __init__(self, kernel_shape, strides, dim_vector,
                 num_routing=3, batchsize=50, name='conv_caps',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer='l2',
                 **kwargs):
        super(Conv_Capsule, self).__init__(**kwargs)
        self.kernel_shape = kernel_shape
        self.strides = strides
        self.dim_vector = dim_vector
        self.num_routing = num_routing
        self.batchsize = batchsize
        self.name = name
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        assert len(input_shape) >= 5, "The input Tensor should have shape=[None, height, width, input_num_capsule, input_dim_vector]"
        self.caps_num_i = self.kernel_shape[0] * self.kernel_shape[1] * self.kernel_shape[2]
        self.output_cap_channel = self.kernel_shape[3]
        self.input_dim_vector = input_shape[4]

        # Transform matrix
        self.W = self.add_weight(
            shape=[1, self.caps_num_i, self.output_cap_channel, self.input_dim_vector, self.dim_vector],
            initializer=tf.truncated_normal_initializer(mean=0., stddev=1.),
            # regularizer=regularizers.l2(5e-4),
            name=self.name+'W')

        # Coupling coefficient.
        self.bias = self.add_weight(shape=[1, 1, 1, self.caps_num_i, self.output_cap_channel, 1, 1],
                                    initializer=self.bias_initializer,
                                    name=self.name+'bias',
                                    trainable=False)
        self.built = True

    def call(self, inputs, training=None):
        inputs_poses = inputs

        i_pose_dim = inputs_poses.shape[-1].value

        inputs_poses = self.kernel_tile(inputs_poses)

        spatial_x = inputs_poses.shape[1].value
        spatial_y = inputs_poses.shape[2].value

        inputs_poses = K.reshape(inputs_poses, shape=[-1, self.kernel_shape[0] * self.kernel_shape[1] * self.kernel_shape[2],
                                                      i_pose_dim])

        votes = self.mat_transform(inputs_poses, size=self.batchsize * spatial_x * spatial_y)

        votes_shape = votes.shape
        votes = K.reshape(votes, shape=[-1, spatial_x, spatial_y, votes_shape[-3].value,
                                        votes_shape[-2].value, votes_shape[-1].value])

        poses = self.capsules_dynamic_routing(votes)

        return poses

    def compute_output_shape(self, input_shape):
        h = (input_shape[1] - self.kernel_shape[0]) // self.strides[1] + 1
        w = (input_shape[2] - self.kernel_shape[1]) // self.strides[2] + 1
        return tuple([self.batchsize, h, w, self.kernel_shape[-1], self.dim_vector])

    def kernel_tile(self, input):

        input_shape = input.shape
        input = tf.reshape(input, shape=[-1, input_shape[1].value, input_shape[2].value,
                                         input_shape[3].value * input_shape[4].value])

        input_shape = input.shape
        tile_filter = np.zeros(shape=[self.kernel_shape[0], self.kernel_shape[1], input_shape[3].value,
                                      self.kernel_shape[0] * self.kernel_shape[1]], dtype=np.float32)
        for i in range(self.kernel_shape[0]):
            for j in range(self.kernel_shape[1]):
                tile_filter[i, j, :, i * self.kernel_shape[0] + j] = 1.0

        tile_filter_op = K.constant(tile_filter, dtype=tf.float32)

        output = tf.nn.depthwise_conv2d(input, tile_filter_op, strides=self.strides, padding='VALID')
        output_shape = output.shape
        output = K.reshape(output, shape=[-1, output_shape[1].value, output_shape[2].value, input_shape[3].value,
                                          self.kernel_shape[0] * self.kernel_shape[1]])
        output = tf.transpose(output, perm=[0, 1, 2, 4, 3])

        return output

    def mat_transform(self, input, size):

        shape = input.shape
        caps_num_i = shape[1].value  
        output = K.reshape(input, shape=[-1, caps_num_i, 1, 1, shape[2].value])

        self.W = K.tile(self.W, [size, 1, 1, 1, 1])

        output = K.tile(output, [1, 1, self.kernel_shape[-1], 1, 1])

        votes = tf.matmul(output, self.W)

        votes = K.reshape(votes, [size, caps_num_i, self.kernel_shape[-1], self.dim_vector])

        return votes

    def capsules_dynamic_routing(self, votes):

        shape = votes.shape

        votes = K.expand_dims(votes, axis=5)

        votes_stopped = K.stop_gradient(votes)
        for i in range(self.num_routing):
            c = tf.nn.softmax(self.bias, dim=4)  # dim=4 is the num_capsule dimension

            if i == self.num_routing - 1:
                outputs = squash(tf.reduce_sum(c * votes, 3, keep_dims=True))

            else:
                outputs = squash(tf.reduce_sum(c * votes_stopped, 3, keep_dims=True))
                self.bias += tf.reduce_sum(votes * outputs, -1, keep_dims=True)

        return K.reshape(outputs, [-1, shape[1].value, shape[2].value, shape[4].value, shape[5].value])


class Class_Capsule(layers.Layer):

    def __init__(self, num_capsule, dim_vector, num_routing=3,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer='l2',
                 bias_regularizer='l2',
                 **kwargs):
        super(Class_Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_vector = dim_vector
        self.num_routing = num_routing
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

    def build(self, input_shape):
        assert len(input_shape) >= 5, "The input Tensor should have shape=[None, height, width, input_num_capsule, input_dim_vector]"
        self.input_num_capsule = input_shape[1] * input_shape[2] * input_shape[3]
        self.input_dim_vector = input_shape[4]

        # Transform matrix
        self.W = self.add_weight(
            shape=[self.input_num_capsule, self.num_capsule, self.input_dim_vector, self.dim_vector],
            initializer=self.kernel_initializer,
            name='W')

        # Coupling coefficient.
        self.bias = self.add_weight(shape=[1, self.input_num_capsule, self.num_capsule, 1, 1],
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    trainable=False)
        self.built = True

    def call(self, inputs, training=None):

        shape = inputs.shape
        inputs = K.reshape(inputs, [-1, shape[1].value * shape[2].value * shape[3].value, shape[4].value])
        inputs_expand = K.expand_dims(K.expand_dims(inputs, 2), 2)

        inputs_tiled = K.tile(inputs_expand, [1, 1, self.num_capsule, 1, 1])

        inputs_hat = tf.scan(lambda ac, x: K.batch_dot(x, self.W, [3, 2]),
                             elems=inputs_tiled,
                             initializer=K.zeros([self.input_num_capsule, self.num_capsule, 1, self.dim_vector]))

        inputs_hat_stopped = K.stop_gradient(inputs_hat)

        assert self.num_routing > 0, 'The num_routing should be > 0.'
        for i in range(self.num_routing):
            c = tf.nn.softmax(self.bias, dim=2)

            if i == self.num_routing - 1:
                outputs = squash(K.sum(c * inputs_hat, 1, keepdims=True))

            else:
                outputs = squash(K.sum(c * inputs_hat_stopped, 1, keepdims=True))
                self.bias += K.sum(inputs_hat * outputs, -1, keepdims=True)
        return K.reshape(outputs, [-1, self.num_capsule, self.dim_vector])

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_vector])


def PrimaryCap1(inputs, dim_vector, n_channels, kernel_size, strides, padding):

    output = layers.Conv2D(filters=dim_vector * n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                     name='primarycap_conv2d')(inputs)
					   
    output = layers.BatchNormalization(momentum=0.9, name='primarycap_bn')(output)
    output = layers.Activation('relu', name='primarycap_relu')(output)
    shape = np.shape(output)

    outputs = layers.Reshape(target_shape=[shape[1].value, shape[2].value, n_channels, dim_vector], name='primarycap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)
	
def PrimaryCap2(inputs, dim_vector, n_channels, kernel_size, strides, padding):

    output = layers.Conv2D(filters=dim_vector * n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                       name='primarycap_conv2d_1')(inputs)					   
    output = layers.BatchNormalization(momentum=0.9, name='primarycap_bn_1')(output)
    output = layers.Activation('relu', name='primarycap_relu_1')(output)
    shape = np.shape(output)

    outputs = layers.Reshape(target_shape=[shape[1].value, shape[2].value, n_channels, dim_vector], name='primarycap_reshape_1')(output)
    return layers.Lambda(squash, name='primarycap_squash_1')(outputs)
	
def ecanet_layer(x, out_dim):
    '''
    ECA module performs inter-channel weighting.
    '''	
    k_size = 3
    squeeze = layers.GlobalAveragePooling2D()(x)
    shape = np.shape(squeeze)
	
    squeeze = layers.Reshape(target_shape=[shape[1].value, 1])(squeeze) 
    excitation = layers.Conv1D(filters=1, kernel_size= k_size, strides=1, padding='same',)(squeeze)
    excitation = layers.Activation('sigmoid')(excitation)
	
    excitation = layers.Reshape((1,1,out_dim))(excitation)
        
    #scale = tf.multiply(x,excitation)
    scale = Lambda(lambda x: x[0] * x[1])([x, excitation])
    scale = layers.BatchNormalization(momentum=0.9)(scale)    
    return scale
def AFC_layer(x):
	
    conv1 = layers.Conv2D(filters=8, kernel_size=3, strides=1, padding='same', name='conv1')(x)
    conv1 = layers.BatchNormalization(momentum=0.9, name='bn1')(conv1)
    conv1 = layers.Activation('relu', name='conv1_relu')(conv1)

    conv2 = layers.Conv2D(filters=16, kernel_size=3,  dilation_rate=2,strides=1, padding='same', name='conv2')(conv1)
    conv2 = layers.BatchNormalization(momentum=0.9, name='bn2')(conv2)
    conv2 = layers.Activation('relu', name='conv2_relu')(conv2)
	
    conv3 = layers.Conv2D(filters=32, kernel_size=3, dilation_rate=3, strides=1, padding='same', name='conv3')(conv2)
    conv3 = layers.BatchNormalization(momentum=0.9, name='bn3')(conv3)
    
    
    conv1_d = layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', name='conv1_1')(conv1)
    conv2_d = layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', name='conv2_1')(conv2)
	
    conv1_d = ecanet_layer(conv1_d, out_dim = 32)
    conv2_d = ecanet_layer(conv2_d, out_dim = 32)
    conv3_d = ecanet_layer(conv3, out_dim = 32)
	
    conv_fuse = add([conv1_d, conv2_d, conv3_d])
    conv_afc = layers.Activation('relu', name='conv3_relu')(conv_fuse)
	
    return conv_afc