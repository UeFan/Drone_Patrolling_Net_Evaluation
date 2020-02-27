import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, concatenate
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.regularizers import l2
from keras.layers.merge import add
from keras import regularizers
from common_flags import FLAGS
import keras_contrib.layers.advanced_activations.srelu as aa
c = FLAGS.output_dimension  # The number of outputs we want to predict
m = FLAGS.distribution_num  # The number of distributions we want to use in the mixture
ROW_AXIS = 1
COL_AXIS = 2
CHANNEL_AXIS = 3  # axis = 3, because it's channel last
# Note: The output size will be (c + 2) * m
from keras import backend as K
def elu_plus_one_plus_epsilon(x):
    """ELU activation with a very small addition to help prevent
    NaN in loss."""
    return K.elu(x) + 1 + K.epsilon()

def conv_net(img_width, img_height, img_channels, output_dim):

    # Input
    img_input = Input(shape=(img_height, img_width, img_channels))
    x1 = Conv2D(24, (5, 5), strides=[2, 2], padding='same')(img_input)
    x1 = Activation('relu')(x1)
    x1 = Dropout(0.5)(x1)
    x1 = keras.layers.normalization.BatchNormalization()(x1)

    x2 = Conv2D(48, (5, 5), strides=[2, 2], padding='same')(x1)
    x2 = Activation('relu')(x2)
    x2 = Dropout(0.5)(x2)
    x2 = keras.layers.normalization.BatchNormalization()(x2)

    x3 = Conv2D(48, (5, 5), strides=[2, 2], padding='same')(x2)
    x3 = Activation('relu')(x3)
    x3 = Dropout(0.5)(x3)
    x3 = keras.layers.normalization.BatchNormalization()(x3)

    x4 = Conv2D(48, (5, 5), strides=[2, 2], padding='same')(x3)
    x4 = Activation('relu')(x4)
    x4 = Dropout(0.5)(x4)
    x5 = keras.layers.normalization.BatchNormalization()(x4)

    x6 = Flatten()(x5)

    x_coll = Dense(500, activation='relu')(x6)
    x_coll = keras.layers.normalization.BatchNormalization()(x_coll)
    x_coll = Dropout(0.5)(x_coll)
    coll = Dense(output_dim, name='trans_output')(x_coll)
    # coll = Activation('sigmoid')(coll)

    dense1_1 = Dense(1000, activation='relu')(x6)
    dense1_1 = keras.layers.normalization.BatchNormalization()(dense1_1)
    dense1_1 = Dropout(0.5)(dense1_1)

    dense2_1 = Dense(100, activation='relu')(dense1_1)
    dense2_1 = keras.layers.normalization.BatchNormalization()(dense2_1)
    dense2_1 = Dropout(0.5)(dense2_1)
    FC_mus = Dense(c * m, activation='tanh')(dense2_1)
    # FC_sigmas = Dense(m, activation=elu_plus_one_plus_epsilon)(dense1_1)  # Keras.exp, W_regularizer=l2(1e-3)
    FC_alphas = Dense(m)(dense2_1)
    outputs = concatenate([FC_mus, FC_alphas], name='direct_output')

    # outputs = Dense((c+1)*m)(dense2_1)

    model = Model(inputs=[img_input], outputs=[outputs, coll])
    # model = Model(inputs=[img_input], outputs=[outputs])

    # Define steering-collision model
    # model = Model(inputs=[img_input], outputs=[steer, coll])
    print(model.summary())

    return model



def resnet8_MDN(img_width, img_height, img_channels, output_dim):
    """
    Define model architecture.
    
    # Arguments
       img_width: Target image widht.
       img_height: Target image height.
       img_channels: Target image channels.
       output_dim: Dimension of model output.
       
    # Returns
       model: A Model instance.
    """

    # Input
    img_input = Input(shape=(img_height, img_width, img_channels))

    x1 = Conv2D(32, (5, 5), strides=[2,2], padding='same')(img_input)
    x1 = MaxPooling2D(pool_size=(3, 3), strides=[2,2])(x1)

    # First residual block
    x2 = keras.layers.normalization.BatchNormalization()(x1)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x2)

    x2 = keras.layers.normalization.BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x2)

    x1 = Conv2D(32, (1, 1), strides=[2,2], padding='same')(x1)
    x3 = add([x1, x2])



    # Second residual block
    x4 = keras.layers.normalization.BatchNormalization()(x3)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x4)

    x4 = keras.layers.normalization.BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x4)

    x3 = Conv2D(64, (1, 1), strides=[2,2], padding='same')(x3)
    x5 = add([x3, x4])
    x3_out = Flatten()(x5)
    x3_out = Activation('relu')(x3_out)
    x3_out = Dropout(0.5)(x3_out)
    #
    #
    # # Third residual block
    # x6 = keras.layers.normalization.BatchNormalization()(x5)
    # x6 = Activation('relu')(x6)
    # x6 = Conv2D(128, (3, 3), strides=[2,2], padding='same',
    #             kernel_initializer="he_normal",
    #             kernel_regularizer=regularizers.l2(1e-4))(x6)
    #
    # x6 = keras.layers.normalization.BatchNormalization()(x6)
    # x6 = Activation('relu')(x6)
    # x6 = Conv2D(128, (3, 3), padding='same',
    #             kernel_initializer="he_normal",
    #             kernel_regularizer=regularizers.l2(1e-4))(x6)
    #
    # x5 = Conv2D(128, (1, 1), strides=[2,2], padding='same')(x5)
    # x7 = add([x5, x6])
    #
    # x = Flatten()(x7)
    # x = Activation('relu')(x)
    # x = Dropout(0.5)(x)

    # Steering channel
    #steer = Dense(output_dim)(x)

    # Collision channel
    x_coll = Dense(500, activation='relu')(x3_out)
    x_coll = keras.layers.normalization.BatchNormalization()(x_coll)
    x_coll = Dropout(0.5)(x_coll)
    coll = Dense(output_dim, name='trans_output')(x_coll)
    # coll = Activation('sigmoid')(coll)

    dense1_1 = Dense(500, activation='relu')(x3_out)
    dense1_1 = keras.layers.normalization.BatchNormalization()(dense1_1)
    dense1_1 = Dropout(0.2)(dense1_1)

    dense2_1 = Dense(100, activation='relu')(dense1_1)
    dense2_1 = keras.layers.normalization.BatchNormalization()(dense2_1)
    dense2_1 = Dropout(0.2)(dense2_1)
    FC_mus = Dense(c * m, activation='tanh')(dense2_1)
    # FC_sigmas = Dense(m, activation=elu_plus_one_plus_epsilon)(dense1_1)  # Keras.exp, W_regularizer=l2(1e-3)
    FC_alphas = Dense(m)(dense2_1)
    outputs = concatenate([FC_mus, FC_alphas], name='direct_output')

    # outputs = Dense((c+1)*m)(dense2_1)

    model = Model(inputs=[img_input], outputs=[outputs, coll])
    # model = Model(inputs=[img_input], outputs=[outputs])

    # Define steering-collision model
    # model = Model(inputs=[img_input], outputs=[steer, coll])
    print(model.summary())

    return model

def _block_name_base(stage, block):
    """Get the convolution name base and batch normalization name base defined by
    stage and block.
    If there are less than 26 blocks they will be labeled 'a', 'b', 'c' to match the
    paper and keras and beyond 26 blocks they will simply be numbered.
    """
    if block < 27:
        block = '%c' % (block + 97)  # 97 is the ascii number for lowercase 'a'
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    return conv_name_base, bn_name_base

def _shortcut(input_feature, residual, conv_name_base=None, bn_name_base=None):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input_feature)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input_feature
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        print('reshaping via a convolution...')
        if conv_name_base is not None:
            conv_name_base = conv_name_base + '1'
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001),
                          name=conv_name_base)(input_feature)
        if bn_name_base is not None:
            bn_name_base = bn_name_base + '1'
        shortcut = keras.layers.normalization.BatchNormalization(axis=CHANNEL_AXIS,
                                      name=bn_name_base)(shortcut)

    return add([shortcut, residual])

def _bn_relu(x, bn_name=None, relu_name=None, to = False):
    """Helper to build a BN -> relu block
    """
    norm = keras.layers.normalization.BatchNormalization(axis=CHANNEL_AXIS, name=bn_name)(x)
    if not to:
        return Activation("relu", name=relu_name)(norm)
    else:
        return aa.SReLU(
            t_left_initializer=keras.initializers.constant(-1),
            t_right_initializer=keras.initializers.constant(-1),
            trainable=False
        )(norm)

def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv residual unit with full pre-activation
    function. This is the ResNet v2 scheme proposed in
    http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    dilation_rate = conv_params.setdefault("dilation_rate", (1, 1))
    conv_name = conv_params.setdefault("conv_name", None)
    bn_name = conv_params.setdefault("bn_name", None)
    relu_name = conv_params.setdefault("relu_name", None)
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(x):
        activation = _bn_relu(x, bn_name=bn_name, relu_name=relu_name)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      dilation_rate=dilation_rate,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer,
                      name=conv_name)(activation)

    return f

def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu residual unit activation function.
       This is the original ResNet v1 scheme in https://arxiv.org/abs/1512.03385
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    dilation_rate = conv_params.setdefault("dilation_rate", (1, 1))
    conv_name = conv_params.setdefault("conv_name", None)
    bn_name = conv_params.setdefault("bn_name", None)
    relu_name = conv_params.setdefault("relu_name", None)
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(x):
        x = Conv2D(filters=filters, kernel_size=kernel_size,
                   strides=strides, padding=padding,
                   dilation_rate=dilation_rate,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name=conv_name)(x)
        return _bn_relu(x, bn_name=bn_name, relu_name=relu_name,to=True)

    return f

def _residual_block(block_function, filters, blocks, stage,
                    transition_strides=None,
                    dilation_rates=None, is_first_layer=False, dropout=None,
                    residual_unit=_bn_relu_conv):
    """Builds a residual block with repeating bottleneck blocks.
       stage: integer, current stage label, used for generating layer names
       blocks: number of blocks 'a','b'..., current block label, used for generating
            layer names
       transition_strides: a list of tuples for the strides of each transition
       transition_dilation_rates: a list of tuples for the dilation rate of each
            transition
    """
    if transition_strides is None:
        transition_strides = [(1, 1)] * blocks
    if dilation_rates is None:
        dilation_rates = [1] * blocks

    def f(x):
        for i in range(blocks):
            is_first_block = is_first_layer and i == 0
            x = block_function(filters=filters, stage=stage, block=i,
                               transition_strides=transition_strides[i],
                               dilation_rate=dilation_rates[i],
                               is_first_block_of_first_layer=is_first_block,
                               dropout=dropout,
                               residual_unit=residual_unit)(x)
        return x

    return f

def basic_block(filters, stage, block, transition_strides=(1, 1),
                dilation_rate=(1, 1), is_first_block_of_first_layer=False, dropout=None,
                residual_unit=_bn_relu_conv):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input_features):
        conv_name_base, bn_name_base = _block_name_base(stage, block)
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            x = Conv2D(filters=filters, kernel_size=(3, 3),
                       strides=transition_strides,
                       dilation_rate=dilation_rate,
                       padding="same",
                       kernel_initializer="he_normal",
                       kernel_regularizer=l2(1e-4),
                       name=conv_name_base + '2a')(input_features)
        else:
            x = residual_unit(filters=filters, kernel_size=(3, 3),
                              strides=transition_strides,
                              dilation_rate=dilation_rate,
                              conv_name_base=conv_name_base + '2a',
                              bn_name_base=bn_name_base + '2a')(input_features)

        if dropout is not None:
            x = Dropout(dropout)(x)

        x = residual_unit(filters=filters, kernel_size=(3, 3),
                          conv_name_base=conv_name_base + '2b',
                          bn_name_base=bn_name_base + '2b')(x)

        return _shortcut(input_features, x)

    return f


def resnet18_MDN(img_width, img_height, img_channels, output_dim):
    """
    Define model architecture.

    # Arguments
       img_width: Target image widht.
       img_height: Target image height.
       img_channels: Target image channels.
       output_dim: Dimension of model output.

    # Returns
       model: A Model instance.
    """

    # Input
    initial_filters = 64
    initial_strides = (2, 2)
    initial_kernel_size = (7, 7)

    img_input = Input(shape=(img_height, img_width, img_channels))
    x = _conv_bn_relu(filters=initial_filters, kernel_size=initial_kernel_size,
                      strides=initial_strides)(img_input)
    x = MaxPooling2D(pool_size=(3, 3), strides=initial_strides, padding="same")(x)

    block = x
    filters = 64
    repetitions = [2,2,2]
    transition_dilation_rate = (1,1)
    block_fn = basic_block
    residual_unit = _bn_relu_conv   # resNet v2
    dropout = 0.5
    for i, r in enumerate(repetitions):
        transition_dilation_rates = [transition_dilation_rate] * r
        transition_strides = [(1, 1)] * r
        if transition_dilation_rate == (1, 1):
            transition_strides[0] = (2, 2)
        block = _residual_block(block_fn, filters=filters,
                                stage=i, blocks=r,
                                is_first_layer=i==0,
                                dropout=dropout,
                                transition_strides=transition_strides,
                                residual_unit=residual_unit)(block)
        filters *= 2

    # Last activation
    x = _bn_relu(block)
    x = GlobalAveragePooling2D()(x)


    x_out = Flatten()(x)


    # Collision channel
    x_coll = Dense(500, activation='relu')(x_out)
    x_coll = keras.layers.normalization.BatchNormalization()(x_coll)
    x_coll = Dropout(0.2)(x_coll)
    trans = Dense(output_dim, name='trans_output')(x_coll)
    # coll = Activation('sigmoid')(coll)

    # dense1_1 = Dense(500, activation='relu')(x3_out)
    # dense1_1 = keras.layers.normalization.BatchNormalization()(dense1_1)
    # dense1_1 = Dropout(0.5)(dense1_1)

    dense2_1 = Dense(500, activation='relu')(x_out)
    dense2_1 = keras.layers.normalization.BatchNormalization()(dense2_1)
    dense2_1 = Dropout(0.2)(dense2_1)
    FC_mus = Dense(c * m, activation='tanh')(dense2_1)
    # FC_sigmas = Dense(m, activation=elu_plus_one_plus_epsilon)(dense1_1)  # Keras.exp, W_regularizer=l2(1e-3)
    FC_alphas = Dense(m)(dense2_1)
    outputs = concatenate([FC_mus, FC_alphas], name='direct_output')

    # outputs = Dense((c+1)*m)(dense2_1)

    model = Model(inputs=[img_input], outputs=[outputs, trans])
    # model = Model(inputs=[img_input], outputs=[outputs])

    # Define steering-collision model
    # model = Model(inputs=[img_input], outputs=[steer, coll])
    print(model.summary())

    return model

def conv_MDN_resnet_trans(img_width, img_height, img_channels, output_dim):
    """
    Define model architecture.

    # Arguments
       img_width: Target image widht.
       img_height: Target image height.
       img_channels: Target image channels.
       output_dim: Dimension of model output.

    # Returns
       model: A Model instance.
    """

    # Input
    initial_filters = 64
    initial_strides = (2, 2)
    initial_kernel_size = (7, 7)

    img_input = Input(shape=(img_height, img_width, img_channels))
    x = _conv_bn_relu(filters=initial_filters, kernel_size=initial_kernel_size,
                      strides=initial_strides)(img_input)
    x = MaxPooling2D(pool_size=(3, 3), strides=initial_strides, padding="same")(x)

    block = x
    filters = 64
    repetitions = [2,2,2]
    transition_dilation_rate = (1,1)
    block_fn = basic_block
    residual_unit = _bn_relu_conv   # resNet v2
    dropout = 0.5
    for i, r in enumerate(repetitions):
        transition_dilation_rates = [transition_dilation_rate] * r
        transition_strides = [(1, 1)] * r
        if transition_dilation_rate == (1, 1):
            transition_strides[0] = (2, 2)
        block = _residual_block(block_fn, filters=filters,
                                stage=i, blocks=r,
                                is_first_layer=(i == 0),
                                dropout=dropout,
                                transition_strides=transition_strides,
                                residual_unit=residual_unit)(block)
        filters *= 2

    # Last activation
    x = _bn_relu(block)
    #x = GlobalAveragePooling2D()(x)


    x_out = Flatten()(x)

    # dense1_1 = Dense(500, activation='relu')(x3_out)
    # dense1_1 = keras.layers.normalization.BatchNormalization()(dense1_1)
    # dense1_1 = Dropout(0.5)(dense1_1)

    dense2_1 = Dense(500, activation='relu')(x_out)
    dense2_1 = keras.layers.normalization.BatchNormalization()(dense2_1)
    dense2_1 = Dropout(0.2)(dense2_1)
    FC_mus = Dense(c * m, activation='tanh')(dense2_1)
    # FC_sigmas = Dense(m, activation=elu_plus_one_plus_epsilon)(dense1_1)  # Keras.exp, W_regularizer=l2(1e-3)
    FC_alphas = Dense(m)(dense2_1)
    outputs = concatenate([FC_mus, FC_alphas], name='direct_output')

    # outputs = Dense((c+1)*m)(dense2_1)


    #translation channel
    x1 = Conv2D(24, (5, 5), strides=[2, 2], padding='same')(img_input)
    x1 = Activation('relu')(x1)
    x1 = Dropout(0.5)(x1)
    x1 = keras.layers.normalization.BatchNormalization()(x1)

    x2 = Conv2D(48, (5, 5), strides=[2, 2], padding='same')(x1)
    x2 = Activation('relu')(x2)
    x2 = Dropout(0.5)(x2)
    x2 = keras.layers.normalization.BatchNormalization()(x2)

    x3 = Conv2D(48, (5, 5), strides=[2, 2], padding='same')(x2)
    x3 = Activation('relu')(x3)
    x3 = Dropout(0.5)(x3)
    x3 = keras.layers.normalization.BatchNormalization()(x3)

    x4 = Conv2D(48, (5, 5), strides=[2, 2], padding='same')(x3)
    x4 = Activation('relu')(x4)
    x4 = Dropout(0.5)(x4)
    x5 = keras.layers.normalization.BatchNormalization()(x4)

    x6 = Flatten()(x5)

    x_trans = Dense(500, activation='relu')(x6)
    x_trans = keras.layers.normalization.BatchNormalization()(x_trans)
    x_trans = Dropout(0.5)(x_trans)
    trans = Dense(output_dim, name='trans_output')(x_trans)

    model = Model(inputs=[img_input], outputs=[outputs, trans])
    # model = Model(inputs=[img_input], outputs=[outputs])

    # Define steering-collision model
    # model = Model(inputs=[img_input], outputs=[steer, coll])
    print(model.summary())

    return model
def s_Resnet_18(img_width, img_height, img_channels, output_dim):
    initial_filters = 64
    initial_strides = (2, 2)
    initial_kernel_size = (7, 7)

    img_input = Input(shape=(img_height, img_width, img_channels))
    x = _conv_bn_relu(filters=initial_filters, kernel_size=initial_kernel_size,
                      strides=initial_strides,padding="valid")(img_input)
    x = MaxPooling2D(pool_size=(3, 3), strides=initial_strides, padding="valid")(x)

    temp = _conv_bn_relu(filters=64, kernel_size=(3,3),strides=(1,1),padding="same")(x)
    temp = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding="same")(temp)
    x = _shortcut(x,temp)
    x = _bn_relu(x,to=True)
    temp = _conv_bn_relu(filters=64, kernel_size=(3,3),strides=(1,1),padding="same")(x)
    temp = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding="same")(temp)
    x = _shortcut(x,temp)
    x = _bn_relu(x,to=True)

    temp = _conv_bn_relu(filters=64, kernel_size=(3,3),strides=(1,1),padding="same")(x)
    temp = Conv2D(filters=128,kernel_size=(3,3),strides=(2,2),padding="same")(temp)
    x = _shortcut(x,temp)
    x = _bn_relu(x,to=True)
    temp = _conv_bn_relu(filters=128, kernel_size=(3,3),strides=(1,1),padding="same")(x)
    temp = Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding="same")(temp)
    x = _shortcut(x,temp)
    x = _bn_relu(x,to=True)

    temp = _conv_bn_relu(filters=128, kernel_size=(3,3),strides=(1,1),padding="same")(x)
    temp = Conv2D(filters=256,kernel_size=(3,3),strides=(2,2),padding="same")(temp)
    x = _shortcut(x,temp)
    x = _bn_relu(x,to=True)
    temp = _conv_bn_relu(filters=256, kernel_size=(3,3),strides=(1,1),padding="same")(x)
    temp = Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding="same")(temp)
    x = _shortcut(x,temp)
    x = _bn_relu(x,to=True)

    temp = _conv_bn_relu(filters=256, kernel_size=(3,3),strides=(1,1),padding="same")(x)
    temp = Conv2D(filters=512,kernel_size=(3,3),strides=(2,2),padding="same")(temp)
    x = _shortcut(x,temp)
    x = _bn_relu(x,to=True)
    temp = _conv_bn_relu(filters=512, kernel_size=(3,3),strides=(1,1),padding="same")(x)
    temp = Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding="same")(temp)
    x = _shortcut(x,temp)
    x = _bn_relu(x,to=True)
    x = GlobalAveragePooling2D()(x)

    orientation =Dense(3, name='orien_output', activation="softmax")(x)

    trans = Dense(3, name='trans_output', activation="softmax")(x)
    model = Model(inputs=[img_input], outputs=[orientation,trans])
    print(model.summary())
    return model


def resnet8(img_width, img_height, img_channels, output_dim):
    """
    Define model architecture.

    # Arguments
       img_width: Target image widht.
       img_height: Target image height.
       img_channels: Target image channels.
       output_dim: Dimension of model output.

    # Returns
       model: A Model instance.
    """

    # Input
    img_input = Input(shape=(img_height, img_width, img_channels))

    x1 = Conv2D(32, (5, 5), strides=[2, 2], padding='same')(img_input)
    x1 = MaxPooling2D(pool_size=(3, 3), strides=[2, 2])(x1)

    # First residual block
    x2 = keras.layers.normalization.BatchNormalization()(x1)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), strides=[2, 2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x2)

    x2 = keras.layers.normalization.BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x2)

    x1 = Conv2D(32, (1, 1), strides=[2, 2], padding='same')(x1)
    x3 = add([x1, x2])

    # Second residual block
    x4 = keras.layers.normalization.BatchNormalization()(x3)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), strides=[2, 2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x4)

    x4 = keras.layers.normalization.BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x4)

    x3 = Conv2D(64, (1, 1), strides=[2, 2], padding='same')(x3)
    x5 = add([x3, x4])

    # Third residual block
    x6 = keras.layers.normalization.BatchNormalization()(x5)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(128, (3, 3), strides=[2, 2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x6)

    x6 = keras.layers.normalization.BatchNormalization()(x6)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(128, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x6)

    x5 = Conv2D(128, (1, 1), strides=[2, 2], padding='same')(x5)
    x7 = add([x5, x6])

    x = Flatten()(x7)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    # Steering channel
    steer = Dense(output_dim)(x)

    # Collision channel
    coll = Dense(output_dim)(x)
    coll = Activation('sigmoid')(coll)

    # Define steering-collision model
    model = Model(inputs=[img_input], outputs=[steer, coll])
    print(model.summary())

    return model