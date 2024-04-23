import tensorflow as tf 
from keras.models import *
from keras.layers import *
from keras.layers import Input
from model_components import MB  # model_components 모듈에서 MB 함수를 임포트
from conv_gru_layer import ConvGRU2D

def up_conv(inputs, size, filters):
    """
Applies a transposed convolution (also known as deconvolution) layer to the input tensor.
This layer can be used for upsampling features.

Args:
    inputs (tensor): Input tensor.
    size (int): Kernel size for the convolution.
    filters (int): Number of filters in the convolution.

Returns:
    tensor: Output tensor after applying transposed convolution.
"""
    out = Conv2DTranspose(filters//3, (size, size), strides = (2, 2), padding = 'same')(inputs)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out

def build_feature_generator():
    """
Constructs the feature generator network which follows an encoder-decoder architecture.
The network uses multiple convolutional and pooling layers to downsample and then upsample the input.

Returns:
    Model: A Keras Model object representing the feature generator.
"""
    input_layer = Input(shape=(132, 132, 1)) #changing this shape
    n1 = 32
    filters = [n1*2]

    e1 = MB(input_layer, filters[0]) # N
    e2 = MaxPooling2D(strides=2)(e1) # 1 + channels[0]//3 + channels[0]//3
    
    e2 = MB(e2, filters[0]) # N to N # N to N
    e3 = MaxPooling2D(strides=2)(e2) #4

    e3 = MB(e3, filters[0]) # N to N # N to N
    e4 = MaxPooling2D(strides=2)(e3) #4
    
    e5 = MB(e4, filters[0]) #encoder

    d4 = Conv2DTranspose(filters[0], (3,3), strides = (2, 2), padding = 'valid')(e5)
    d4 = BatchNormalization()(d4)
    d4 = Activation('relu')(d4)   
    
    d4 = Concatenate()([e3, d4])
    d4 = MB(d4, filters[0])
    
    d3 = up_conv(d4,4, filters[0])
    d3 = Concatenate()([e2, d3])
    d3 = MB(d3, filters[0])
    
    d2 = up_conv(d3,4, filters[0])
    d2 = Concatenate()([e1, d2])
    d2 = MB(d2, filters[0])

    output_for_training = Conv2D(3, (1, 1), padding='same', activation='relu')(d2)
    return input_layer, output_for_training

def spatiotemporalEncoder(input_layer):
    """
Constructs a spatiotemporal encoder using multiple convolutional and pooling layers.
This encoder is designed to process and compress input features for further processing.

Args:
    input_layer (tensor): Input tensor to the encoder.

Returns:
    tensor: Compressed feature tensor.
"""
    n1 = 32
    filters = [n1*2]

    e1 = MB(input_layer, filters[0])
    e2 = MaxPooling2D(strides=2)(e1)
    e2 = MB(e2, filters[0])
    e3 = MaxPooling2D(strides=2)(e2)
    e3 = MB(e3, filters[0])
    e4 = MaxPooling2D(strides=2)(e3)
    e5 = MB(e4, filters[0])

    return e5

def build_convgru_chain(input_tensor):
    """
Constructs a sequence of ConvLSTM2D layers for processing temporal sequences.
This function allows each layer to either return the full sequence or only the last output.

Args:
    input_tensor (tensor): Input tensor to the ConvLSTM2D chain.

Returns:
    list of tensors: Outputs from each ConvLSTM2D layer.
"""
    num_layers = 47
    x = input_tensor
    outputs = []
    outputs = []
    for i in range(num_layers):
        if i == num_layers - 1:
            # For the last layer, do not return sequences
            conv_gru_layer = ConvGRU2D(64, (3, 3), padding='same')
            x, _ = conv_gru_layer(x)  # Assuming ConvGRU2D returns output and state
            outputs.append(Lambda(lambda x: x[:, -1, :, :, :])(x))
        else:
            # For other layers, return the full sequence and collect the last output
            conv_gru_layer = ConvGRU2D(64, (3, 3), padding='same')
            x, _ = conv_gru_layer(x)  # Assuming ConvGRU2D returns output and state
            outputs.append(Lambda(lambda x: x[:, -1, :, :, :])(x))


    return outputs

def build_multi_kernel_decoder(input_tensor, prev_d2=None):
    """
Constructs a multi-kernel decoder that can optionally concatenate its output with the output from a previous decoder.

Args:
    input_tensor (tensor): Input tensor to the decoder.
    prev_d2 (tensor, optional): Output from a previous decoder to concatenate with.

Returns:
    tuple: Decoded tensor and the final convolution output.
"""
    # prev_d2 is the d2 from the previous decoder
    input_layer = input_tensor
    d4 = Conv2DTranspose(64, (3,3), strides=(2, 2), padding='valid')(input_layer)
    d4 = BatchNormalization()(d4)
    d4 = Activation('relu')(d4)
    d4 = MB(d4, 64)  # Assuming MB is a function returning modified tensor
    
    d3 = up_conv(d4, 4, 64)
    d3 = MB(d3, 64)
    
    d2 = up_conv(d3, 4, 64)
    d2 = MB(d2, 64)
    if prev_d2 is not None:
        d2 = Concatenate()([prev_d2, d2])
    
    final_conv = Conv2D(1, (1, 1), padding='same', activation='relu')(d2)
    return d2, final_conv

def FeatureConnection(input_tensor):
    """
Generates importance scores using a simple convolution layer. This could be replaced with a more complex mechanism.

Args:
    input_tensor (tensor): Input tensor for generating scores.

Returns:
    tensor: Importance scores.
"""
    importance_scores = Conv2D(1, (1, 1), activation='softmax', padding='same')(input_layer)
    return importance_scores


def build_stgnet():
    """
Builds the entire STGNet model by combining the feature generator, spatiotemporal encoder, ConvGRU chain, and decoders.

Returns:
    Model: Compiled Keras model ready for training.
"""
    input_img, feature_gen_output = build_feature_generator()
    encoded_features = spatiotemporalEncoder(feature_gen_output)
    convgru_outputs = build_convgru_chain(tf.expand_dims(encoded_features, 1))

    prev_d2 = None
    decoded_finals = []
    for output in convgru_outputs:
        d2, final_output = build_multi_kernel_decoder(output, prev_d2)
        prev_d2 = d2
        decoded_finals.append(final_output)

    # hierachical multi-task learning
    model = Model(inputs=input_img, outputs=[feature_gen_output] + decoded_finals)
    # loss weight for multi-task learning
    loss_weights = [1.0] + [0.5] * len(decoded_finals)
    
    model.compile(optimizer='adam', loss=['mse'] + ['mse'] * len(decoded_finals),
                  loss_weights=loss_weights,metrics=['accuracy'])
    return model

stgnet = build_stgnet()
stgnet.summary()

#%%
#Data preprocess for hierachical learning
target_data = [target_feature_gen_output] + target_decoded_finals

# training
history = stgnet.fit(x=input_data, y=target_data, batch_size=200, epochs=2000, validation_split=0.2)