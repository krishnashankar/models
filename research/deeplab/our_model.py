import tensorflow as tf
import numpy as np
from deeplab.core import feature_extractor
from nets.mobilenet import mobilenet_v2
from nets import resnet_v1
from nets import resnet_v2

slim = tf.contrib.slim


def upscale_deconvolution(features,
                          output_filter_size=3,
                          scale=2,
                          weight_decay=0.0001,
                          is_training=False):
    deconvolution_kernel_size = (2 * scale - scale % 2)
    deconvolution_stride = scale
    with tf.contrib.framework.arg_scope(
                    [tf.contrib.layers.conv2d_transpose],
                    activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm,
                    weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                    weights_initializer=bilinear_initializer(scale,output_filter_size)):
        with tf.contrib.framework.arg_scope(
                [tf.contrib.layers.batch_norm], is_training=is_training):
            output = tf.contrib.layers.conv2d_transpose(features,
                                                        output_filter_size,
                                                        kernel_size=deconvolution_kernel_size,
                                                        stride=deconvolution_stride)
            return output
                          

def bilinear_initializer(scale,
                         num_filters):
    filter_size = (2 * scale - scale % 2)
    num_channels = num_filters

    #Create bilinear weights in numpy array
    bilinear_kernel = np.zeros([filter_size, filter_size], dtype=np.float32)
    scale_factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
            center = scale_factor - 1
    else:
            center = scale_factor - 0.5
    for x in range(filter_size):
        for y in range(filter_size):
            bilinear_kernel[x,y] = (1 - abs(x - center) / scale_factor) * \
                                   (1 - abs(y - center) / scale_factor)
    weights = np.zeros((filter_size, filter_size, num_channels, num_channels))
    for i in range(num_channels):
        weights[:, :, i, i] = bilinear_kernel

    #assign numpy array to constant_initalizer and pass to get_variable
    bilinear_init = tf.constant_initializer(value=weights, dtype=tf.float32)
                                        
    

def extract_features_resnet_v1_fcn(images,
                                   output_filter_size=3,
                                   weight_decay=0.0001,
                                   reuse=None,
                                   is_training=False,
                                   fine_tune_batch_norm=False):

    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        _, endpoints = resnet_v1.resnet_v1_101(images,
                                                   is_training=is_training,
                                                   output_stride=16,
                                                   global_pool=False,
                                                   reuse=reuse)

    with tf.name_scope("Concat_Blocks"):
        resnet_block_1 = endpoints['resnet_v1_101/block1']
        resnet_block_2 = endpoints['resnet_v1_101/block2']
        resnet_block_3 = endpoints['resnet_v1_101/block3']


        
        resized_block_2 = tf.image.resize_bilinear(resnet_block_2, tf.shape(resnet_block_1)[1:3],align_corners=True)
        resized_block_3 = tf.image.resize_bilinear(resnet_block_3, tf.shape(resnet_block_1)[1:3],align_corners=True)
        
        concatenated_blocks = tf.concat([resnet_block_1, resized_block_2, resized_block_3], 3)
            
            
            
    with tf.variable_scope("Blocks_Conv"):
        common_res_conv_features = slim.conv2d(concatenated_blocks,
                                               output_filter_size,
                                               kernel_size=1,
                                               activation_fn=None,
                                               weights_regularizer=slim.l2_regularizer(weight_decay),
                                               weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                               reuse=reuse)
        
        
        return tf.identity(common_res_conv_features, name='output_features')
            
   

def upsample_features(images,
                      features,
                      output_filter_size=3,
                      upsample_method='resize_bilinear',
                      is_training=False,
                      deconvolution_stride=8,
                      deconvolution_kernel_size=16,
                      reuse=None,
                      weight_decay=0.0001,
                      scope="Upsampled_Outputs"):
    
    with tf.name_scope(scope):
        if upsample_method =='resize_bilinear':
            output = tf.image.resize_bilinear(features, tf.shape(images)[1:3], align_corners=True, name='bilinear_upsampled_outputs')
            
        elif upsample_method =='resize_nearest_neighbor':
            output = tf.image.resize_nearest_neighbor(features, tf.shape(images)[1:3], align_corners=True, name='nn_upsampled_outputs')
            
        elif upsample_method =='transposed_convolution':
            with tf.variable_scope("first_deconvolution"):
                once_deconv = upscale_deconvolution(features,
                                      output_filter_size=output_filter_size,
                                      scale=2,
                                      is_training=is_training)
            with tf.variable_scope("second_deconvolution"):
                twice_deconv = upscale_deconvolution(features,
                                      output_filter_size=output_filter_size,
                                      scale=2,
                                      is_training=is_training)

            output = tf.image.resize_bilinear(twice_deconv,tf.shape(images)[1:3], name='deconv_upsampled_outputs')
                    
        elif upsample_method == 'resized_convolution':
            output = tf.image.resize_bilinear(features, tf.shape(images)[1:3], align_corners=True, name='nn_upsampled_outputs')
            with tf.variable_scope("resized_conv"):
                output = slim.conv2d(output,
                                     output_filter_size,
                                     kernel_size=4,
                                     activation_fn=None,
                                     weights_regularizer=slim.l2_regularizer(weight_decay),
                                     weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                     reuse=reuse)
                
        return output;
    
def get_predictions(images,
                    output_filter_size=3,
                    upsample_method='resize_bilinear',
                    weight_decay=0.0001,
                    reuse=None,
                    is_training=False,
                    fine_tune_batch_norm=False):
    
    features = extract_features_resnet_v1_fcn(images,
                                              output_filter_size=output_filter_size,
                                              weight_decay=weight_decay,
                                              reuse=reuse,
                                              is_training=is_training,
                                              fine_tune_batch_norm=fine_tune_batch_norm)

    
    logits = upsample_features(images,
                               features,
                               output_filter_size=output_filter_size,
                               upsample_method=upsample_method,
                               is_training=is_training,
                               weight_decay=weight_decay,
                               reuse=reuse)
    
    

    predictions = tf.argmax(logits, 3)

    return predictions
