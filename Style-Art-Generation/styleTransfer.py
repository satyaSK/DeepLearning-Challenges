#Using Transfer Learning 
from __future__ import print_function
import time
import numpy as  np
from PIL import Image
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
from keras import backend
from keras.models import Model 
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions

#simple euclidean distance for calculating the content loss
def content_loss(combination, content):
	euclidean = backend.sum(backend.square(combination - content))
	return euclidean

def gram_matrix(x):
	features = backend.batch_flatten(backend.permute_dimensions(x,(2,0,1)))
	gram_matrix = backend.dot(features, backend.transpose(features))
	return gram_matrix

def style_loss(combination,style,style1):
	s1= gram_matrix(style)
	s2 = gram_matrix(style1)
	c = gram_matrix(combination)
	channels = 3
	size = height * width
	return backend.sum(backend.square(s1-c) + backend.square(s2-c) )/(4. * (channels ** 2) * (size ** 2))
#-------for denoising the final combination image------
def total_variation_loss(x):
    a = backend.square(x[:, :height-1, :width-1, :] - x[:, 1:, :width-1, :])
    b = backend.square(x[:, :height-1, :width-1, :] - x[:, :height-1, 1:, :])
    return backend.sum(backend.pow(a + b, 1.25))


width, height = 244,244

content_image_path = 'C:/Users/Dell/Desktop/python/BasicCodes/DeeplearningTensorflow/MyGithub/profilepic.jpg'
content_image =  Image.open(content_image_path)
content_image = content_image.resize((width,height))
content_array = np.asarray(content_image, dtype='float32')
content_array = np.expand_dims(content_array, axis=0)
##make mean 0
content_array[:,:,:,0] -= 103.939# for R
content_array[:,:,:,1] -= 116.779# for G
content_array[:,:,:,2] -= 123.68# for B
#convert content array to BGR
content_array = content_array[:,:,:,::-1]

#STYLE IMAGE PATHS COULD BE LOOPED FOR 'N' STYLES

# -------------------------------------------------
#STYLE IMAGE 1
style_image_path ='C:/Users/Dell/Desktop/python/BasicCodes/DeeplearningTensorflow/MyGithub/style0.jpg'
style_image = Image.open(style_image_path)
style_image = style_image.resize((width, height))
style_array = np.asarray(style_image, dtype='float32')
style_array = np.expand_dims(style_array, axis=0)
##make the mean 0--normalize
style_array[:, :, :, 0] -= 103.939
style_array[:, :, :, 1] -= 116.779
style_array[:, :, :, 2] -= 123.68
##convert to BGR
style_array = style_array[:, :, :, ::-1]


# --------------------------------------------------
#STYLE IMAGE 2

style1_image_path ='C:/Users/Dell/Desktop/python/BasicCodes/DeeplearningTensorflow/MyGithub/style1.jpg'
style1_image = Image.open(style1_image_path)
style1_image = style1_image.resize((width, height))
style1_array = np.asarray(style1_image, dtype='float32')
style1_array = np.expand_dims(style1_array, axis=0)
##make the mean 0--normalize
style1_array[:, :, :, 0] -= 103.939
style1_array[:, :, :, 1] -= 116.779
style1_array[:, :, :, 2] -= 123.68
##convert to BGR
style1_array = style1_array[:, :, :, ::-1]
# --------------------------------------------------

content_image = backend.variable(content_array)
style_image = backend.variable(style_array)
style1_image = backend.variable(style1_array)
combination_image = backend.placeholder((1,height,width,3))

#THIS IS THE TENSOR HAVING THE ALL THE 4 IMAGES CONCATENATED
input_tensor = backend.concatenate([content_image, style_image,combination_image,style1_image], axis=0)

myModel = VGG16(weights='imagenet', include_top='false',input_tensor= input_tensor)
layers = dict([(layer.name, layer.output) for layer in myModel.layers])
alpha = 0.025#content emphasis
beta = 5.0#style emphasis
theta = 5.0#style1 emphasis(optional)
gama = 1.0# combination variation emphasis
loss = backend.variable(0.)# initial loss


# -------------------------------------------------------
#CONTENT LOSS
layer_features = layers['block2_conv2']
content_image_features = layer_features[0,:,:,:]
combination_image_features = layer_features[2,:,:,:]
loss += alpha*(content_loss(combination_image_features, content_image_features))

#--------------------------------------------------------
#STYLE LOSS
feature_layers = ['block1_conv2','block2_conv2','block3_conv3','block4_conv3','block5_conv3']
for name in feature_layers:
	layer_features = layers[name]
	style_image_features = layer_features[1,:,:,:]
	style1_image_features = layer_features[3,:,:,:]
	combination_features = layer_features[2,:,:,:]
	loss += (beta/len(feature_layers))*style_loss(combination_features,style_image_features,style1_image_features)

#-------------------------------------------------------
#VARIATION LOSS
loss += gama*total_variation_loss(combination_image)

#defining gradients
grads = backend.gradients(loss, combination_image)

#############################################
outputs = [loss]
outputs += grads
f_outputs = backend.function([combination_image], outputs)

def eval_loss_and_grads(x):
    x = x.reshape((1, height, width, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    grad_values = outs[1].flatten().astype('float64')
    return loss_value, grad_values

class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()
#########################################################
iterations = 10
x = np.random.uniform(0, 255, (1, height, width, 3)) - 128.


for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))


x = x.reshape((height, width, 3))
x = x[:, :, ::-1]
x[:, :, 0] += 103.939
x[:, :, 1] += 116.779
x[:, :, 2] += 123.68
x = np.clip(x, 0, 255).astype('uint8')

Image.fromarray(x)

