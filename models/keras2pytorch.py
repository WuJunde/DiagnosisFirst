import tensorflow as tf
import numpy as np
import torch
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50
from keras.layers.convolutional import Conv2D
import keras
import collections

inputpath = './RadImageNet-ResNet50_notop.h5'
outpath = './RadImageNet-ResNet50_notop_torch.pth'
testimg = '../20478.PNG'

def simple_test(net):
    img = Image.open(testimg).convert('RGB')

    trans_data = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    ])

    img = trans_data(img).unsqueeze(0) 
    out = net(img)
    return out.squeeze(0)[0]   

def keras_to_pyt(km, pm=None):
	weight_dict = dict()
	for layer in km.layers:
		if (type(layer) is Conv2D) and ('0' not in layer.get_config()['name']):
			weight_dict[layer.get_config()['name'] + '.weight'] = np.transpose(layer.get_weights()[0], (3, 2, 0, 1))
			# weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1] as mean 
		elif type(layer) is keras.layers.Dense:
			weight_dict[layer.get_config()['name'] + '.weight'] = np.transpose(layer.get_weights()[0], (1, 0))
			weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]
	# print(weight_dict.keys())
	if pm:
		pyt_state_dict = pm.state_dict()
		for key in pyt_state_dict.keys():
			pyt_state_dict[key] = torch.from_numpy(weight_dict[key])
		pm.load_state_dict(pyt_state_dict)
		return pm
	return weight_dict

net = resnet50(num_classes = 1)
out = simple_test(net)
print('before output is', out)

tf_keras_model = tf.keras.models.load_model(inputpath)
weights = tf_keras_model.get_weights()

weights = keras_to_pyt(tf_keras_model)
values = list(weights.values())
i = 0
for name, param in net.named_parameters():
	if 'conv' in name:
		# print(name)
		param.data = torch.tensor(values[i])
		i += 1


out = simple_test(net)
print('after output is', out)

torch.save(net.state_dict(), outpath)