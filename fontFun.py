#!/usr/bin/python3

imgDir='fontImages'

import os
import collections
from keras.preprocessing import image
fonts = collections.defaultdict(dict)
for file in os.listdir(imgDir):
	if file.endswith(".png"):
		path = os.path.join(imgDir, file)
		fileName = os.path.basename(path)
		fontName = fileName[:-10]
		letter = fileName[-5:-4]
		img = image.load_img(path)
		img = image.img_to_array(img)
		fonts[fontName][letter] = img

As, Bs = [], []
for fontName, letters in fonts.items():
	As.append( letters['A']	)
	Bs.append( letters['B']	)
import numpy as np
As, Bs = np.array(As), np.array(Bs)
As = As.astype('float32')/255. 
Bs = Bs.astype('float32')/255. 
print('shapes: ', As.shape, Bs.shape)
AsShape, BsShape = As[0].shape, Bs[0].shape
		
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

input_img = Input(shape=AsShape) # 1ch=black&white, 28 x 28

x = Convolution2D(16, (3, 3), activation='relu', padding='same')(input_img) #nb_filter, nb_row, nb_col
x = MaxPooling2D((2, 2), padding='same')(x)
x = Convolution2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Convolution2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

print("shape of encoded", K.int_shape(encoded))
x = Convolution2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, (3, 3), activation='relu', padding='same')(x) 
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(3, (5, 5), activation='sigmoid', padding='same')(x)
print("shape of decoded", K.int_shape(decoded))

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
print(autoencoder.summary())

from keras.callbacks import TensorBoard

autoencoder.fit(As, Bs, epoch=50, batch_size=128,
		shuffle=True, validation_split=0.1, verbose=1,
		callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
