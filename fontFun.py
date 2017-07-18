#!/usr/bin/python3

imgDir='fontImages'
valSplit = 0.1

# utility function for showing images
def show_imgs(x_test, decoded_imgs=None, n=10):
	import matplotlib.pyplot as plt
	from keras.preprocessing import image
	plt.figure(figsize=(20, 4))
	for i in range(n):
		ax = plt.subplot(2, n, i+1)
		img = image.array_to_img(x_test[i])
		plt.imshow(img)
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

		if decoded_imgs is not None:
			ax = plt.subplot(2, n, i+ 1 +n)
			img = image.array_to_img(decoded_imgs[i])
			plt.imshow(img)
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
	plt.show()

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
		img = image.load_img(path, grayscale=True)
		img = image.img_to_array(img)
		fonts[fontName][letter] = img

As, Bs = [], []
for fontName, letters in fonts.items():
	try:
		A = letters['A']
		B = letters['B']
		As.append(A)
		Bs.append(B)
	except: pass
import numpy as np
As, Bs = np.array(As), np.array(Bs)
As = As.astype('float32')/255.
Bs = Bs.astype('float32')/255.
nData = len(As)
perm = np.random.permutation(nData)
As, Bs, = As[perm], Bs[perm]
nVal = int(nData * valSplit)

AsVal, BsVal = As[:nVal], Bs[:nVal]
AsTrain, BsTrain = As[nVal:], Bs[nVal:]
print('shapes: ', AsTrain.shape, BsVal.shape)
AsShape, BsShape = AsTrain[0].shape, BsTrain[0].shape
		
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.models import Model
from keras import backend as K

input_img = Input(shape=AsShape) # 1ch=black&white, 28 x 28

# x = BatchNormalization()(input_img)
x = Convolution2D(16, (3, 3), activation='relu', padding='same')(input_img)
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
decoded = Convolution2D(1, (5, 5), activation='sigmoid', padding='same')(x)
print("shape of decoded", K.int_shape(decoded))

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
print(autoencoder.summary())

from keras.callbacks import TensorBoard, EarlyStopping
import datetime
dt = datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
callbacks=[TensorBoard(log_dir='./Logs/%s' % (dt)), EarlyStopping(patience=5)]

autoencoder.fit(AsTrain, BsTrain, epochs=1000, batch_size=1,
		shuffle=True, validation_data=(AsVal, BsVal), verbose=1,
		callbacks=[TensorBoard(log_dir='./Logs/%s' % (dt))])
BsPred = autoencoder.predict(AsVal)
show_imgs(AsVal, BsPred)
