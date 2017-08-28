#!/usr/bin/python3

imgDir='fontImages'
valSplit = 0.1
seed = 0


# utility function for showing images
def show_imgs(x_test, decoded_imgs=None, y_test=None, n=10, show=True):
	import matplotlib.pyplot as plt
	from keras.preprocessing import image
	if decoded_imgs is     None and y_test is     None: plt.figure(figsize=(n  *0.5, 6))
	if decoded_imgs is not None and y_test is     None: plt.figure(figsize=(n*2*0.5, 6))
	if decoded_imgs is not None and y_test is not None: plt.figure(figsize=(n*3*0.5, 6))
	for i in range(n):
		ax = plt.subplot(3, n, i+1)
		img = image.array_to_img(x_test[i])
		plt.imshow(img)
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		if decoded_imgs is not None:
			ax = plt.subplot(3, n, i+1+n)
			img = image.array_to_img(decoded_imgs[i])
			plt.imshow(img)
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
		if y_test is not None:
			ax = plt.subplot(3, n, i+1+n*2)
			img = image.array_to_img(y_test[i])
			plt.imshow(img)
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
	if show:
		plt.show()
	else:
		import io
		buf = io.BytesIO()
		plt.savefig(buf, format='png')
		buf.seek(0)
		plt.close()
		return buf

import numpy as np
np.random.seed(seed)
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
As, Bs = np.array(As), np.array(Bs)
As = As.astype('float32')/255.
Bs = Bs.astype('float32')/255.
perm = np.random.permutation(len(As))
As, Bs, = As[perm], Bs[perm]

#As, Bs = As[:100], Bs[:100]

nVal = int(len(As) * valSplit)
AsVal, BsVal = As[:nVal], Bs[:nVal]
AsTrain, BsTrain = As[nVal:], Bs[nVal:]
#Amean, Bmean = AsVal.mean(), BsVal.mean()
#Astd, Bstd = AsVal.std(), BsVal.std()
#AsTrain = (AsTrain - Amean) / Astd
#AsVal = (AsVal - Amean) / Astd
#BsTrain = (BsTrain - Bmean) / Bstd
#BsVal = (BsVal - Bmean) / Bstd

print('shapes: ', AsTrain.shape, BsVal.shape)
AsShape, BsShape = AsTrain[0].shape, BsTrain[0].shape
		
import keras
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, BatchNormalization, Lambda, RepeatVector, Concatenate
from keras.models import Model
from keras import backend as K
import tensorflow as tf
 
class Visualize(keras.callbacks.Callback):
	def __init__(self, log_dir='./Logs'):
		self.logdir = log_dir
	def on_epoch_end(self, epoch, logs={}):
		if epoch % 1 != 0: return
		BsPred = self.model.predict(AsVal)
		buf = show_imgs(AsVal, BsPred, BsVal, show=False)
		image = tf.image.decode_png(buf.getvalue(), channels=4) # Add the batch dimension
		image = tf.expand_dims(image, 0) # Add image summary
		summary_op = tf.summary.image("Bness_%04d" % (epoch), image)
		with tf.Session() as sess:
			summary = sess.run(summary_op)
			writer = tf.summary.FileWriter(self.logdir)
			writer.add_summary(summary)
			writer.close()

input_img = Input(shape=AsShape)

#c = Convolution2D(16, (3, 3), activation='relu', padding='same')(input_img)
#c = MaxPooling2D((2, 2), padding='same')(c)
#c = Convolution2D(16, (3, 3), activation='relu', padding='same')(c)
#c = MaxPooling2D((2, 2), padding='same')(c)
#c = Convolution2D(16, (3, 3), activation='relu', padding='same')(c)
#c = MaxPooling2D((2, 2), padding='same')(c)
#c = Convolution2D(16, (3, 3), activation='relu', padding='same')(c)
#c = MaxPooling2D((2, 2), padding='same')(c)
#print('shape of content: ', K.int_shape(c))
#c = UpSampling2D((16, 16))(c)
#print('shape of content: ', K.int_shape(c))

c = Convolution2D(16, (3, 3), activation='relu', padding='same')(input_img)
c = MaxPooling2D((2, 2), padding='same')(c)
c = Convolution2D(8, (3, 3), activation='relu', padding='same')(c)
c = MaxPooling2D((2, 2), padding='same')(c)
c = Convolution2D(8, (3, 3), activation='relu', padding='same')(c)
c = MaxPooling2D((2, 2), padding='same')(c)
c = Convolution2D(8, (3, 3), activation='relu', padding='same')(c)
c = MaxPooling2D((2, 2), padding='same')(c)
print("shape of encoded content", K.int_shape(c))
c = Convolution2D(8, (3, 3), activation='relu', padding='same')(c)
c = UpSampling2D((2, 2))(c)
c = Convolution2D(8, (3, 3), activation='relu', padding='same')(c)
c = UpSampling2D((2, 2))(c)
c = Convolution2D(8, (3, 3), activation='relu', padding='same')(c)
c = UpSampling2D((2, 2))(c)
c = Convolution2D(16, (3, 3), activation='relu', padding='same')(c) 
c = UpSampling2D((2, 2))(c)
c = Convolution2D(16, (5, 5), activation='sigmoid', padding='same')(c)
print("shape of decoded content", K.int_shape(c))

s = Convolution2D(16, (3, 3), activation='relu', padding='same')(input_img)
s = Convolution2D(16, (3, 3), activation='relu', padding='same')(s)
s = Lambda(lambda x: K.mean(x, axis=[1,2], keepdims=True))(s)
s = UpSampling2D(AsShape[0:-1])(s)
print('shape of style: ', K.int_shape(s))

x = Concatenate()([s, c])
x = Convolution2D(16, (3, 3), activation='relu', padding='same')(x)
x = Convolution2D(16, (3, 3), activation='relu', padding='same')(x)
x = Convolution2D(1, (5, 5), activation='sigmoid', padding='same')(x)

#x = Convolution2D(16, (3, 3), activation='relu', padding='same')(input_img)
#x = MaxPooling2D((2, 2), padding='same')(x)
#x = Convolution2D(8, (3, 3), activation='relu', padding='same')(x)
#x = MaxPooling2D((2, 2), padding='same')(x)
#x = Convolution2D(8, (3, 3), activation='relu', padding='same')(x)
#x = MaxPooling2D((2, 2), padding='same')(x)
#x = Convolution2D(8, (3, 3), activation='relu', padding='same')(x)
#encoded = MaxPooling2D((2, 2), padding='same')(x)
#print("shape of encoded", K.int_shape(encoded))
#x = Convolution2D(8, (3, 3), activation='relu', padding='same')(encoded)
#x = UpSampling2D((2, 2))(x)
#x = Convolution2D(8, (3, 3), activation='relu', padding='same')(x)
#x = UpSampling2D((2, 2))(x)
#x = Convolution2D(8, (3, 3), activation='relu', padding='same')(x)
#x = UpSampling2D((2, 2))(x)
#x = Convolution2D(16, (3, 3), activation='relu', padding='same')(x) 
#x = UpSampling2D((2, 2))(x)
#decoded = Convolution2D(1, (5, 5), activation='sigmoid', padding='same')(x)
#print("shape of decoded", K.int_shape(decoded))

autoencoder = Model(input_img, x)
#autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
print(autoencoder.summary())

from keras.callbacks import TensorBoard, EarlyStopping
import datetime
dt = datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
logDir = './Logs/%s' % (dt)
callbacks=[	TensorBoard(log_dir=logDir), 
		EarlyStopping(patience=10), 
		Visualize(log_dir=logDir)]

autoencoder.fit(AsTrain, BsTrain, epochs=1000, batch_size=5,
		shuffle=True, validation_data=(AsVal, BsVal), verbose=1,
		callbacks=callbacks)
BsPred = autoencoder.predict(AsVal)
#BsPredDisp = BsPred * Bstd + Bmean
#AsValDisp = AsVal * Astd + Amean
#BsValDisp = BsVal * Bstd + Bmean
#show_imgs(AsValDisp, BsPredDisp, BsValDisp)
#show_imgs(AsVal, BsPred, BsVal)
