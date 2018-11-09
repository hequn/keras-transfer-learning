import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.layers import AveragePooling2D, BatchNormalization, ZeroPadding2D, Dense, Dropout, Conv2D, MaxPooling2D, \
    Flatten

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(ZeroPadding2D((1,1), input_shape=(38, 38, 1)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same',))
# model.add(Conv2D(64, (3, 3), activation='relu', padding='same',))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same',))
# model.add(Conv2D(128, (3, 3), activation='relu', padding='same',))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(AveragePooling2D((5,5)))

model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


test_x = []
img_src = cv2.imdecode(np.fromfile(r'D:/2018052122CowImageTargetDealedXML/saveClassification/100364/000667-4.jpg', dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img_src, (38, 38), interpolation=cv2.INTER_CUBIC)
# img = np.random.randint(0,255,(38,38))
img = (255 - img) / 255
img = np.reshape(img, (38, 38, 1))
test_x.append(img)

###################################################################
layer = model.layers[1]
weight = layer.get_weights()
# print(weight)
print(np.asarray(weight).shape)
model_v1 = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model_v1.add(ZeroPadding2D((1, 1), input_shape=(38, 38, 1)))
model_v1.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model_v1.layers[1].set_weights(weight)

re = model_v1.predict(np.array(test_x))
print(np.shape(re))
re = np.transpose(re, (0,3,1,2))
for i in range(32):
    plt.subplot(4,8,i+1)
    plt.imshow(re[0][i]) #, cmap='gray'
plt.show()

##################################################################
model_v2 = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model_v2.add(ZeroPadding2D((1, 1), input_shape=(38, 38, 1)))
model_v2.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model_v2.add(BatchNormalization())
model_v2.add(MaxPooling2D(pool_size=(2, 2)))
model_v2.add(Dropout(0.25))

model_v2.add(Conv2D(64, (3, 3), activation='relu', padding='same', ))
print(len(model_v2.layers))
layer1 = model.layers[1]
weight1 = layer1.get_weights()
model_v2.layers[1].set_weights(weight1)
layer5 = model.layers[5]
weight5 = layer5.get_weights()
model_v2.layers[5].set_weights(weight5)
re2 = model_v2.predict(np.array(test_x))
re2 = np.transpose(re2, (0,3,1,2))
for i in range(64):
    plt.subplot(8,8,i+1)
    plt.imshow(re2[0][i]) #, cmap='gray'
plt.show()

##################################################################
model_v3 = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model_v3.add(ZeroPadding2D((1, 1), input_shape=(38, 38, 1)))
model_v3.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model_v3.add(BatchNormalization())
model_v3.add(MaxPooling2D(pool_size=(2, 2)))
model_v3.add(Dropout(0.25))

model_v3.add(Conv2D(64, (3, 3), activation='relu', padding='same', ))
# model.add(Conv2D(64, (3, 3), activation='relu', padding='same',))
model_v3.add(BatchNormalization())
model_v3.add(MaxPooling2D(pool_size=(2, 2)))
model_v3.add(Dropout(0.25))

model_v3.add(Conv2D(128, (3, 3), activation='relu', padding='same', ))

print(len(model_v3.layers))
layer1 = model.layers[1]
weight1 = layer1.get_weights()
model_v3.layers[1].set_weights(weight1)
layer5 = model.layers[5]
weight5 = layer5.get_weights()
model_v3.layers[5].set_weights(weight5)
layer9 = model.layers[9]
weight9 = layer9.get_weights()
model_v3.layers[9].set_weights(weight9)
re3 = model_v3.predict(np.array(test_x))
re3 = np.transpose(re3, (0,3,1,2))
for i in range(121):
    plt.subplot(11,11,i+1)
    plt.imshow(re3[0][i]) #, cmap='gray'
plt.show()

# from keras import backend as K
#
# # with a Sequential model
# get_3rd_layer_output = K.function([model.layers[0].input],
#                                   [model.layers[3].output])
# layer_output = get_3rd_layer_output([X])[0]
