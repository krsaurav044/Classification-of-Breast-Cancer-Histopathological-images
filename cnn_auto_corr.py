# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 23:16:11 2019

@author: saurav
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from sklearn.feature_extraction import image
from keras.regularizers import l2
from sklearn.manifold import LocallyLinearEmbedding

img_dir = "D:\\ml\\projects\\breast_cancer\\dataset\\benign\\40x"

images=[]
name=[]
i=0
for filename in os.listdir(img_dir):
    img = cv2.imread(os.path.join(img_dir,filename))
    if img is not None:
        img=resize(img,(120,180,3), mode = 'constant')
        images.append(img)
        name.append(0)
        i += 1


img_dir1 = "D:\\ml\\projects\\breast_cancer\\dataset\\melignant\\40x"

for filename1 in os.listdir(img_dir1):
    img = cv2.imread(os.path.join(img_dir1,filename1))
    if img is not None:
        img=resize(img,(120,180,3), mode = 'constant')
        images.append(img)
        name.append(1)
        i += 1

X_train, X_test, y_train, y_test = train_test_split(images, name, test_size=0.25, random_state=42)
X_test=np.asarray(X_test)
X_train=np.asarray(X_train)
y_train=np.asarray(y_train).reshape(1496,1)
y_test=np.asarray(y_test).reshape(499,1)
X_train=X_train.reshape(1496,120*180*3)
X_test=X_test.reshape(499,120*180*3)
from sklearn.preprocessing import StandardScaler
st=StandardScaler()
X_train=st.fit_transform(X_train)
X_test=st.transform(X_test)
X_train=X_train.reshape(1496,120,180,3)
X_test=X_test.reshape(499,120,180,3)
'''X_total=np.zeros((1995,227,227,3))
X_total[0:1396]=X_train[0:1396]
X_total[1396:1995]=X_test[0:599]'''
from sklearn.preprocessing import OneHotEncoder
en=OneHotEncoder(categorical_features=[0])
y_train=en.fit_transform(y_train).toarray()
#y_test=en.transform(y_test).toarray()

plt.imshow(X_train[1])

#patches=image.extract_patches_2d(X_train1,(64,64))

from keras import backend as K
from keras import regularizers
from keras import losses

#correntropy loss function
def correntropy(sigma=1.):
    def func(y_true, y_pred):
        return K.mean(K.exp(-K.sqr(y_true - y_pred)/2*sigma*sigma), -1)
    return func



from keras.models import Sequential,Model
from keras.layers import Dense,Input, Conv2D,Conv2DTranspose,Convolution2D, MaxPooling2D, Convolution2DTranspose, UpSampling2D
from keras.layers import BatchNormalization, Dropout, Flatten

#from keras.models import model_from_json
input_img =Input(shape=(120,180,3))

x=Conv2D(4,(3,3),activation='relu',strides=(1,1),padding='same')(input_img)
x=MaxPooling2D((2,2))(x)
x=Conv2D(8,(3,3),activation='relu',strides=(1,1),padding='same')(x)
encoded = MaxPooling2D((2,2))(x)
x=Conv2D(16,(3,3),activation='relu',strides=(1,1),padding='same')(x)
x = MaxPooling2D((2,2))(x)
x=Conv2D(32,(3,3),activation='relu',strides=(1,1),padding='same')(x)
encoded = MaxPooling2D((2,2))(x)

x=Conv2DTranspose(32,(3,3),activation='relu',strides=(1,1),padding='same')(encoded)
x = UpSampling2D((2,2))(x)
x=Conv2DTranspose(16,(3,3),activation='relu',strides=(1,1),padding='same')(x)
x = UpSampling2D((2,2))(x)
x=Conv2DTranspose(8,(5,5),activation='relu',strides=(1,1),padding='same')(x)
x=UpSampling2D((2,2))(x)
x=Conv2DTranspose(4,(3,3),activation='relu',strides=(1,1),padding='same')(x)
decoded=UpSampling2D((2,2))(x)
decoded=Conv2DTranspose(3,(1,5),activation='relu',strides=(1,1),kernel_regularizer=regularizers.l2(0.1))(x)
autoencoder=Model(input_img,decoded)
autoencoder1=Model(input_img,encoded)
corentropy=correntropy()
autoencoder.compile(optimizer='adam',loss=corentropy,metrics=['accuracy'])
history1=autoencoder.fit(X_train,X_train,validation_split=0.1,batch_size=64,epochs=150)
y=autoencoder.predict(X_train)

plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history1.history['acc'])
plt.plot(history1.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

autoencoder.save_weights('weight_absolute.h5')
for l1,L2 in zip(autoencoder1.layers[:9],autoencoder.layers[0:9]):
    l1.set_weights(L2.get_weights())
autoencoder.get_weights()[0][1]
autoencoder1.get_weights()[0][1]

X_t = autoencoder1.predict(X_train)
X_t = X_t.reshape(1496,15*22*32)
X_te = autoencoder1.predict(X_test)
X_te = X_te.reshape(499,15*22*32)

from sklearn.preprocessing import StandardScaler
st=StandardScaler()
X_t=st.fit_transform(X_t)
X_te=st.transform(X_te)



model=Sequential()
model.add(Dense(output_dim=512,activation='relu',input_dim=15*22*32))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())
model.add(Dense(output_dim=512,activation='relu'))
model.add(Dense(output_dim=1024,activation='relu'))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())
model.add(Dense(output_dim=2,activation='sigmoid',kernel_regularizer=regularizers.l2(0.01)))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history2=model.fit(X_t,y_train,validation_split=0.1,batch_size=64,nb_epoch=50)
y_pred = model.predict(X_te)


plt.plot(history2.history['acc'])
plt.plot(history2.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


############

#model1.save_weights("model.h5")

#model_json = model1.to_json()
#with open("model.json", "w") as json_file:
#    json_file.write(model_json)

'''json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

y_p=loaded_model.predict(X_test)'''

#y_p=en.inverse_transform(y_p)

#y_pred=model1.predict(X_test)
y_pred=en.inverse_transform(y_pred)
#y_test=en.inverse_transform(y_test)


from sklearn.metrics import accuracy_score
accuracy1=accuracy_score(y_pred,y_test)

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test, y_pred)

imge=model.predict(im)
plt.imshow(X_t[3])
plt.imshow(y[4])































