from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
classifier=Sequential()
#convolution
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))
#max pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
#Flattening


classifier.add(Flatten())
#Hidden and output layers
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))
classifier.compile(optimizer='adam',loss= 'binary_crossentropy',metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
        'C:\\Users\\aksha\\Documents\\Machine_Learning_AZ_Template_Folder\\Machine Learning A-Z Template Folder\\Part 8 - Deep Learning\\Section 40 - Convolutional Neural Networks (CNN)\\Convolutional_Neural_Networks\\dataset\\training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'C:\\Users\\aksha\\Documents\\Machine_Learning_AZ_Template_Folder\\Machine Learning A-Z Template Folder\\Part 8 - Deep Learning\\Section 40 - Convolutional Neural Networks (CNN)\\Convolutional_Neural_Networks\\dataset\\test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
from PIL import Image
classifier.fit_generator(
        train_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        nb_val_samples=2000)




