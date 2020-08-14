#importing Libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Intitalising CNN
classifier = Sequential()


#STEP 1 Convolution
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))  # input shape :- 3 channel
#Step2 MAxPooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#2nd convo layer
classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Step 3 Flattening
classifier.add(Flatten())

#Step 4 Full Connection
classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dense(output_dim=1,activation='sigmoid'))


#Compiling CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


#Fitting CNN
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        '/content/drive/My Drive/Colab Notebooks/CNN/P16-Convolutional-Neural-Networks/Part 2 - Convolutional Neural Networks/dataset/training_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        '/content/drive/My Drive/Colab Notebooks/CNN/P16-Convolutional-Neural-Networks/Part 2 - Convolutional Neural Networks/dataset/test_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')
classifier.fit_generator(
        training_set,
        samples_per_epoch=8000,
        nb_epoch=25,
        validation_data=test_set,
        nb_val_samples=2000)


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('/content/drive/My Drive/Colab Notebooks/CNN/P16-Convolutional-Neural-Networks/Part 2 - Convolutional Neural Networks/dataset/single_prediction/cat_or_dog_1.jpg',target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
results = classifier.predict(test_image)
print(training_set.class_indices)
if results[0][0] == 1:
  print('dog')
else:
  print('cat')
