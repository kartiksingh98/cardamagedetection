from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNet
from keras.layers import Dropout
from keras.applications.inception_v3 import preprocess_input, decode_predictions

#create base model using MobileNet.
base_model=MobileNet(weights='imagenet', include_top=False, input_shape=(224,224,3))

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(512, activation='relu')(x)
x=Dense(256, activation='relu')(x)
prediction=Dense(3, activation='softmax')(x)

model=Model(inputs=base_model.input, outputs=prediction)

for layer in model.layers[:-5]:
    layer.trainable=False
for layer in model.layers[-5:]:
    layer.trainable=True
"""
benchmark model-deactivated
classifier=Sequential()
classifier.add(Convolution2D(32, 3, 3, input_shape=(256,256,3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(32, 3, 3, activation="relu"))
classifier.add(Flatten())

classifier.add(Dense(output_dim = 64, activation = 'relu'))

classifier.add(Dense(output_dim = 32, activation = 'relu'))

classifier.add(Dense(output_dim = 3, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(classifier.summary())
"""
train_datagen=ImageDataGenerator(rescale=1./255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 )
test_datagen=ImageDataGenerator(rescale=1./255)
train_set=train_datagen.flow_from_directory("C:/Users/karti/Desktop/Data/train/",
                                            target_size=(224,224),
                                            batch_size=32,
                                            color_mode='rgb',
                                            class_mode='categorical',
                                            shuffle=True
                                            )
test_set=test_datagen.flow_from_directory("C:/Users/karti/Desktop/Data/test/",
                                            target_size=(224,224),
                                            batch_size=32,
                                            color_mode='rgb',
                                            class_mode='categorical')

from keras.callbacks import ModelCheckpoint  #Checkpoint to save the best weights of the model.
checkpointer = ModelCheckpoint(filepath='weights.best.cnn.hdf5',
                               verbose=1, save_best_only=True)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

model.fit_generator(train_set,
                    samples_per_epoch=88,
                         epochs=3,
                         validation_data=test_set,
                         nb_val_samples=28,
                         callbacks=[checkpointer],
                         )

model.save('weightscnn.h5')


import numpy as np
from keras.preprocessing import image
img = image.load_img('1641.JPG', target_size=(256, 256))
img= image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)
classes = model.predict(img)
print(classes)
