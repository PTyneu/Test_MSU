

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, AvgPool2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

train_datagen = ImageDataGenerator(
    validation_split=0.25,
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2)

validation_datagen = ImageDataGenerator(
    validation_split=0.25,
    rescale=1./255,
)

train_datagen_flow = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/МГУ_тестовое/imgs_details/train',
    target_size=(150, 150),
    #batch_size=33,
    batch_size=1,
    class_mode='binary',
    subset='training',
    seed=12345)

val_datagen_flow = validation_datagen.flow_from_directory(
    '/content/drive/MyDrive/МГУ_тестовое/imgs_details/train',
    target_size=(150, 150),
    #batch_size=10,
    batch_size=1,
    class_mode='binary',
    subset='validation',
    seed=12345) 

datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = datagen.flow_from_directory(
    '/content/drive/MyDrive/МГУ_тестовое/imgs_details/test',
    target_size=(150, 150),
    #batch_size=21,
    batch_size=1,
    class_mode='binary')

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model_cnn_save_path = 'best_model_cnn.h5'
checkpoint_callback_cnn = ModelCheckpoint(model_cnn_save_path, 
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      verbose=1)

history_cnn = model.fit_generator(
    train_datagen_flow,
    steps_per_epoch=33,
    epochs=20,
    validation_data=val_datagen_flow,
    validation_steps=10,
    callbacks=[checkpoint_callback_cnn])

plt.plot(history_cnn.history['accuracy'], 
         label='Доля верных ответов на обучающем наборе')
plt.plot(history_cnn.history['val_accuracy'], 
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()

scores = model.evaluate_generator(test_generator, 1)
print("Accuracy на тестовой выборке: %.2f%%" % (scores[1]*100))

