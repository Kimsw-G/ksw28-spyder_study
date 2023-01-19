import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import load_model

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(y_train.shape)
IMG_PIX = x_train.shape[1] * x_train.shape[2]
TARGETS_LEN = len(set(y_train))
x_train = x_train / 255
x_test = x_test / 255

x_train_flattern = Flatten(input_shape=(x_train.shape[1],x_train.shape[2]))
x_test_flattern = Flatten(input_shape=(x_test.shape[1],x_test.shape[2]))

y_train = to_categorical(y_train ,TARGETS_LEN)
y_test = to_categorical(y_test, TARGETS_LEN)

dense1 = Dense(100,activation='relu')
dense2 = Dense(TARGETS_LEN,activation='softmax')
model = Sequential()
model.add(x_train_flattern)
model.add(dense1)
model.add(dense2)
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = 'accuracy')

history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.2)

print(history.history)
plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"],label='accur')
plt.plot(history.history["val_accuracy"],label='val_accur')
plt.legend()
plt.show()

model.save('aaa.h5')
