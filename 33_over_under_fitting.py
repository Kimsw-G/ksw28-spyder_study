from tensorflow import keras
(train_input, train_target), (test_input, test_target) = \
    keras.datasets.fashion_mnist.load_data()
    
from sklearn.model_selection import train_test_split
train_scaled = train_input / 255.0
train_scaled, val_scaled, train_target, val_target = \
    train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)
    
def model_fn(a_layer=None):
    model = keras.Sequential()
    data_2d = keras.layers.Flatten(input_shape=(28,28))
    dense1 = keras.layers.Dense(100,activation='relu')
    dense2 = keras.layers.Dense(10,activation='softmax')
    model.add(data_2d)
    model.add(dense1)
    if a_layer:
        model.add(a_layer)
    model.add(dense2)
    return model    

model = model_fn()
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=20, 
                    validation_data=(val_scaled, val_target))

print(history.history.key())

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.show()
