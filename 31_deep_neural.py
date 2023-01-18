from tensorflow import keras
(train_input, train_target), (test_input, test_target) = \
    keras.datasets.fashion_mnist.load_data()
    
from sklearn.model_selection import train_test_split
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1,28*28)
train_scaled, val_scaled, train_target, val_target = \
    train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)
    
dense1 = keras.layers.Dense(100,activation='sigmoid', input_shape=(784,))
dense2 = keras.layers.Dense(10,activation='softmax')

model = keras.Sequential([dense1, dense2])
# model.summary()


dense1 = keras.layers.Dense(100,activation='sigmoid', input_shape=(784,),name='hidden')
dense2 = keras.layers.Dense(10,activation='softmax',name='output')

model = keras.Sequential([dense1, dense2], name='패션 MNIST 모델')
# model.summary()

dense1 = keras.layers.Dense(100,activation='sigmoid', input_shape=(784,))
dense2 = keras.layers.Dense(10,activation='softmax')

model = keras.Sequential()
model.add(dense1)
model.add(dense2)
# model.summary()

# model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
# model.fit(train_scaled, train_target, epochs=5)


flatten0 = keras.layers.Flatten(input_shape=(28,28))
dense1 = keras.layers.Dense(100,activation='relu')
dense2 = keras.layers.Dense(10,activation='softmax')
model = keras.Sequential()
model.add(flatten0)
model.add(dense1)
model.add(dense2)

model.summary()

(train_input, train_target), (test_input, test_target) = \
    keras.datasets.fashion_mnist.load_data()
train_scaled = train_input / 255.0
train_scaled, val_scaled, train_target, val_target = \
    train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)
    
# model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
# model.fit(train_scaled, train_target, epochs=5)

flatten0 = keras.layers.Flatten(input_shape=(28,28))
dense1 = keras.layers.Dense(100,activation='relu')
dense2 = keras.layers.Dense(10,activation='softmax')
model = keras.Sequential()
model.add(flatten0)
model.add(dense1)
model.add(dense2)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs=5)