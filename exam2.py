import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
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

model = load_model('aaa.h5')

plt.imshow(x_test[0])
plt.show()


my_predict = model.predict(x_test[:1])[0]
print("real : ",y_test[0].argmax())
print("predict : ",my_predict)