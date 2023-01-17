# from google.colab import drive
# drive.mount('/content/drive')
from tensorflow import keras
model = keras.models.load_model('best-cnn-model.h5')