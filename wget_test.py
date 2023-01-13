import numpy as np

!wget https://github.com/rickiepark/hg-mldl/blob/master/fruits_300.npy?raw=true -O fruits_300.npy
fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1,100*100)

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_2d)

print(km.labels_)