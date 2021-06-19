import numpy as np
from sklearn.cluster import KMeans
KMeans.predict()


a = np.identity(2)

a = np.repeat(a[None,...], 3, axis=0)
print(a.shape)
for i in range(3):
    print(a[i])