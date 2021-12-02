import math
import numpy as np
import matplotlib.pyplot as plt

lr0: 0.0001  # initial learning rate (SGD=1E-2, Adam=1E-3)
lrf: 0.2  # final OneCycleLR learning rate (lr0 * lrf)
epochs = 60
lf =  lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine

list_lf = []
for i in range(epochs):
    value = lf(i)
    list_lf.append(value)

x = np.linspace(0, 60, 60)
plt.title('yolov5-s leaning rate line show ')
plt.figure(1, figsize=(8, 6))
plt.plot(x, list_lf, color='blue', lw=2, linestyle='--')
plt.show()
