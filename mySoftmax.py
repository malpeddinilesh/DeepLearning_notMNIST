# Softmax

scores = [3.0,1.0,2.0]

import numpy as np

def softmax(X):
    # Compute Softmax values in X 
    ex = np.exp(X-np.max(X))
    return ex / ex.sum(axis=0)

print(softmax(scores))

scores = [1.0,2.0,3.0]
print(softmax(scores))

scores = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])

print(softmax(scores))

# plot Softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2,6.0,0.1)
scores = np.vstack([x, np.ones_like(x),0.2*np.ones_like(x)])

plt.plot(x,softmax(x).T, linewidth=2)
plt.show()
