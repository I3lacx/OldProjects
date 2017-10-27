import matplotlib.pyplot as plt
import numpy as np

file = open("dataSets\mnist_small_test_in.txt")

with file as f:
    data = f.read().split()

train_in = []
for x in data:
    train_in.append(list(map(int, x.split(','))))

#my training data
train_in = np.array(train_in)

#my sigmoid function:
def mySigmoid(x, deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

#training dataSets
X = np.array( [[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]] )

#output data
y = np.array([0,1,1,0]).T
y.shape = (4,1)

np.random.seed(42)

syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

errors = []

for iter in range(10000):
    #forward propagation
    l0 = X
    l1 = mySigmoid(np.dot(l0,syn0)) #hidden layer
    l2 = mySigmoid(np.dot(l1,syn1)) #prediction

    #my error
    l2_error = y - l2
    errors.append(sum(l2_error**2))

    l2_delta = np.multiply(l2_error, mySigmoid(l2, True))
    l1_error = np.dot(l2_delta, syn1.T)

    l1_delta = np.multiply(l1_error, mySigmoid(l1, True))

    #new Weights
    syn0 += np.dot(l0.T, l1_delta)
    syn1 += np.dot(l1.T, l2_delta)

print("I AM FINISHED")
print(l2)

plt.plot(errors)
plt.show()
