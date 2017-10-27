import matplotlib.pyplot as plt
import numpy as np
import random

global test_X
global test_in

#sigmoid function + derivative
def mySigmoid(x, deriv=False):
    #input x is 1 dim vector with 10 or 784 elem
    #maybe need to change the code below to loop through array
    if(deriv==True):
        out = np.empty((len(x),1))
        for i in range(len(x)):
            out[i] = x[i]*(1-x[i])
        return out
    return 1/(1+np.exp(-x))

def one_hot(x):
    out = np.zeros((10,1))
    out[x] = 1
    return out

#input one_hot array with 10 elements returnes the one with the highest prob
def findGuess(x):
    highestVal = max(x)
    for i in range(len(x)):
        if(x[i] == highestVal):
            return i
    print("------------_ERROR_----------------")
    return 0

def testRun(syn0, syn1):
    correct_count = 0

    for index in range(len(test_in)):
        #forward propagation
        l0 = np.empty((784,1))
        l0 = test_X[index]

        l1 = np.empty((28,1))
        l1 = mySigmoid(np.dot(l0.T,syn0)).T

        l2 = np.empty((10,1))
        l2 = mySigmoid(np.dot(l1.T,syn1)).T

        guess = findGuess(l2)
        if(guess == test_out[index]):
            correct_count += 1

    percentage = correct_count / len(test_out)
    print("correct: ", percentage)
    return percentage


file1 = open("dataSets\mnist_small_train_in.txt")
file2 = open("dataSets\mnist_small_train_out.txt")

file3 = open("dataSets\mnist_small_test_in.txt")
file4 = open("dataSets\mnist_small_test_out.txt")

with file1 as f:
    train_in_raw = f.read().split()

with file2 as f:
    train_out = list(map(int, f.read().split()))

with file3 as f:
    test_in_raw = f.read().split()

with file4 as f:
    test_out = list(map(int, f.read().split()))


train_in = []
test_in = []
for x in train_in_raw:
    train_in.append(list(map(int, x.split(','))))

for x in test_in_raw:
    test_in.append(list(map(int, x.split(','))))

Y = np.empty((6006,10,1))
for i in range(len(train_out)):
    Y[i] = (one_hot(train_out[i]))

#input Zeugs
X = np.array(train_in)
X.shape = (6006,784,1)

#output Zeugs
Y = np.array(Y)

#the array we will iterate through to have different inputs not always 0s
indexArr = np.arange(len(X))
random.shuffle(indexArr)

#for validation
test_X = np.array(test_in)
test_X.shape = ((len(test_in), 784, 1))

#Weights
syn0 = 2*np.random.random((784,28))-1
syn1 = 2*np.random.random((28,10))-1

errors = []
percentages = []
countTest = 0 #this counter will run every 1000 iterations a testRun

for i in range(5):
    for index in indexArr:

        #to verify and print get the percentages
        if(countTest % 1000 == 0):
            print("Iteration ", countTest)
            percentages.append(testRun(syn0, syn1))

        countTest += 1

        #forward propagation
        l0 = np.empty((784,1))
        l0 = X[index]

        l1 = np.empty((28,1))
        l1 = mySigmoid(np.dot(l0.T,syn0)).T

        l2 = np.empty((10,1))
        l2 = mySigmoid(np.dot(l1.T,syn1)).T

        #print("my prediction: ", l2, " of Y: ", Y[index])

        l2_error = np.empty((10,1))
        l2_error = Y[index] - l2

        errors.append(sum(l2_error**2))

        l2_delta = np.empty((10,1))
        l2_delta = np.multiply(l2_error, mySigmoid(l2, True))

        l1_error = np.empty((28,1))
        l1_error = np.dot(syn1, l2_delta)

        l1_delta = np.empty((28,1))
        l1_delta = np.multiply(l1_error, mySigmoid(l1, True))

        syn0 += (np.dot(l0, l1_delta.T))
        syn1 += (np.dot(l1, l2_delta.T))

plt.plot(percentages)
plt.show()
