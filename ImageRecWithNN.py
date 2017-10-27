import matplotlib.pyplot as plt
import numpy as np

#my sigmoid function:
def mySigmoid(x, deriv=False):
    out = np.empty(x.shape)
    #I iterate over each element to apply sigma derivative to each elem
    if(deriv==True):
        for i in range(len(x)):
            for j in range(len(x[i])):
                out[i][j] = x[i][j]*(1-x[i][j])
        return out
    #else quasi
    for i in range(len(x)):
        for j in range(len(x[i])):
            try:
                out[i][j] = 1/(1+np.exp(-x[i][j]))
            except:
                print("Dieses x wirft ein Error: ",x[i][j], " mit indizes i,j :", i, j)
    return out

#convert a number into 1-hot bit notation (0-9)
def one_hot(x):
    out = np.zeros((10))
    out[x] = 1
    return out

file1 = open("dataSets\mnist_small_train_in.txt")
file2 = open("dataSets\mnist_small_train_out.txt")

with file1 as f:
    data = f.read().split()

with file2 as f:
    train_out = list(map(int, f.read().split()))

train_in = []
for x in data:
    train_in.append(list(map(int, x.split(','))))

#my training data
train_in = np.array(train_in)

#training dataSets
X = train_in

Y = []
#output data
for x in train_out:
    Y.append(one_hot(x))

Y = np.array(Y)


np.random.seed(42)

#random weights to connect the nodes
syn0 = 2*np.random.random((784,1500))-1
syn1 = 2*np.random.random((1500,10))-1

errors = []
alpha = 0.05

for iter in range(1000):
    #forward propagation
    l0 = X
    #print("input:", np.dot(l0,syn0)[0])
    #print("ma synapse0:",syn0[:,0])
    l1 = mySigmoid(np.dot(l0,syn0)) #hidden layer
    l2 = mySigmoid(np.dot(l1,syn1)) #prediction
    #print("thats my l1", l1[0])

    #my error
    l2_error = Y - l2

    myError = sum(sum(l2_error**2))
    print("The error:",myError)
    errors.append(myError)

    #print("sigmoid: of ", l2[0], " = " , mySigmoid(l2, True)[0])

    l2_delta = np.multiply(l2_error, mySigmoid(l2, True))
    #print("l2_delta: ", l2_delta[0])
    l1_error = np.dot(l2_delta, syn1.T)

    l1_delta = np.multiply(l1_error, mySigmoid(l1, True))

    #in l1 delta ist kein Wert >= 1! l1_delta is fine größter Wert 0.1
    #print("l1_delta maximum", max(l1_delta.flatten()))

    #print("l0 shape: ", l0.shape)
    #print("l1_delta shape: ", l1_delta.shape)
    #new Weights
    #print("syn0 before:", syn0[:,0])
    print("max of dot", max((l0.T.dot(l1_delta) * alpha).flatten()))
    print("max in syn0", max(syn0.flatten()))
    syn0 += (l0.T.dot(l1_delta) * alpha)
    syn1 += (np.dot(l1.T, l2_delta) * alpha)

print("I AM FINISHED")
print("Prediction:", l2[0])
print("Real Y:", Y[0])


plt.plot(errors)
plt.show()
