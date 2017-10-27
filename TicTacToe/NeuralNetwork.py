import numpy as np

class Neural_Network(object):

    def __init__(self):
        self.inputLayerSize = 9
        self.outputLayerSize = 3
        self.hiddenLayerSize = 9
        self.alpha = 1

        self.W1 = np.random.rand(self.inputLayerSize, \
                                    self.hiddenLayerSize)
        self.W2 = np.random.rand(self.hiddenLayerSize, \
                                    self.outputLayerSize)

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoidPrime(self,x):
        return np.exp(-x)/((1+np.exp(-x))**2)

    def forward(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def costFunction(self,X,y):
        self.yHat = self.forward(X)
        return sum(0.5*(y-self.yHat)**2)

    def costFunctionPrime(self, X, y):
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        self.dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3,self.W2.T) * self.sigmoidPrime(self.z2)
        self.dJdW1 = np.dot(X.T, delta2)


    def updateWeights(self):
        self.W1 = self.W1 - self.alpha * self.dJdW1
        self.W2 = self.W2 - self.alpha * self.dJdW2

    def trainMe(self, trainIn, trainOut):
        costArray = []
        for i in range(len(trainIn)):
            costArray.append(sum(self.costFunction(trainIn[i].T, trainOut[i])))
            self.costFunctionPrime(trainIn[i].T, trainOut[i])
            self.updateWeights()
        return costArray
