import numpy as np
import matplotlib.pyplot as plt
from TicTacToe import *
from NeuralNetwork import *


def generateData(nb_examples):
    gameFields = []
    results = []

    for i in range(nb_examples):
        newField = TicTacToe()
        field,result = randomGame(newField)
        gameFields.append(field)
        results.append(result)

    return gameFields, results

def into3outputs(out):
    newOutArr = []
    for x in out:
        if(x == 1):
            newOutArr.append([0,1,0])
        elif(x == 2):
            newOutArr.append([0,0,1])
        elif(x == 0):
            newOutArr.append([1,0,0])
        else:
            print("WHHHAT DA FUCK")
    return newOutArr

def testIt(NN, testInput, testOutput):
    wrongCount = 0
    for i in range(len(testInput)):
        pred = NN.forward(testInput[i].T)
        pred = pred.argmax()
        if(pred != testOutput[i]):
            wrongCount += 1
    return wrongCount

trainIn, trainOut = generateData(1000)
testInput, testOutput = generateData(100)

#transform Output Data
trainOut = into3outputs(trainOut)

NN = Neural_Network()
costFunct = NN.trainMe(trainIn, trainOut)
print(testIt(NN, testInput, testOutput))

#plt.plot(costFunct)
#plt.show()
