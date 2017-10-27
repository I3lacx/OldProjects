import numpy as np
import random

class TicTacToe(object):
    def __init__(self):
        self.size = 9
        self.field = np.zeros((self.size, 1))
        self.currentPlayer = 1

    def convert(self, index):
        if(index >= len(self.field)):
            print("Error index too high")
            return " "

        if(self.field[index] == 0):
            return " "
        elif(self.field[index] == 1):
            return "O"
        elif(self.field[index] == 2):
            return "X"
        else:
            print("!ERROR! value not found")
            return " "

    def printMe(self):
        for i in range(3):
            print(" ---------------- ")
            for j in range(3):
                print(" | ", self.convert(j+i*3), end='')
            print(" |")
        print(" ---------------- ")

    def place(self, index):
        if(index >= len(self.field)):
            print("Error index too high")
            return

        if(self.field[index] != 0):
            #print("Error already used field, invalid move")
            return

        #field wird auf zahl gesetzt welcher spieler drann ist
        self.field[index] = self.currentPlayer

        #check for winner
        winner = self.checkWinner()
        if(winner != 0):
            #print("Aaaand the winner ist Player: ", winner)
            return

        #check for draw
        if(self.checkDraw == 1):
            #print("Game End Its a draw")
            return

        self.nextPlayer()

    def nextPlayer(self):
        if(self.currentPlayer == 1):
            self.currentPlayer = 2
        else:
            self.currentPlayer = 1

    def checkDraw(self):
        for i in range(self.size):
            if(self.field[i] == 0):
                return 0
        return 1

    def checkWinner(self):
        #horizontal check
        for i in range(0,9,3):
            if(self.field[i] == self.field[i+1] \
            and self.field[i] == self.field[i+2] \
            and self.field[i] != 0):
                #the winner is:
                return self.field[i][0]

        #vertical check
        for i in range(0,3):
            if(self.field[i] == self.field[i+3] \
            and self.field[i] == self.field[i+6] \
            and self.field[i] != 0):
            #the winner is:
                return self.field[i][0]

        #cross check
        if(self.field[0] == self.field[4] \
        and self.field[0] == self.field[8] \
        and self.field[0] != 0 \
        or self.field[2] == self.field[4] \
        and self.field[2] == self.field[6] \
        and self.field[2] != 0):
            #the winner is:
            return self.field[4][0]

        #still no winner
        return 0

def randomGame(game):
    while game.checkWinner() == 0 and game.checkDraw() == 0:
        game.place(random.randint(0,8))

    #returns the field as a list and the result
    winner = game.checkWinner()
    return game.field, winner
