import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time

def randomfloat(a,b,c):
    return random.randint(a, b)/10

class MulCell:
    def __init__(self,_num):
        self.num = _num
        self.x = 0
        self.w = randomfloat(7, 10, 10)

    def forward(self,x):
        z = x * self.w
        self.x = x
        return z
    
    def backward(self,dz):
        dx = self.w * dz 
        dw = self.x * dz
        return dx,dw

    def getw(self):
        return self.w

    def setw(self,_w):
        self.w = _w

class AddCell:
    def __init__(self,_num):
        self.num = _num
        self.x = 0
        self.b = randomfloat(0, 2, 10)

    def forward(self,x):
        z = x + self.b
        self.x = x
        return z
    
    def backward(self,dz):
        dx = dz 
        db = dz
        return dx,db
    
    def getb(self):
        return self.b

    def setb(self,_b):
        self.b = _b

class Graph:
    def __init__(self):
        self.cell_0 = MulCell(0)
        self.cell_1 = AddCell(1)
        self.cells = [self.cell_0,self.cell_1]
        self.loss = []
        self.epochs = []
    
    def forward(self,x,y):
        output = x
        for cell in self.cells:
            output += cell.forward(output)
        loss = y - output
        return loss

    def backward(self):
        dz = 1
        da,db = self.cells[1].backward(dz)
        dw,dx = self.cells[0].backward(da)
        #print(f'da:{da},db:{db}')
        #print(f'dw:{dw},dx:{dx}')
        return dw,db
    
    def sgd(self,x,y):
        lr = 0.001
        loss = self.forward(x, y)
        dw,db = self.backward()
        _w = loss * dw
        _b = loss * db
        w = self.cell_0.getw()
        b = self.cell_1.getb()
        w += lr * _w
        b += lr * _b
        self.cell_0.setw(w)
        self.cell_1.setb(b)
        #print(f'w:{w},b:{b}')
        print(f'loss:{loss}')
        return loss

    def predict(self,test):
        outputs = 0
        true = 0
        for x in test:
            output = x
            for cell in self.cells:
                output += cell.forward(output)
            print(f'input:{x},predict:{output},true:{x * 4}')
            outputs +=output
            true += x*4
        print(f'Accuracy:{(outputs/true)*100}%')
    
    def train(self,_num,x_data,y_data):
        self.loss = [list(),list(),list()]
        self.epochs = []
        for i in range(_num):
            print(f'epoch: {i}\n')
            for j in range(3):
                self.loss[j].append(self.sgd(x_data[j], y_data[j]))
            self.epochs.append(i)
            print('\n\n\n')
    
    def visiualize(self):
        plt.plot(self.epochs,self.loss[0],color = 'green',label = f'data {0}')
        plt.plot(self.epochs,self.loss[1],color = 'red',label = f'data {1}')
        plt.plot(self.epochs,self.loss[2],color = 'skyblue',label = f'data {2}')
        plt.ylim((-1,6))
        plt.legend()
        plt.show()

def main():
    x_data = [1.0, 2.0, 3.0]
    y_data = [4.0, 8.0, 12.0]
    x_test = [3,6,17,5,20]
    Graph1 = Graph()
    Graph1.train(400,x_data,y_data)
    Graph1.visiualize()
    Graph1.predict(x_test)

if __name__ == "__main__":
    main()