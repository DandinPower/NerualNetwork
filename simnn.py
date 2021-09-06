import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
def randomfloat(a,b,c):
    return random.randint(a, b)/c

def abs(x):
    if x >= 0:
        return x
    else:
        return -x

def posorneg(x,target):
    pos = 0
    neg = 0
    for data in x:
        if data > 0:
            pos += 1
        elif data <0:
            neg += 1
    x = random.randint(1,pos+neg)
    if x <= pos:
        return target
    else:
        return -target

class MulCell:
    def __init__(self,_num):
        self.num = _num
        self.x = 0
        self.w = randomfloat(1, 10, 10)

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
        self.b = randomfloat(-10, 10, 10)

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

class Relu:
    def __init__(self,_num):
        self.num = _num
        self.x = 0
    
    def forward(self,x):
        self.x = x
        if x >= 0:
            return x
        else:
            return 0

    def backward(self,dz):
        if self.x >= 0:
            return dz
        else:
            return 0

class Neural:
    def __init__(self,_num):
        self.num = _num
        self.cells = [MulCell(0),AddCell(1),Relu(2)]

    def forward(self,x):
        z = self.cells[0].forward(x)
        y = self.cells[1].forward(z)
        s = self.cells[2].forward(y)
        return s

    def backward(self,dz):
        ds = self.cells[2].backward(dz)
        da,db = self.cells[1].backward(ds)
        dw,dx = self.cells[0].backward(da)
        return (dw,db)

    def getw(self):
        return self.cells[0].getw()

    def getb(self):
        return self.cells[1].getb()

    def setw(self,_w):
        return self.cells[0].setw(_w)

    def setb(self,_b):
        return self.cells[1].setb(_b)

class Graph:
    def __init__(self):
        self.neurals = [Neural(0),Neural(1),Neural(2)]
        self.loss = []
        self.epochs = []
    
    def forward(self,x,y):
        loss = 0
        score = 0
        for neural in self.neurals:
            score += neural.forward(x)
        loss = y - score
        loss /= y
        return loss

    def backward(self):
        dz = 1
        dwdbs = []
        for neural in self.neurals:
            dwdb = neural.backward(dz)
            dwdbs.append(dwdb)
        return dwdbs
    
    def sgd(self,x,y):
        lr = 0.0001
        loss = 0
        y_total = 0
        for i in range(len(x)):
            y_total += y[i]
            posnegs = []
            results = self.forward(x[i],y[i])
            print(x[i],y[i])
            print(results)
            posnegs.append(results)
            loss += abs(results)
        y_total /= len(x)
        dwdbs = self.backward()
        for i in range(3):
            dw = dwdbs[i][0]
            db = dwdbs[i][1]
            _w = posorneg(posnegs, loss) * dw * y_total
            _b = posorneg(posnegs, loss) * db * y_total
            w = self.neurals[i].getw()
            b = self.neurals[i].getb()
            w += lr * _w
            b += lr * _b
            self.neurals[i].setw(w)
            self.neurals[i].setb(b)
        return loss

    def predict(self,test):
        outputs = 0
        true = 0
        for x in test:
            output = 0
            for neural in self.neurals:
                output += neural.forward(x)
            print(f'input:{x},predict:{output},true:{x * 4}')
            outputs +=output
            true += x*4
        print(f'Accuracy:{(outputs/true)*100}%')
    
    def train(self,_num,x_data,y_data):
        self.loss = []
        self.epochs = []
        for i in range(_num):
            print(f'epoch: {i}\n')
            loss = self.sgd(x_data, y_data)
            self.loss.append(loss)
            print(f'loss: {loss}\n')
            self.epochs.append(i)
            print('\n\n\n')
    
    def visiualize(self):
        plt.plot(self.epochs,self.loss,color = 'green',label = f'data {0}')
        plt.show()
    
    def visualparm(self):
        for nerual in self.neurals:
            print(f'w: {nerual.getw()} b: {nerual.getb()}')


def main():
    x_data = [2,3,5,8,13,21,34]
    y_data = [8,12,20,32,52,84,136]
    x_test = [1,4,6,7,9]
    Graph1 = Graph()
    Graph1.train(120,x_data,y_data)
    Graph1.visiualize()
    Graph1.predict(x_data)
    Graph1.predict(x_test)
    #Graph1.visualparm()


if __name__ == '__main__':
    main()
