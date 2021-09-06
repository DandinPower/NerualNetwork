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
        return (dw,db,dx)

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
        self.neurals_0 = [Neural(0),Neural(1),Neural(2)]
        self.neurals_1 = [Neural(0),Neural(1),Neural(2)]
        self.loss = []
        self.epochs = []
    
    def forward(self,x,y):
        loss = 0
        score = 0
        for neural_0 in self.neurals_0:
            tempy = neural_0.forward(x)
            for neural_1 in self.neurals_1:
                score += neural_1.forward(tempy)
        loss = y - score
        #loss = loss * loss
        if loss <= 0:
            posorneg = -1
        else: 
            posorneg = 1
        loss = abs(loss)
        return loss,posorneg

    def backward(self):
        dz = 1
        gradients_0 = []
        gradients_1 = []
        for neural_1 in self.neurals_1:
            gradient_1 = neural_1.backward(dz)
            dx = gradient_1[2]
            gradients_1.append(gradient_1)
            for neural_0 in self.neurals_0:
                gradient_0 = neural_0.backward(dx)
                gradients_0.append(gradient_0)
        gradients = [gradients_0,gradients_1]
        return gradients
    
    def sgd(self,x,y):
        lr = 0.0001
        loss = 0
        posornegs = []
        for i in range(len(x)):
            results,ret = self.forward(x[i],y[i])
            posornegs.append(ret)
            print(x[i],y[i])
            print(results)
            loss += results
        loss /= len(x)
        trueloss= loss
        loss = posorneg(posornegs,loss)
        gradients = self.backward()        
        for i in range(2):
            for j in range(3):
                gradient = gradients[i][j]
                dw = gradient[0]
                db = gradient[1]
                _w = loss * dw
                _b = loss * db
                if i == 0:
                    w = self.neurals_0[j].getw() + lr * _w
                    b = self.neurals_0[j].getb() + lr * _b
                    self.neurals_0[j].setw(w)
                    self.neurals_0[j].setb(b)
                else:
                    w = self.neurals_1[j].getw() + lr * _w
                    b = self.neurals_1[j].getb() + lr * _b
                    self.neurals_1[j].setw(w)
                    self.neurals_1[j].setb(b)
        return trueloss

    def predict(self,test):
        outputs = 0
        true = 0
        for x in test:
            output = 0
            for neural_0 in self.neurals_0:
                tempoutput = neural_0.forward(x)
                for neural_1 in self.neurals_1:

                    output += neural_1.forward(tempoutput)
            print(f'input:{x},predict:{output},true:{x * 2}')
            outputs +=output
            true += x*2
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
        for nerual_0 in self.neurals_0:
            print(f'1st layer -> w: {nerual_0.getw()} b: {nerual_0.getb()}')
        
        for nerual_1 in self.neurals_1:
            print(f'2nd layer -> w: {nerual_1.getw()} b: {nerual_1.getb()}')


def main():
    x_data = [4,5,6]
    y_data = [8,10,12]
    x_test = [1,4,6,7,9]
    Graph1 = Graph()
    Graph1.train(3000,x_data,y_data)
    Graph1.visiualize()
    Graph1.predict(x_data)
    Graph1.predict(x_test)
    Graph1.visualparm()


if __name__ == '__main__':
    main()
