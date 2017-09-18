# -*- coding: utf-8 -*-
"""
Created on Wed Sep 06 15:25:50 2017
A simple code for Bayesian Structural learning based on an algorithm on this website
https://www.slideshare.net/butest/an-algorithm-for-bayesian-network-construction-from-data
@author: Homagni
"""
import pandas
import numpy as np

'''
loading the datafile and storing as variable X
'''
dataframe = pandas.read_csv("titanic.csv", header=None)
dataset = dataframe.values
print(dataset[0])
featureLength=len(dataset[0])
print("The number of nodes in the graph ",featureLength)
X = dataset[1:,0:featureLength].astype(float) #The dataset 
'''
finding out number of symbols used for dicretizing each field
'''
numsymbols=np.zeros(featureLength)
maxx=np.zeros(featureLength)-np.ones(featureLength) #Assuming there is no symbol annotation below 0
numsymbols=np.zeros(featureLength)
for i in range(len(X)):
    for j in range(featureLength):
        if(X[i,j]>maxx[j]):
            maxx[j]=X[i,j]
            numsymbols[j]+=1


'''
Define a function to calculate P(l|k) for symbol l in vector Y and symbol k in vector X
'''
def gettransitionfunc(X,Y,k,l):
    count=0.0
    for i in range(len(X)):
        
        if(X[i]==k+1 and Y[i]==l+1):
            count+=1
    count=count/len(X)
    #print(count)
    return count
'''
Define function to calculate mutual information matrix between 2 variables 
n1 and n2 given their transition matrix X and the parent dataset D
'''   
def getMI(X,D,n1,n2):#
    count=0.0
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if(X[i,j]!=0):
                count+=X[i,j]*np.log(X[i,j]/(calcIndivProb(D[:,n1],i+1)*calcIndivProb(D[:,n2],j+1)))
            
    return count
'''
define function to calculate individual probability of symbol k in a sequence X
'''

def calcIndivProb(X,k):
    count=0.0
    for i in range(len(X)):
        if(X[i]==k):
            count+=1
    count=count/len(X)
    return count
'''
Code to Calculate the array of pairwise transition matrices or 
simply the joint probability matrix P(X,Y) for given 2 variables X and Y here M[i,j]
'''    
M={}#Store the mutual information matrices
count=0
print(numsymbols)
for i in range(featureLength):
    for j in range(featureLength):
        probmatrix=np.zeros((int(numsymbols[i]),int(numsymbols[j])))
        for k in range(int(numsymbols[i])):
            for l in range(int(numsymbols[j])):
                probmatrix[k,l]=gettransitionfunc(X[:,i],X[:,j],k,l)
                
        if(i!=j):
            #M.append([probmatrix])
            M[i,j]=probmatrix
        else:
            M[i,j]=-1
             
'''
Define a function for calculating partial correlation coeeficient for X and Y given Z 
This will be used for testing conditional independence
See wikipedia page on partial correlation
use this website to check for correctness
https://www.wessa.net/rwasp_partialcorrelation.wasp#output
'''
def partialCorrelation(X,Y,Z):
    e_x=calculateResiduals(X,Z)
    e_y=calculateResiduals(Y,Z)
    e_xy=0.0
    for i in range(len(e_x)):
        e_xy+=e_x[i]*e_y[i]
    e_xy=e_xy*len(e_x)
    e_xs=0.0
    e_ys=0.0
    e_x2=0.0
    e_y2=0.0
    N=len(e_x)
    for i in range(len(e_x)):
        e_xs+=e_x[i]
        e_x2+=e_x[i]*e_x[i]
        e_ys+=e_y[i]
        e_y2+=e_y[i]*e_y[i]
    rho=(e_xy-e_xs*e_ys)/(np.sqrt((N*e_x2-e_xs*e_xs)*(N*e_y2-e_ys*e_ys)))
    return rho
    
    
    
def calculateResiduals(X,Z): #Z needs to be augmented with 1
    ZZ=np.zeros((len(Z),2))
    for i in range (len(Z)):
        ZZ[i][0]=Z[i]
        ZZ[i][1]=1
    Z=ZZ
    wx=np.zeros((Z.shape[0],Z.shape[1]))
    print("Z")
    print(wx)
    wx=gradientDescent(X,wx,Z,0.01,0.0001,500)
    print("Weights")
    print(wx)
    residualx=np.zeros(len(X))
    for j in range (len(X)):
        dot=0.0
        for i in range (wx.shape[1]):
            dot+=wx[j][i]*Z[j][i]
        residualx[j]=X[j]-dot
    return residualx
    
def regressionObjective(x,w,z):
    square=0.0
    dot=np.zeros(len(x))
    for j in range (len(x)):
        
        for i in range (len(w)):
            dot[j]+=w[i]*z[j][i]
        square+=(x[j]-dot[j])*(x[j]-dot[j])
    return square

def gradient(x,w,z,epsilon):
    value=np.zeros(len(w))
    wf0=[w[0]+epsilon,w[1]]
    wb0=[w[0]-epsilon,w[1]]
    value[0]=(regressionObjective(x,wf0,z)-regressionObjective(x,wb0,z))/(2*epsilon)
    
    wf1=[w[0],w[1]+epsilon]
    wb1=[w[0],w[1]-epsilon]
    value[1]=(regressionObjective(x,wf1,z)-regressionObjective(x,wb1,z))/(2*epsilon)
    return value
    
def gradientDescent(x,w,z,alpha,epsilon,itern):
    for i in range(itern):
        k=w
        for j in range (w.shape[0]):
            w[j]=w[j]-alpha*gradient(x,k[j],z,epsilon)
    return w
    
'''
testing the matrix generation
'''

print(M[1,2])
print(M[2,0].shape)
print(getMI(M[3,4],X,3,4))
A=np.asarray([2,6,10,20])
B=np.asarray([1,3,2,4])
C=np.asarray([-1,2,-3,4])
print(partialCorrelation(A,B,C))
'''
Starting the Drafting phase
'''
E={}
S={}
R={}
'''
Create a sorted dictionary of node pairs based on ranking of their mutual information and store it in S
'''
count=0
for i in range (featureLength):
    for j in xrange(i+1,featureLength,1):
        val=getMI(M[i,j],X,i,j)
        S[count]=[i,j,val]
        count+=1
print("Unsorted node pairs with mutual informations")
print(S)
print("now sorting")
SS=S
values=np.zeros(len(S))
print(S[0])
print(S[0][2])
for i in range(len(S)):
    val=S[i][2]
    values[i]=val
          

A={}  

values=np.sort(values)

for i in range(len(values)):
    for j in range(len(S)):
        if(S[j][2]==values[i]):
            #print(j)
            #print(S[j])
            A[count]=S[j]
            count+=1
                 
print(A)

        



