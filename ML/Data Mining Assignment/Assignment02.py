# -*- coding: utf-8 -*-

#  DO NOT CHANGE THIS PART!!!
#  DO NOT USE PACKAGES EXCEPT FOR THE PACKAGES THAT ARE IMPORTED BELOW!!!
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data=pd.read_csv('https://drive.google.com/uc?export=download&id=1AoCh22pmLHhdQtYdYUAJJqOCwF9obgVO', sep='\t')
data['class']=(data['class']=='g')*1

X=data.drop('class',axis=1).values
y=data['class'].values

trainX,testX,trainY,testY=train_test_split(X,y,stratify=y,test_size=0.2,random_state=11)

#TODO: Logistic regression



#TODO: B - Predict output based on probability of class g (y=1)
clf=LogisticRegression()
clf.fit(trainX,trainY)
a=clf.predict_proba(testX)
prob1 = clf.predict_proba(testX)[:,1]
cutoff = []
accuracy = []
for i in range(18):
    cutoff.append(round(0.1+i*0.05,2))
    count = 0
    for j in cutoff:
        box = []
        box.append([1 if k>j else 0 for k in prob1])
        box = np.array(box,ndmin=2).T

    for m in range(len(box)):
        if box[m] == testY[m]:
            count += 1
       
    accuracy.append(count/len(testY))
#TODO: Draw a line plot (x=cutoff, y=accuracy)
plt.plot(cutoff,accuracy)

#TODO: Bernoulli naïve Bayes
#TODO: B - Estimate parameters of Bernoulli naïve Bayes
# Write user-defined function to estimate parameters of Bernoulli naïve Bayes
def BNB(X,y):
    ######## BERNOULLI NAIVE BAYES ########
    # INPUT 
    # X: n by p array (n=# of observations, p=# of input variables)
    # y: output (len(y)=n, categorical variable)
    # OUTPUT
    # pmatrix: 2-D array(list) of size c by p with the probability p_ij where c is number of unique classes in y
        
    # TODO: Bernoulli NB
    sample_mean = X.mean(0)
    classi = 1*(X>sample_mean)
    pmatrix = []
    arr = []
    arr = classi.tolist()
    
    I = [[0 for rows in range(2)]for cols in range(10)]

    for i in range(len(X)):
        for j in range(len(I)):
            if arr[i][j] == 1 and y[i] == 1:
                I[j][0]+=arr[i][j]
            elif arr[i][j] == 1 and y[i] == 0:
                I[j][1]+=arr[i][j]
    
   
    
    for j in range(len(I)):
        I[j][0] = I[j][0]/(np.count_nonzero(y))
        I[j][1] = I[j][1]/(len(y)-np.count_nonzero(y))
    
        
    pmatrix = I
    return pmatrix
matrix=BNB(trainX,trainY)
#TODO: calculate p values of several Bernoulli distributions

#TODO: C - Predict output based on probability of class g (y=1)
testXbin = np.zeros(np.shape(testX),dtype='int')
testXmean = np.mean(testX,axis=0)
for i in range(len(testX)):
    for j in range(np.shape(testX)[1]):
        if testX[i,j]>=testXmean[j]:
            testXbin[i,j]=1
testXbin=(testX>=testXmean)*1
prob = [np.count_nonzero(testY)/len(testY),(len(testY)-np.count_nonzero(testY))/len(testY)]
bnb = []
n,p=np.shape(testX)
for i in range(n):
    r = 1
    q = 1
    result = 0
    for j in range(p):
        if testXbin[i][j] == 1:
            q *= matrix[j][0]
            r *= matrix[j][1]
        else:
            q *= 1-matrix[j][0]
            r *= 1-matrix[j][1]
    q = q*prob[0]
    r = r*prob[1]
    result = q/(r+q)
    bnb.append(result)
#TODO: Draw a line plot (x=cutoff, y=accuracy)
accuracy2 = []
for i in cutoff:
    count = 0
    box2 = []
    for j in bnb:
        if j > i:
            box2.append(1)
        else:
            box2.append(0)
    for k in range(len(testY)):
        if box2[k] == testY[k]:
            count +=1
    accuracy2.append(count/len(testY))
plt.plot(cutoff,accuracy2)
#TODO: Nearest neighbor 
#TODO: B - k-NN with uniform weights
# Write user-deinfed function of k-NN 
# Use imported distance functions implemented by sklearn
def euclidean_dist(a,b):
    ######## EUCLIDEAN DISTANCE ########
    # INPUT
    # a: 1-D array 
    # b: 1-D array 
    # a and b have the same length
    # OUTPUT
    # d: Euclidean distance between a and b
    
    # TODO: Euclidean distance
    d = 0
    d=euclidean_distances(a,b)
    
    return d

def manhattan_dist(a,b):
    ######## EUCLIDEAN DISTANCE ########
    # INPUT
    # a: 1-D array 
    # b: 1-D array 
    # a and b have the same length
    # OUTPUT
    # d: Manhattan distance between a and b
    
    # TODO: Manhattan distance
    d = 0
    d=manhattan_distances(a,b)
    return d

def knn(trainX,trainY,testX,k,dist=euclidean_dist):
    ######## K-NN Classification ########
    # INPUT 
    # trainX: training input dataset, n by p size 2-D array
    # trainY: training output target, 1-D array with length of n
    # testX: test input dataset, m by p size 2-D array
    # k: the number of the nearest neighbors
    # dist: distance measure function
    # OUTPUT
    # y_pred: predicted output target of testX, 1-D array with length of m
    #         When tie occurs, the final class is select in alpabetical order
    #         EX) if "A" ties "B", select "A" and if "2" ties "4", select 2
    
    # TODO: k-NN classification
    distance=[]        
    y_pred = []
    distance = dist(testX,trainX)
    
    classes = []
    
    for l in range(len(testX)):
        classes.append(trainY[np.argsort(distance[l])[:k]])
    
    for n in range(len(testX)):
        unique, counts = np.unique(classes[n], return_counts = True)
        
        if len(unique)==1:
            y_pred.append(unique[0])
        else:
            if counts[0] >= counts[1]:
                y_pred.append(0)
            else:
                y_pred.append(1)
    return y_pred


# TODO: Calculate accuracy of test set 
#       with varying the number neareset neighbors (k) and distance metrics
#       using k-NN
"""
pred3e = knn(trainX,trainY,testX,3)
pred5e = knn(trainX,trainY,testX,5)
pred7e = knn(trainX,trainY,testX,7)
pred3m = knn(trainX,trainY,testX,3,dist=manhattan_dist)
pred5m = knn(trainX,trainY,testX,5,dist=manhattan_dist)
pred7m = knn(trainX,trainY,testX,7,dist=manhattan_dist)
count = 0
count1 = 0
count2 = 0
count3 = 0
count4 = 0
count5 = 0

for i in range(len(testX)):
    if pred3e[i] == testY[i]:
        count +=1

for i in range(len(testX)):
    if pred5e[i] == testY[i]:
        count1 += 1
        
for i in range(len(testX)):
    if pred7e[i] == testY[i]:
        count2 += 1

for i in range(len(testX)):
    if pred3m[i] == testY[i]:
        count3 += 1

for i in range(len(testX)):
    if pred5m[i] == testY[i]:
        count4 += 1

for i in range(len(testX)):
    if pred7m[i] == testY[i]:
        count5 += 1
accuracy3e = count/len(testY)
accuracy5e = count1/len(testY)
accuracy7e = count2/len(testY)
accuracy3m = count3/len(testY)
accuracy5m = count4/len(testY)
accuracy7m = count5/len(testY)



#TODO: C - weighted k-NN
# Write user-deinfed function of weighted k-NN
# Use imported distance functions implemented by sklearn
def wknn(trainX,trainY,testX,k,dist=euclidean_dist):
    ######## Weighted K-NN Classification ########
    # INPUT 
    # trainX: training input dataset, n by p size 2-D array
    # trainY: training output target, 1-D array with length of n
    # testX: test input dataset, m by p size 2-D array
    # k: the number of the nearest neighbors
    # dist: distance measure function
    # OUTPUT
    # y_pred: predicted output target of testX, 1-D array with length of m
    #         When tie occurs, the final class is select in alpabetical order
    #         EX) if "A" ties "B", select "A" and if "2" ties "4", select 2
    
    # TODO: weighted k-NN classification
    distance=[]        
    y_pred = []
    distance = dist(testX,trainX)
    
    classes = []
    weight = []
    for i in range(len(testX)):
        weight.append(1/np.sort(distance[i])[:k])
    for l in range(len(testX)):
        classes.append(trainY[np.argsort(distance[l])[:k]])
    
    for n in range(len(testX)):
        class1 = 0
        class2 = 0
        for m in range(k):
            if classes[n][m] == 1:
                class1 += weight[n][m]
            else:
                class2 += weight[n][m]
        if class2>=class1:
            y_pred.append(0)
        else:
            y_pred.append(1)
    return y_pred


# TODO: Calculate accuracy of test set 
#       with varying the number neareset neighbors (k) and distance metrics
#       using weighted k-NN

wpred3e = wknn(trainX,trainY,testX,3)
wpred5e = wknn(trainX,trainY,testX,5)
wpred7e = wknn(trainX,trainY,testX,7)
wpred3m = wknn(trainX,trainY,testX,3,dist=manhattan_dist)
wpred5m = wknn(trainX,trainY,testX,5,dist=manhattan_dist)
wpred7m = wknn(trainX,trainY,testX,7,dist=manhattan_dist)

wcount = 0
wcount1 = 0
wcount2 = 0
wcount3 = 0
wcount4 = 0
wcount5 = 0
for i in range(len(testX)):
    if wpred3e[i] == testY[i]:
        wcount +=1

for i in range(len(testX)):
    if wpred5e[i] == testY[i]:
        wcount1 += 1
        
for i in range(len(testX)):
    if wpred7e[i] == testY[i]:
        wcount2 += 1

for i in range(len(testX)):
    if wpred3m[i] == testY[i]:
        wcount3 += 1

for i in range(len(testX)):
    if wpred5m[i] == testY[i]:
        wcount4 += 1

for i in range(len(testX)):
    if wpred7m[i] == testY[i]:
        wcount5 += 1
waccuracy3e = wcount/len(testY)
waccuracy5e = wcount1/len(testY)
waccuracy7e = wcount2/len(testY)
waccuracy3m = wcount3/len(testY)
waccuracy5m = wcount4/len(testY)
waccuracy7m = wcount5/len(testY)



strainX = np.zeros(np.shape(trainX))
stestX = np.zeros(np.shape(testX))
trainXmean = np.mean(trainX,axis=0)
testXmean = np.mean(testX,axis=0)
trainXsd = np.sqrt(np.var(trainX,axis=0))
testXsd = np.sqrt(np.var(testX,axis=0))
    
for i in range(len(trainX)):
    for j in range(10):
        strainX[i][j] = (trainX[i][j]-trainXmean[j])/trainXsd[j]
for i in range(len(testX)):
    for j in range(10):
        stestX[i][j] = (testX[i][j]-testXmean[j])/testXsd[j]
        


spred3e = knn(strainX,trainY,stestX,3)
spred5e = knn(strainX,trainY,stestX,5)
spred7e = knn(strainX,trainY,stestX,7)
spred3m = knn(strainX,trainY,stestX,3,dist=manhattan_dist)
spred5m = knn(strainX,trainY,stestX,5,dist=manhattan_dist)
spred7m = knn(strainX,trainY,stestX,7,dist=manhattan_dist)
scount = 0
scount1 = 0
scount2 = 0
scount3 = 0
scount4 = 0
scount5 = 0

for i in range(len(testX)):
    if spred3e[i] == testY[i]:
        scount +=1

for i in range(len(testX)):
    if spred5e[i] == testY[i]:
        scount1 += 1
        
for i in range(len(testX)):
    if spred7e[i] == testY[i]:
        scount2 += 1

for i in range(len(testX)):
    if spred3m[i] == testY[i]:
        scount3 += 1

for i in range(len(testX)):
    if spred5m[i] == testY[i]:
        scount4 += 1

for i in range(len(testX)):
    if spred7m[i] == testY[i]:
        scount5 += 1
saccuracy3e = scount/len(testY)
saccuracy5e = scount1/len(testY)
saccuracy7e = scount2/len(testY)
saccuracy3m = scount3/len(testY)
saccuracy5m = scount4/len(testY)
saccuracy7m = scount5/len(testY)



swpred3e = wknn(strainX,trainY,stestX,3)
swpred5e = wknn(strainX,trainY,stestX,5)
swpred7e = wknn(strainX,trainY,stestX,7)
swpred3m = wknn(strainX,trainY,stestX,3,dist=manhattan_dist)
swpred5m = wknn(strainX,trainY,stestX,5,dist=manhattan_dist)
swpred7m = wknn(strainX,trainY,stestX,7,dist=manhattan_dist)

swcount = 0
swcount1 = 0
swcount2 = 0
swcount3 = 0
swcount4 = 0
swcount5 = 0
for i in range(len(testX)):
    if swpred3e[i] == testY[i]:
        swcount +=1

for i in range(len(testX)):
    if swpred5e[i] == testY[i]:
        swcount1 += 1
        
for i in range(len(testX)):
    if swpred7e[i] == testY[i]:
        swcount2 += 1

for i in range(len(testX)):
    if swpred3m[i] == testY[i]:
        swcount3 += 1

for i in range(len(testX)):
    if swpred5m[i] == testY[i]:
        swcount4 += 1

for i in range(len(testX)):
    if swpred7m[i] == testY[i]:
        swcount5 += 1
swaccuracy3e = swcount/len(testY)
swaccuracy5e = swcount1/len(testY)
swaccuracy7e = swcount2/len(testY)
swaccuracy3m = swcount3/len(testY)
swaccuracy5m = swcount4/len(testY)
swaccuracy7m = swcount5/len(testY)
"""


"""
#TODO: k-means clustering
#TODO: B - k-means clustering
# Write user-defined function of k-means clustering
def kmeans(X,k=2,max_iter=300):
    ############ K-MEANS CLUSTERING ##########
    # INPUT
    # X: n by p array (n=# of observations, p=# of input variables)
    # k: the number of clusters
    # max_iter: the maximum number of iteration
    # OUTPUT
    # label: cluster label (len(label)=n)
    # centers: cluster centers (k by p)
    ##########################################
    # If average distance between old centers and new centers is less than 0.000001, stop
    
    # TODO: k-means clustering
   
    center1 = []
    center2 = []
    
    center1.append(X[6624])
    center2.append(X[17551])
    
    label = []
    centers = []
    iteration = 0
    while(iteration<=max_iter):
        for i in range(len(X)):
            cluster1 = []
            cluster2 = []
            if plt.mlab.dist(X[i],center1[i]) <= plt.mlab.dist(X[i],center2[i]):
                label.append('cluster1')
                cluster1.append(X[i])
            else:
                label.append('cluster2')
                cluster2.append(X[i])
        
        
            center1.append(np.mean(cluster1,axis=0))
            center2.append(np.mean(cluster2,axis=0))
        
        if iteration>3 and plt.mlab.dist(center1[len(center1)-1],center1[len(center1)-2]) <0.00001:
            break
        
        iteration += 1
       
        label.clear()
    centers.append(center1[len(center1)-1])
    centers.append(center2[len(center2)-1])
    return (label, centers)

label,centers = kmeans(X)
"""
# TODO: Calculate centroids of two clusters



#TODO: C - homogeneity and completeness 
kmeans = KMeans(n_clusters=2).fit(X)
centers2 = kmeans.cluster_centers_
label2 = kmeans.predict(X)
class1 = np.count_nonzero(y)
class0 = len(X)-class1
H = len(X)

cluster1 = 0
cluster2 = 0
for i in range(len(label2)):
    if label2[i] ==0:
        cluster1 += 1
    else:
        cluster2 += 1
count1 = 0
count2 = 0
count3 = 0
count4 = 0
for i in range(len(X)):
    if label2[i] == 0 and y[i] ==1:
        count1 += 1
    elif label2[i] == 1 and y[i] == 1:
        count2 += 1
    elif label2[i] == 0 and y[i] == 0:
        count3 += 1
    elif label2[i] == 1 and y[i] == 0:
        count4 += 1
Hc = -((class1/H*(np.log(class1/H)))+(class0/H*(np.log(class0/H))))   
Hck = -((count1/H*np.log(count1/cluster1))+(count2/H*np.log(count2/cluster2))+(count3/H*np.log(count3/cluster1))+(count4/H*np.log(count4/cluster2)))  
Homogeneity = 1-Hck/Hc

    
Hk = -((cluster1/H*np.log(cluster1/H))+(cluster2/H*np.log(cluster2/H)))   
Hkc = -((count1/H*np.log(count1/class1))+(count2/H*np.log(count2/class1))+(count3/H*np.log(count3/class0))+(count4/H*np.log(count4/class0))) 
Completeness = 1-Hkc/Hk    



    