#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 09:04:29 2017

@author: Askery Canabarrro
"""
print(__doc__)
#from ___future__ import division

from datetime import datetime
start=datetime.now()

print('####################################')
print('program running...')

#packages
import numpy as np
import pandas as pd
import csv
import random
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
#from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
#from matplotlib.colors import ListedColormap
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn import svm
#from sklearn.gaussian_process import GaussianProcessClassifier
#from sklearn.gaussian_process.kernels import RBF
#from sklearn.preprocessing import StandardScaler
#from sklearn.datasets import make_moons, make_circles, make_classification
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# names of the methods we gonna test (for plotting usage)
names = [
        "MLP",
        #"RBF SVC",
        #"Linear SVC",
        "AdaBoost",
        # "Naive Bayes", 
         #"BernoulliNB",
         #"MultinomialNB",
         "Decision Tree",
         "RandomForestClassifier"
         ]
# definition of the methods in [names] (for computational usage)
classifiers = [
    MLPClassifier(solver='lbfgs', alpha=1e-5),
    #svm.SVC(),    
    #SVC(kernel="linear", C=0.025),
    AdaBoostClassifier(),
    #GaussianNB(),
    #BernoulliNB(),
    #MultinomialNB(),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    ]


# --------------------------------------------------------------------------
# \begin{creation of the data - Xdata and ydata
#
datasize    = 3000
colnames    = ['theta','phi','psi','x','y','u','v']  #col names for pandas
raw         = np.random.random(size = [datasize,7])
df_raw      = pd.DataFrame(raw, columns = colnames)
#print(df_raw)
df_raw['theta'] = df_raw['theta']*(np.pi/2)
df_raw['phi'] = df_raw['phi']*(np.pi/2)
df_raw['psi'] = df_raw['psi']*(np.pi/2)
df_raw['x'] = df_raw['x']
df_raw['y'] = df_raw['y']
df_raw['u'] = df_raw['u']*(2*np.pi)
df_raw['v'] = df_raw['v']*(2*np.pi)
#print(df_raw)
#
df_G = np.sin(df_raw['theta'])**4 * np.cos(df_raw['phi'])**2 * \
        np.sin( df_raw['phi'] )**2 * np.cos(df_raw['psi'])**2
#
#df_B = np.sin(df_raw['theta'])**2 * ( 1 -  np.sin(df_raw['phi'])**2 * np.sin( df_raw['psi'] )**2 )        
#  
df_H = np.sin(df_raw['theta'])**2 * np.cos(df_raw['theta'])**2 * \
        np.sin( df_raw['phi'] )**2 * np.sin(df_raw['psi'])**2

#
df_raw['x'] = df_raw['x']*df_H
df_raw['y'] = df_raw['y']*df_G
 
#
x = np.array (df_raw['x'])
y = np.array (df_raw['y'])
G = np.array(df_G)
H = np.array(df_H)
max_x_y = np.maximum(x,y)
minG_H =  np.minimum(G,H)
#     
df_Xdata= df_raw
Xdata = list (df_Xdata.values)
ydata = list ( map(lambda x: 0 if x >= 0 else 1, max_x_y - minG_H ) )
#ydata = list ( map(lambda x: 1 if x >= 0 else 0, max_x_y - minG_H ) )

# comment if not to save to file
np.savetxt('Xdata.txt', Xdata, delimiter = '    ')
np.savetxt('ydata.txt', ydata, delimiter = '    ')


#\end{creation of raw data}
# --------------------------------------------------------------------------

# \begin{split data in Train and Test}
Xtrain, Xtest, ytrain, ytest = train_test_split(Xdata, ydata, test_size=0.33, random_state=42)
# \end{split}

print('------------------------------------')
print('Train and test sets generated     :D')
print('------------------------------------')
print('proceeding to ML part             :D')
print('------------------------------------')

# list of indices of Xtrain
ind = list(range(len(Xtrain)))

# number of avarages
nmed = 5    
for gnb in classifiers:
    print('------------------------------------')
    print(gnb)
    print('------------------------------------')
    
   
    
    meanacc     = []
    accvssize   = []
    for ntrset in range(1,51):
        # training set generation
        for j in range(nmed):
            # this picks n indices from Xtrain - it is a list!
            #n = list (map (lambda _: random.choice(ind), range(ntrset)))
            n = random.sample(ind, ntrset)
            
            # this takes the elements from Xtrain and ytrain for previous list n
            Xtrainprime = [Xtrain[i] for i in n]
            ytrainprime = [ytrain[i] for i in n]

            train_data      = Xtrainprime
            train_target    = ytrainprime
            test_data       = Xtest
            test_target     = ytest
            
            # Train the classifier
            model = gnb.fit(train_data, train_target)  
            
            # Make predictions
            preds = gnb.predict(test_data)
            #print(len(preds))
            
            # from sklearn.metrics import accuracy_score    
            # Evaluate accuracy
            #print(j,accuracy_score(test_target, preds))
            meanacc.append(accuracy_score(test_target, preds))
    
    #Evaluate mean accuracy
    #print(meanacc)
    
        #print(ntrset,sum(meanacc)/len(meanacc))
        accvssize.append([ntrset,sum(meanacc)/len(meanacc)])
        
        #
        #with open("acc_vs_trsize.csv", "a") as f:
        #    f.write(str(ntrset) + ',' + str(sum(meanacc)/len(meanacc)) + "\n" )
    
        if ntrset % 50 == 0:
            print('step ',ntrset)
            
#    with open("output.csv", "w+") as out:
#            writer = csv.writer(out)
#            writer.writerows(accvssize)
          
    out = pd.DataFrame(np.array(accvssize), columns = ['c1','c2'])
    #print(out)
    
    xx = out['c1']
    yy = out['c2']
    plt.plot(xx,yy)

plt.xlabel('Training Set Size')
plt.ylabel('Output Layer Accuracy')

plt.legend(names)
plt.plot()
#
##
print ('job duration in s: ', datetime.now() - start)