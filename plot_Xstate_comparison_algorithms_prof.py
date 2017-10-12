#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Askery Canabarrro

Comparison of sklearn classification algorithms for 2 qubits Xstate
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
import random
import matplotlib.pyplot as plt
import os
from sklearn import svm
from sklearn.svm import NuSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#

# names of the methods we gonna test (for plotting usage)
names = [
        "MLP",
        "Linear SVC",
        "RBF SVC",
        "NuSVC",
        "AdaBoost",
        "DecisionTree",
         "RandomForestClassifier"
         ]
# definition of the methods in [names] (for computational usage)
classifiers = [
    MLPClassifier(solver='lbfgs', alpha=1e-5),
    svm.SVC(),    
    svm.SVC(kernel="linear", C=0.025),
    NuSVC(),
    AdaBoostClassifier(),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    ]


# --------------------------------------------------------------------------
# \begin{creation of the data - Xdata and ydata
#
Xdata       = []
ydata       = []
size        = 10000
while (sum(ydata) < size):
    #colnames    = ['theta','phi','psi','x','y','u','v']  #col names for pandas
    theta = np.random.random()*(np.pi/2)
    phi = np.random.random()*(np.pi/2)
    psi = np.random.random()*(np.pi/2)
    x = np.random.random()
    y = np.random.random()
    u = np.random.random()*(2*np.pi)
    v        = np.random.random()*(2*np.pi)
    #
    G = np.sin(theta)**4 * np.cos(phi)**2 * np.sin( phi )**2 * np.cos(psi)**2
    #
    #df_B = np.sin(df_raw['theta'])**2 * ( 1 -  np.sin(df_raw['phi'])**2 * np.sin( df_raw['psi'] )**2 )        
    #  
    H = np.sin(theta)**2 * np.cos(theta)**2 *np.sin( phi )**2 * np.sin( psi )**2

    #
    x = x*H
    y = y*G
    #
    Xdata. extend ( [np.array([theta,phi,psi,x,y,u,v])] )

    max_x_y = np.maximum(x,y)
    minG_H =  np.minimum(G,H)
    
    if ( max_x_y - minG_H ) >= 0:
        ydata. append (0)
    else:
        ydata. append (1)
#\end{creation of raw data}
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
"""
This part makes the data equally distributated in terms of the binary 
classification, i.e., equal number of entangled and separable states.
Comment all if you do not care! But if you pick a number (0 or 1) you will get 50%
error score if sets are equally distributed. If not, say, if you have 80% of 
nonentangled instances, you may get a score of 80% just by always saying "0"  
"""
allind = list(range(len(Xdata)))
entind = [i for i, j in enumerate(ydata) if j == 1 ]

def returnNotMatches(a, b):
    a = set(a)
    b = set(b)
    return list(a - b)

nenind = returnNotMatches(allind,entind)

n = random.sample(nenind, len(entind))

eqdind = entind + n
random.shuffle(eqdind)

Xdata = [Xdata[i] for i in eqdind]
ydata = [ydata[i] for i in eqdind]

# comment if not to save to file - they are splitted below
np.savetxt('Xdata.txt', Xdata, delimiter = '    ')
np.savetxt('ydata.txt', ydata, delimiter = '    ')
#
# --------------------------------------------------------------------------


# \begin{split data in Train and Test}
#
Xtrain, Xtest, ytrain, ytest = train_test_split(Xdata, ydata, test_size=0.33, random_state=42)

# comment if not to save to file
np.savetxt('Xtrain.txt', Xtrain, delimiter = '    ')
np.savetxt('ytrain.txt', ytrain, delimiter = '    ')
np.savetxt('Xtest.txt', Xtest, delimiter = '    ')
np.savetxt('ytest.txt', ytest, delimiter = '    ')
#
# \end{split}

print('------------------------------------')
print('Train and test sets generated     :D')
print('------------------------------------')
print('elapsed time to create data: ', datetime.now() - start )
print('------------------------------------')
print('proceeding to ML part             :D')
print('------------------------------------')

# list of indices of Xtrain
ind = list(range(len(Xtrain)))

# number of avarages 
nmed = 1
min_trsize = 50
trset_inc  = 100  
for gnb,name in zip(classifiers,names):
    ti = datetime.now()
    # this is for visual progress of the execution
    print('------------------------------------')
    print(gnb)
    print('------------------------------------')
      
    meanacc     = []
    accvssize   = []
    # some SVM algorithms have bad or impossible performance for small training set
    # so we start around 20
    for ntrset in range(min_trsize,len(Xtrain),trset_inc):
        # training set generation
        for j in range(nmed):
            # this picks n indices from Xtrain - it is a list!
            #n = list (map (lambda _: random.choice(ind), range(ntrset)))
            n = random.sample(ind, ntrset)
            
            # this takes the elements from Xtrain and ytrain for previous list n
            Xtrainprime = [Xtrain[i] for i in n]
            ytrainprime = [ytrain[i] for i in n]
            
            # Train the classifier
            model = gnb.fit(Xtrainprime, ytrainprime)  
            
            # Make predictions
            preds = gnb.predict(Xtest)
            #print(len(preds))
            
            # Evaluate accuracy
            meanacc.append(accuracy_score(ytest, preds))
    
        #Evaluate mean accuracy
        accvssize.append([ntrset,sum(meanacc)/len(meanacc)])
        
        #
        #with open("acc_vs_trsize.csv", "a") as f:
        #    f.write(str(ntrset) + ',' + str(sum(meanacc)/len(meanacc)) + "\n" )
    
        if ntrset % 50 == 0:
            print('step ',ntrset)
                      
    out = pd.DataFrame(np.array(accvssize), columns = ['c1','c2'])
    
    xx = out['c1']
    yy = out['c2']
    plt.plot(xx,yy)
    
    # print job duration for each classifier
    print ('duration: ', datetime.now() - ti)
    folder  =  name 
    #folder  = "/out/"+str(name) 
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = str(folder) + "/" + name + ".txt"    
    np.savetxt(path, accvssize, delimiter = '    ')

    #print (yy)

plt.xlabel('Training Set Size')
plt.ylabel('Output Layer Accuracy')

plt.legend(names)
plt.plot()
#
##
print ('job duration in s: ', datetime.now() - start)
