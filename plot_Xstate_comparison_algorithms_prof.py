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
size        = 300
while (sum(ydata) < size):
    colnames    = ['theta','phi','psi','x','y','u','v']  #col names for pandas
    raw         = np.random.random(size = [1,7])
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
    Xdata. extend (df_Xdata.values )
    ydata. extend (  list (map(lambda x: 0 if x >= 0 else 1, max_x_y - minG_H ) ) )

# comment if not to save to file - they are splitted below
np.savetxt('Xdata.txt', Xdata, delimiter = '    ')
np.savetxt('ydata.txt', ydata, delimiter = '    ')
#
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
print('proceeding to ML part             :D')
print('------------------------------------')

# list of indices of Xtrain
ind = list(range(len(Xtrain)))

# number of avarages 
nmed = 5    
for gnb in classifiers:
    ti = datetime.now()
    # this is for visual progress of the execution
    print('------------------------------------')
    print(gnb)
    print('------------------------------------')
      
    meanacc     = []
    accvssize   = []
    # some SVM algorithms have bad or impossible performance for small training set
    # so we start around 20
    for ntrset in range(100,len(Xtrain)):
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

plt.xlabel('Training Set Size')
plt.ylabel('Output Layer Accuracy')

plt.legend(names)
plt.plot()
#
##
print ('job duration in s: ', datetime.now() - start)
