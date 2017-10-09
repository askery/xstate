#packages
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn import metrics

#

# --------------------------------------------------------------------------
# \begin{creation of the data - Xdata and ydata
#
Xdata       = []
ydata       = []
size        = 30000
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
#np.savetxt('Xdata.txt', Xdata, delimiter = '    ')
#np.savetxt('ydata.txt', ydata, delimiter = '    ')
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
x_train, x_test, y_train, y_test = train_test_split(Xdata, ydata, test_size=0.33, random_state=42)
# \end{split}

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

X_FEATURE = 'x'  # Name of the input feature.


def main(unused_argv):

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  feature_columns = [
      tf.feature_column.numeric_column(
          X_FEATURE, shape=np.array(x_train).shape[1:])]
  classifier = tf.estimator.DNNClassifier(
      feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=2)

  # Train.
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={X_FEATURE: x_train}, y=y_train, num_epochs=None, shuffle=True)
  classifier.train(input_fn=train_input_fn, steps=10000)

  # Predict.
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={X_FEATURE: x_test}, y=y_test, num_epochs=1, shuffle=False)
  predictions = classifier.predict(input_fn=test_input_fn)
  y_predicted = np.array(list(p['class_ids'] for p in predictions))
  y_predicted = y_predicted.reshape(np.array(y_test).shape)

  # Score with sklearn.
  score = metrics.accuracy_score(y_test, y_predicted)
  print('Accuracy (sklearn): {0:f}'.format(score))

  # Score with tensorflow.
  scores = classifier.evaluate(input_fn=test_input_fn)
  print('Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))


if __name__ == '__main__':
  tf.app.run()