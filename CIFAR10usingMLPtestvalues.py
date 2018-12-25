
# coding: utf-8

# In[ ]:


import numpy as np
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


def unpickle(file):
 '''Load byte data from file'''
 with open(file, 'rb') as f:
  data = pickle.load(f, encoding='latin-1')
  return data


# In[4]:


from keras.datasets import cifar10
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()


# In[5]:


print('X_train shape:', X_train.shape)
print('y_train shape:', Y_train.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', Y_test.shape)


# In[ ]:


X_train_unshuf_unscaled = X_train


# In[7]:


X_train_unshuf_reshaped_unscaled=X_train_unshuf_unscaled.reshape(50000,32*3*32)
X_train_unshuf_reshaped_unscaled


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val  = train_test_split(X_train_unshuf_reshaped_unscaled, Y_train, test_size = 1/10, random_state = 42)


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train.astype(np.float32))


# In[ ]:


X_val_s = scaler.fit_transform(X_val.astype(np.float32))
y_train = y_train.astype(np.int32)


# In[ ]:


import tensorflow as tf


# In[ ]:


from sklearn.metrics import accuracy_score


# In[13]:


#learning curve

Train_size_list_10epfix = [50000]
Train_error_list_10epfix = []
Val_error_list_10epfix = []
Train_time_list_10epfix = []
Val_time_list_10epfix = []

config = tf.contrib.learn.RunConfig(tf_random_seed=42) 
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train_s)
    
dnn_1_clf = tf.contrib.learn.DNNClassifier(activation_fn=tf.nn.relu, hidden_units=[7000,6000], n_classes=10, feature_columns=feature_columns, config=config)
dnn_1_clf = tf.contrib.learn.SKCompat(dnn_1_clf) #### to be compatible with sklearn
    

from datetime import datetime
start=datetime.now()
dnn_1_clf.fit(X_train_s, y_train, batch_size=200, steps=(5*1100))
Train_elapse = (datetime.now()-start).total_seconds()
print('dnn_1_clf training time', Train_elapse)
    
y_train_pred = dnn_1_clf.predict(X_train_s) #return dictionary
Train_error = 1 - accuracy_score(y_train, y_train_pred['classes'])

    
print(type(y_train_pred))

start=datetime.now()
y_val_pred =  dnn_1_clf.predict(X_val_s)
    
Val_elapse = (datetime.now()-start).total_seconds()

Val_time_list_10epfix.append(Val_elapse)
Val_error = 1 - accuracy_score(y_val, y_val_pred['classes'])
      
    
print('Train error, Val_error',  Train_error, '/', Val_error)
                                   
Train_error_list_10epfix.append(Train_error)
Val_error_list_10epfix.append(Val_error)
    
print("Train_size: ", Train_size_list_10epfix)
print("Train_error: ", Train_error_list_10epfix)
print("Val_error: ", Val_error_list_10epfix)
print("Train_time: ", Train_elapse)
print("Val_time: ", Val_elapse)  

valaccuracy = accuracy_score(y_val, y_val_pred['classes'])
trainaccuracy = accuracy_score(y_train, y_train_pred['classes'])
print('Validation error', Val_error)
print('Train error', Train_error)
print('Val Accuracy', valaccuracy)
print('Train Accuracy', trainaccuracy)


# 
# 
# ```
# BEST NEURON SIZE = [7000,6000]
# ACTIVATION FUNCTION = RELU
# VAL ERROR = 0.4134
# VAL ACCURACY = 0.5866
# TRAIN ACCURACY = 1
# ```
# 
# 

# **FIXING THE VALUES AND FITTING IT TO THE MODEL**

# In[14]:


config = tf.contrib.learn.RunConfig(tf_random_seed=42)
# set the random seed for tensorflow initializers(for consistency between reruns)

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train_s)
dnn_clf_opt = tf.contrib.learn.DNNClassifier(hidden_units=[7000,6000], n_classes=10,
                                         feature_columns=feature_columns, config=config)
dnn_clf_opt = tf.contrib.learn.SKCompat(dnn_clf_opt) # if TensorFlow >= 1.1
dnn_clf_opt.fit(X_train_s, y_train, batch_size=200, steps=10*1100)


# **THE FINAL TEST ACCURACY**

# In[15]:


X_test_reshaped = X_test.reshape(10000,3072)
X_test_s = scaler.fit_transform(X_test_reshaped.astype(np.float32))
y_test_pred = dnn_clf_opt.predict(X_test_s)
accuracy_score(Y_test, y_test_pred['classes'])


# In[16]:


type(y_test_pred['classes'])


# In[17]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(classification_report(Y_test,y_test_pred['classes']))


# **CONFUSION MATRIX**

# In[18]:


conf_mx_DNN_1L = confusion_matrix(Y_test, y_test_pred['classes'])



row_sum = np.sum(conf_mx_DNN_1L, axis =1, keepdims = True)
norm_conf_mx_DNN_1L = conf_mx_DNN_1L / row_sum
norm_conf_mx_DNN_1L.shape


# In[19]:


np.fill_diagonal(norm_conf_mx_DNN_1L, 0)
plt.matshow(norm_conf_mx_DNN_1L, cmap = plt.cm.Reds)

print(norm_conf_mx_DNN_1L)

