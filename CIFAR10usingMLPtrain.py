
# coding: utf-8

# In[3]:


import numpy as np
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # IMPORTING THE CIFAR-10 DATASET

# In[4]:


def unpickle(file):
 '''Load byte data from file'''
 with open(file, 'rb') as f:
  data = pickle.load(f, encoding='latin-1')
  return data


def load_cifar10_data(data_dir):
 '''Return train_data, train_labels, test_data, test_labels
 The shape of data is 32 x 32 x3'''
 train_data = None
 train_labels = []

 for i in range(1, 6):
  data_dic = unpickle(data_dir + "/data_batch_{}".format(i))
  if i == 1:
   train_data = data_dic['data']
  else:
   train_data = np.vstack((train_data, data_dic['data']))
  train_labels += data_dic['labels']

 test_data_dic = unpickle(data_dir + "/test_batch")
 test_data = test_data_dic['data']
 test_labels = test_data_dic['labels']

 train_data = train_data.reshape((len(train_data), 3, 32, 32))
 train_data = np.rollaxis(train_data, 1, 4)
 train_labels = np.array(train_labels)

 test_data = test_data.reshape((len(test_data), 3, 32, 32))
 test_data = np.rollaxis(test_data, 1, 4)
 test_labels = np.array(test_labels)

 return train_data, train_labels, test_data, test_labels

data_dir = 'cifar-10-batches-py'

train_data, train_labels, test_data, test_labels = load_cifar10_data(data_dir)

print(train_data.shape)
print(train_labels.shape)

print(test_data.shape)
print(test_labels.shape)

# In order to check where the data shows an image correctly
plt.imshow(train_data[2])
plt.show()

X_train_data = train_data
Y_train_labels = train_labels

X_test_data = test_data
Y_test_labels = test_labels


# In[5]:


print(type(X_train_data), X_train_data)


# In[6]:


print(type(test_data), test_data)


# In[7]:


print('X_train_data', type(X_train_data), X_train_data.shape)


# In[8]:


print('Y_train_labels', type(Y_train_labels), Y_train_labels.shape)


# In[9]:


print('test_data', type(test_data), test_data.shape)


# In[10]:


print('test_labels', type(test_labels), test_labels.shape)


# In[11]:


print(type(Y_train_labels), Y_train_labels)


# # CHECKING THE DISTRIBUTION OF THE TRAIN AND TEST DATA

# In[12]:


class_train, counts_train = np.unique(Y_train_labels, return_counts = True)

distribution_train = dict(zip(class_train, counts_train))
print(distribution_train )


# In[13]:


plt.bar(list(distribution_train.keys()),distribution_train.values(),width =0.5)
plt.xlabel('Class Distribution')
plt.ylabel('Counts')
plt.show()


# In[14]:


class_test, counts_test = np.unique(Y_test_labels, return_counts = True)

distribution_test = dict(zip(class_test, counts_test))
print(distribution_test)


# In[15]:


plt.bar(list(distribution_test.keys()),distribution_test.values(),width =0.5)
plt.xlabel('Class distribution')
plt.ylabel('Counts')
plt.show()


# In[17]:


X_train_unshuf_unscaled = X_train_data
X_test = test_data
X_train_data.shape


# # RESHAPING THE DATA

# In[18]:


X_train_unshuf_reshaped_unscaled=X_train_unshuf_unscaled.reshape(50000,32*3*32)
X_test_reshaped = X_test.reshape(10000,3072)
print(X_train_unshuf_reshaped_unscaled.shape)
X_test_reshaped.shape


# # SPLITTING THE DATA

# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val  = train_test_split(X_train_unshuf_reshaped_unscaled, Y_train_labels, test_size = 1/10, random_state = 42)


# In[20]:


X_train.shape


# In[21]:


X_val.shape


# In[22]:


y_train.shape


# In[23]:


y_val.shape


# # SCALING THE DATA

# In[24]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train.astype(np.float32))


# In[25]:


X_val_s = scaler.fit_transform(X_val.astype(np.float32))


# In[26]:


plt.imshow(X_train[1].reshape(32,32,3), interpolation = 'nearest') #plt.axis("off")
plt.show()


# In[27]:


plt.imshow(X_train_s[1].reshape(32,32,3), interpolation = 'nearest')
plt.show()


# In[36]:


fig_object, ax_object = plt.subplots(1, 10, figsize=(12,5))
ax_object = ax_object.reshape(10,)
      
for i in range(len(ax_object)):
    ax = ax_object[i]
    ax.imshow(X_train[i].reshape(32,32,3), interpolation = 'nearest')
    ax.set_title(i)       
plt.show()

fig_object, ax_object = plt.subplots(1, 10, figsize=(12,5))
ax_object = ax_object.reshape(10,)
      
for i in range(len(ax_object)):
    ax = ax_object[i]
    ax.imshow(X_train_s[i].reshape(32,32,3), interpolation = 'nearest')
    ax.set_title(i)       
plt.show()


# In[29]:


import tensorflow as tf


# In[30]:


from sklearn.metrics import accuracy_score


# # Performace of Unscaled [just for comparison]

# In[239]:


#learning curve

Nron_size_list_Unsc = [500]

config = tf.contrib.learn.RunConfig(tf_random_seed=42) 
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
    
dnn_1_clf = tf.contrib.learn.DNNClassifier(activation_fn=tf.nn.relu, hidden_units=[500], n_classes=10, feature_columns=feature_columns, config=config)
dnn_1_clf = tf.contrib.learn.SKCompat(dnn_1_clf) #### to be compatible with sklearn
    

from datetime import datetime
start=datetime.now()
dnn_1_clf.fit(X_train_s, y_train, batch_size=200, steps=(1100))
Train_elapse = (datetime.now()-start).total_seconds()
print('dnn_1_clf training time', Train_elapse)
Train_time_list_1ep.append(Train_elapse)
    
y_train_pred = dnn_1_clf.predict(X_train) #return dictionary
Train_error = 1 - accuracy_score(y_train, y_train_pred['classes'])

    
print(type(y_train_pred))

start=datetime.now()
y_val_pred =  dnn_1_clf.predict(X_val)
    
Val_elapse = (datetime.now()-start).total_seconds()

Val_time_list_1ep.append(Val_elapse)
Val_error = 1 - accuracy_score(y_val, y_val_pred['classes'])
      
    
print('Train error, Val_error',  Train_error, '/', Val_error)
                                   
accuracy = accuracy_score(y_val, y_val_pred['classes'])
    
print("Nron_size_list_Unsc: ", Nron_size_list_Unsc)
print('Val_error', Val_error)
print('Train_error', Train_error)
print('Val_accuracy', accuracy)


# In[248]:


Nron_size_list_Unsc_s = [500]

config = tf.contrib.learn.RunConfig(tf_random_seed=42) 
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train_s)
    
dnn_1_clf = tf.contrib.learn.DNNClassifier(activation_fn=tf.nn.relu, hidden_units=[500], n_classes=10, feature_columns=feature_columns, config=config)
dnn_1_clf = tf.contrib.learn.SKCompat(dnn_1_clf) #### to be compatible with sklearn
    

from datetime import datetime
start=datetime.now()
dnn_1_clf.fit(X_train_s, y_train, batch_size=200, steps=(1100))
Train_elapse = (datetime.now()-start).total_seconds()
print('dnn_1_clf training time', Train_elapse)
Train_time_list_1ep.append(Train_elapse)
    
y_train_pred = dnn_1_clf.predict(X_train_s)
                                 
Train_error_s = 1 - accuracy_score(y_train, y_train_pred['classes'])

    
print(type(y_train_pred))

start=datetime.now()
y_val_pred =  dnn_1_clf.predict(X_val_s)
    
Val_elapse_s = (datetime.now()-start).total_seconds()

Val_time_list_1ep.append(Val_elapse)
Val_error_s = 1 - accuracy_score(y_val, y_val_pred['classes'])
      
    
print('Train error_s, Val_error_s',  Train_error_s, '/', Val_error_s)
                                   
accuracy_s = accuracy_score(y_val, y_val_pred['classes'])
    
print("Nron_size_list_Unsc_s: ", Nron_size_list_Unsc_s)
print('Val_error_s', Val_error_s)
print('Train_error_s', Train_error_s)
print('Val_accuracy_s', accuracy_s)


# In[266]:


import pylab as plt

Order = [1, 2]
Values = [0.604, 0.392]

LABELS = ["Train_error_unscaled", "Train_error_scaled"]
plt.figure(figsize = (5,5))
plt.bar(Order, Values, align='center', width=0.3)
plt.xticks(Order, LABELS)
plt.xlabel('Unscaled and Scaled inputs')
plt.ylabel('Training Error')
plt.show()


# In[267]:



Order = [1, 2]
Values = [0.644, 0.492]

LABELS = ["Val_error_unscaled", "Val_error_scaled"]
plt.figure(figsize = (5,5))
plt.bar(Order, Values, align='center', width=0.3)
plt.xticks(Order, LABELS)
plt.xlabel('Unscaled and Scaled inputs')
plt.ylabel('Validation Error')
plt.show()


# # Using Scaled from here onwards

# # Observation of changing training set size for different epochs

# In[270]:


#learning curve

Train_size_list_1ep = []
Train_error_list_1ep = []
Val_error_list_1ep = []
Train_time_list_1ep = []
Val_time_list_1ep = []

from sklearn.model_selection import train_test_split

for j in [ 0.999, 0.995, 0.99,0.985,0.98, 0.975,0.97, 0.96,0.95, 0.94, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.1, 0.001, 0]:

    X_train_scaled_partial, X_noneed, y_train_partial, y_noneed  = train_test_split(X_train_s, y_train, test_size = j, random_state = 42)
    print('Train_set_size:', y_train_partial.shape[0])
    Train_size_list_1ep.append(y_train_partial.shape[0])
    
    
    config = tf.contrib.learn.RunConfig(tf_random_seed=42) 
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train_scaled_partial)
    
    dnn_1_clf = tf.contrib.learn.DNNClassifier(hidden_units=[500], n_classes=10, feature_columns=feature_columns, config=config)
    dnn_1_clf = tf.contrib.learn.SKCompat(dnn_1_clf) #### to be compatible with sklearn
    
    
    start=datetime.now()
    dnn_1_clf.fit(X_train_scaled_partial, y_train_partial, batch_size=50, steps=((1-j)*1100) )
    Train_elapse = (datetime.now()-start).total_seconds()
    print('dnn_1_clf training time', Train_elapse)
    Train_time_list_1ep.append(Train_elapse)
    
    y_train_partial_pred = dnn_1_clf.predict(X_train_scaled_partial) #return dictionary
    Train_error = 1 - accuracy_score(y_train_partial, y_train_partial_pred['classes'])

    
    print(type(y_train_partial_pred))
    
    

   
    start=datetime.now()
    y_val_pred =  dnn_1_clf.predict(X_val_s)
    
    Val_elapse = (datetime.now()-start).total_seconds()
    
    Val_time_list_1ep.append(Val_elapse)
    
    Val_error = 1 - accuracy_score(y_val, y_val_pred['classes'])
      
    
    print('Train error, Val_error',  Train_error, '/', Val_error)
                                   
    Train_error_list_1ep.append(Train_error)
    Val_error_list_1ep.append(Val_error)
    
print("Train_size_list: ", Train_size_list_1ep)
print("Train_error_list: ", Train_error_list_1ep)
print("Val_error_list: ", Val_error_list_1ep)
print("Train_time_list: ", Train_time_list_1ep)
print("Val_time_list: ", Val_time_list_1ep)


# In[272]:


#learning curve

Train_size_list_5ep = []
Train_error_list_5ep = []
Val_error_list_5ep = []
Train_time_list_5ep = []
Val_time_list_5ep = []

from sklearn.model_selection import train_test_split

for j in [ 0.999, 0.995, 0.99,0.985,0.98, 0.975,0.97, 0.96,0.95, 0.94, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.1, 0.001, 0]:

    X_train_scaled_partial, X_noneed, y_train_partial, y_noneed  = train_test_split(X_train_s, y_train, test_size = j, random_state = 42)
    print('Train_set_size:', y_train_partial.shape[0])
    Train_size_list_5ep.append(y_train_partial.shape[0])
    
    
    config = tf.contrib.learn.RunConfig(tf_random_seed=42) 
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train_scaled_partial)
    
    dnn_1_clf = tf.contrib.learn.DNNClassifier(hidden_units=[500], n_classes=10, feature_columns=feature_columns, config=config)
    dnn_1_clf = tf.contrib.learn.SKCompat(dnn_1_clf) #### to be compatible with sklearn
    
    
    start=datetime.now()
    dnn_1_clf.fit(X_train_scaled_partial, y_train_partial, batch_size=50, steps=((1-j)*1100*5) )
    Train_elapse = (datetime.now()-start).total_seconds()
    print('dnn_1_clf training time', Train_elapse)
    Train_time_list_5ep.append(Train_elapse)
    
    y_train_partial_pred = dnn_1_clf.predict(X_train_scaled_partial) #return dictionary
    Train_error = 1 - accuracy_score(y_train_partial, y_train_partial_pred['classes'])

    
    print(type(y_train_partial_pred))
    
    

   
    start=datetime.now()
    y_val_pred =  dnn_1_clf.predict(X_val_s)
    
    Val_elapse = (datetime.now()-start).total_seconds()
    
    Val_time_list_5ep.append(Val_elapse)
    
    Val_error = 1 - accuracy_score(y_val, y_val_pred['classes'])
      
    
    print('Train error, Val_error',  Train_error, '/', Val_error)
                                   
    Train_error_list_5ep.append(Train_error)
    Val_error_list_5ep.append(Val_error)
    
print("Train_size_list: ", Train_size_list_5ep)
print("Train_error_list: ", Train_error_list_5ep)
print("Val_error_list: ", Val_error_list_5ep)
print("Train_time_list: ", Train_time_list_5ep)
print("Val_time_list: ", Val_time_list_5ep)


# In[273]:


#learning curve

Train_size_list_10ep = []
Train_error_list_10ep = []
Val_error_list_10ep = []
Train_time_list_10ep = []
Val_time_list_10ep = []

from sklearn.model_selection import train_test_split

for j in [ 0.999, 0.995, 0.99,0.985,0.98, 0.975,0.97, 0.96,0.95, 0.94, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.1, 0.001, 0]:

    X_train_scaled_partial, X_noneed, y_train_partial, y_noneed  = train_test_split(X_train_s, y_train, test_size = j, random_state = 42)
    print('Train_set_size:', y_train_partial.shape[0])
    Train_size_list_10ep.append(y_train_partial.shape[0])
    
    
    config = tf.contrib.learn.RunConfig(tf_random_seed=42) 
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train_scaled_partial)
    
    dnn_1_clf = tf.contrib.learn.DNNClassifier(hidden_units=[500], n_classes=10, feature_columns=feature_columns, config=config)
    dnn_1_clf = tf.contrib.learn.SKCompat(dnn_1_clf) #### to be compatible with sklearn
    
    
    start=datetime.now()
    dnn_1_clf.fit(X_train_scaled_partial, y_train_partial, batch_size=50, steps=((1-j)*1100*10) )
    Train_elapse = (datetime.now()-start).total_seconds()
    print('dnn_1_clf training time', Train_elapse)
    Train_time_list_10ep.append(Train_elapse)
    
    y_train_partial_pred = dnn_1_clf.predict(X_train_scaled_partial) #return dictionary
    Train_error = 1 - accuracy_score(y_train_partial, y_train_partial_pred['classes'])

    
    print(type(y_train_partial_pred))
    
    

   
    start=datetime.now()
    y_val_pred =  dnn_1_clf.predict(X_val_s)
    
    Val_elapse = (datetime.now()-start).total_seconds()
    
    Val_time_list_10ep.append(Val_elapse)
    
    Val_error = 1 - accuracy_score(y_val, y_val_pred['classes'])
      
    
    print('Train error, Val_error',  Train_error, '/', Val_error)
                                   
    Train_error_list_10ep.append(Train_error)
    Val_error_list_10ep.append(Val_error)
    
print("Train_size_list: ", Train_size_list_10ep)
print("Train_error_list: ", Train_error_list_10ep)
print("Val_error_list: ", Val_error_list_10ep)
print("Train_time_list: ", Train_time_list_10ep)
print("Val_time_list: ", Val_time_list_10ep)


# In[190]:


#Drawing learning curve
plt.figure(figsize = (15,10))
plt.plot(Train_size_list_1ep, Train_error_list_1ep, 'go-', label = "Train(1Epoch)")
plt.plot(Train_size_list_1ep, Val_error_list_1ep, 'yo-', label = "Val(1Epoch)")
plt.plot(Train_size_list_5ep, Train_error_list_5ep, 'rs-', label = "Train(5Epoch)")
plt.plot(Train_size_list_5ep, Val_error_list_5ep, 'bs-', label = "Val(5Epoch)")
plt.plot(Train_size_list_10ep, Train_error_list_10ep, 'k*-', label = "Train(10Epoch)")
plt.plot(Train_size_list_10ep, Val_error_list_10ep, 'c*-', label = "Val(10Epoch)")
 

plt.xlabel('Training Set Size (Sample numbers)',fontsize=16)
plt.ylabel('Error rate',fontsize=16)
plt.legend(loc="upper right", fontsize=16)
#plt.xlim(0,10000)


# In[274]:


#Drawing train/val time v.s. training set

plt.figure(figsize = (10,8))
plt.plot(Train_size_list_1ep, Train_time_list_1ep, 'go-', label = "Train(1Epoch)")
plt.plot(Train_size_list_1ep, Val_time_list_1ep, 'yo-', label = "Val(1Epoch)")
plt.plot(Train_size_list_5ep, Train_time_list_5ep, 'rs-', label = "Train(5Epoch)")
plt.plot(Train_size_list_5ep, Val_time_list_5ep, 'bs-', label = "Val(5Epoch)")
plt.plot(Train_size_list_10ep, Train_time_list_10ep, 'k*-', label = "Train(10Epoch)")
plt.plot(Train_size_list_10ep, Val_time_list_10ep, 'c*-', label = "Val(10Epoch)")
 

plt.xlabel('Training Set Size (Sample numbers)')
plt.ylabel('Time(Seconds)')
plt.legend(loc="upper left", fontsize=10)


# #  USING THE CODE BELOW TO CHANGE VALUES OF NEURONS FOR DIFFERENT ACTIVATION FUNCTIONS TO FIND THE BEST ACTIVATION FUNCTION

# In[191]:


Nron_size_list_RELU = []
Train_error_list_RELU = []
Val_error_list_RELU = []
Train_time_list_RELU = []
Val_time_list_RELU = []

for j in [100, 150, 200, 250, 300, 350, 400, 450, 500, 1000, 1500, 2000, 3000, 4000]:

    Nron_size_list_RELU.append(j)
    print('Neuron_set_size:', j)
    
    config = tf.contrib.learn.RunConfig(tf_random_seed=42) 
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train_s)
    
    dnn_1_clf_nron = tf.contrib.learn.DNNClassifier(hidden_units=[j], n_classes=10, feature_columns=feature_columns, config=config)
    dnn_1_clf_nron = tf.contrib.learn.SKCompat(dnn_1_clf_nron) # if TensorFlow >= 1.1

    start=datetime.now()
    dnn_1_clf_nron.fit(X_train_s, y_train, batch_size=200, steps=((1-0)*1100) )
    Train_elapse = (datetime.now()-start).total_seconds()
    print('dnn_1_clf_nron training time', Train_elapse)
    Train_time_list_RELU.append(Train_elapse)
    
    y_train_pred = dnn_1_clf_nron.predict(X_train_s) #return dictionary
    Train_error = 1 - accuracy_score(y_train, y_train_pred['classes'])


   
    start=datetime.now()
    y_val_pred =  dnn_1_clf_nron.predict(X_val_s)
    
    Val_elapse = (datetime.now()-start).total_seconds()
    #print('dnn_1_clf val time', Val_elapse)
    Val_time_list_RELU.append(Val_elapse)
    
    Val_error = 1 - accuracy_score(y_val, y_val_pred['classes'])

    print('Train error, Val_error',  Train_error, '/', Val_error)
                                   
    Train_error_list_RELU.append(Train_error)
    Val_error_list_RELU.append(Val_error)
    
print("Nron_size_list_RELU: ", Nron_size_list_RELU)
print("Train_error_list_RELU: ", Train_error_list_RELU)
print("Val_error_list_RELU: ", Val_error_list_RELU)
print("Train_time_list_RELU: ", Train_time_list_RELU)
print("Val_time_list_RELU: ", Val_time_list_RELU)   


# In[192]:


Nron_size_list_SIGMO = []
Train_error_list_SIGMO = []
Val_error_list_SIGMO= []
Train_time_list_SIGMO = []
Val_time_list_SIGMO = []


for j in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 1000, 1500, 2000, 3000, 4000]:

    Nron_size_list_SIGMO.append(j)
    print('Neuron_set_size:', j)
    
    
    config = tf.contrib.learn.RunConfig(tf_random_seed=42) 
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train_s)
    
    dnn_1_clf_nron = tf.contrib.learn.DNNClassifier(hidden_units=[j], n_classes=10,activation_fn=tf.nn.sigmoid, feature_columns=feature_columns, config=config)
    dnn_1_clf_nron = tf.contrib.learn.SKCompat(dnn_1_clf_nron) # if TensorFlow >= 1.1

    start=datetime.now()
    dnn_1_clf_nron.fit(X_train_s, y_train, batch_size=200, steps=((1-0)*1100) )
    Train_elapse = (datetime.now()-start).total_seconds()
    print('dnn_1_clf_nron training time', Train_elapse)
    Train_time_list_SIGMO.append(Train_elapse)
    
    y_train_pred = dnn_1_clf_nron.predict(X_train_s) #return dictionary
    Train_error = 1 - accuracy_score(y_train, y_train_pred['classes'])

   
    start=datetime.now()
    y_val_pred =  dnn_1_clf_nron.predict(X_val_s)
    
    Val_elapse = (datetime.now()-start).total_seconds()
    #print('dnn_1_clf val time', Val_elapse)
    Val_time_list_SIGMO.append(Val_elapse)
    
    Val_error = 1 - accuracy_score(y_val, y_val_pred['classes'])
 
    
    print('Train error, Val_error',  Train_error, '/', Val_error)
                                   
    Train_error_list_SIGMO.append(Train_error)
    Val_error_list_SIGMO.append(Val_error)
    
print("Nron_size_list_SIGMO: ", Nron_size_list_SIGMO)
print("Train_error_list_SIGMO: ", Train_error_list_SIGMO)
print("Val_error_list_SIGMO: ", Val_error_list_SIGMO)
print("Train_time_list_SIGMO: ", Train_time_list_SIGMO)
print("Val_time_list_SIGMO: ", Val_time_list_SIGMO) 


# In[195]:


Nron_size_list_ELU = []
Train_error_list_ELU = []
Val_error_list_ELU = []
Train_time_list_ELU = []
Val_time_list_ELU = []


for j in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 1000, 2000, 3000, 4000]:
    Nron_size_list_ELU.append(j)
    print('Neuron_set_size:', j)
    
    config = tf.contrib.learn.RunConfig(tf_random_seed=42) 
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train_s)
    
    dnn_1_clf_nron = tf.contrib.learn.DNNClassifier(hidden_units=[j], n_classes=10,activation_fn=tf.nn.elu, feature_columns=feature_columns, config=config)
    dnn_1_clf_nron = tf.contrib.learn.SKCompat(dnn_1_clf_nron) # if TensorFlow >= 1.1
    
    start=datetime.now()
    dnn_1_clf_nron.fit(X_train_s, y_train, batch_size=200, steps=((1-0)*1100) )
    Train_elapse = (datetime.now()-start).total_seconds()
    print('dnn_1_clf_nron training time', Train_elapse)
    Train_time_list_ELU.append(Train_elapse)
    
    y_train_pred = dnn_1_clf_nron.predict(X_train_s) #return dictionary
    Train_error = 1 - accuracy_score(y_train, y_train_pred['classes'])

   
    start=datetime.now()
    y_val_pred =  dnn_1_clf_nron.predict(X_val_s)
    
    Val_elapse = (datetime.now()-start).total_seconds()
    #print('dnn_1_clf val time', Val_elapse)
    Val_time_list_ELU.append(Val_elapse)
    
    Val_error = 1 - accuracy_score(y_val, y_val_pred['classes'])

    print('Train error, Val_error',  Train_error, '/', Val_error)
                                   
    Train_error_list_ELU.append(Train_error)
    Val_error_list_ELU.append(Val_error)
    
print("Nron_size_list_ELU: ", Nron_size_list_ELU)
print("Train_error_list_ELU: ", Train_error_list_ELU)
print("Val_error_list_ELU: ", Val_error_list_ELU)
print("Train_time_list_ELU: ", Train_time_list_ELU)
print("Val_time_list_ELU: ", Val_time_list_ELU)


# In[194]:


Nron_size_list_TANH = []
Train_error_list_TANH = []
Val_error_list_TANH= []
Train_time_list_TANH = []
Val_time_list_TANH = []


for j in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 1000, 2000, 3000, 4000]:
    Nron_size_list_TANH.append(j)
    print('Neuron_set_size:', j)
    

    config = tf.contrib.learn.RunConfig(tf_random_seed=42) 
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train_s)
    
    dnn_1_clf_nron = tf.contrib.learn.DNNClassifier(hidden_units=[j], n_classes=10,activation_fn=tf.nn.tanh, feature_columns=feature_columns, config=config)
    dnn_1_clf_nron = tf.contrib.learn.SKCompat(dnn_1_clf_nron) # if TensorFlow >= 1.1

    
    start=datetime.now()
    dnn_1_clf_nron.fit(X_train_s, y_train, batch_size=200, steps=((1-0)*1100) )
    Train_elapse = (datetime.now()-start).total_seconds()
    print('dnn_1_clf_nron training time', Train_elapse)
    Train_time_list_TANH.append(Train_elapse)
    
    y_train_pred = dnn_1_clf_nron.predict(X_train_s) #return dictionary
    Train_error = 1 - accuracy_score(y_train, y_train_pred['classes'])

    start=datetime.now()
    y_val_pred =  dnn_1_clf_nron.predict(X_val_s)
    
    Val_elapse = (datetime.now()-start).total_seconds()
    #print('dnn_1_clf val time', Val_elapse)
    Val_time_list_TANH.append(Val_elapse)
    
    Val_error = 1 - accuracy_score(y_val, y_val_pred['classes'])
 
    
    print('Train error, Val_error',  Train_error, '/', Val_error)
                                   
    Train_error_list_TANH.append(Train_error)
    Val_error_list_TANH.append(Val_error)
    
print("Nron_size_list_TANH: ", Nron_size_list_TANH)
print("Train_error_list_TANH: ", Train_error_list_TANH)
print("Val_error_list_TANH: ", Val_error_list_TANH)
print("Train_time_list_TANH: ", Train_time_list_TANH)
print("Val_time_list_TANH: ", Val_time_list_TANH)


# # PLOTTING ERRORS OF DIFFERENT ACTIVATION FUNCTIONS

# In[202]:


plt.figure(figsize = (15,10))
plt.plot(Nron_size_list_RELU, Train_error_list_RELU, 'go-', label = "Training(Relu)")
plt.plot(Nron_size_list_RELU, Val_error_list_RELU, 'yo-', label = "Validation(Relu)")
plt.plot(Nron_size_list_SIGMO, Train_error_list_SIGMO, 'rs-', label = "Training(Sigmoid)")
plt.plot(Nron_size_list_SIGMO, Val_error_list_SIGMO, 'bs-', label = "Validation(Sigmoid)") 
plt.plot(Nron_size_list_TANH, Train_error_list_TANH, 'k*-', label = "Training(tanh)")
plt.plot(Nron_size_list_TANH, Val_error_list_TANH, 'c*-', label = "Validation(tanh)")
plt.plot(Nron_size_list_TANH, Train_error_list_ELU, 'mp-', label = "Training(elu)")
plt.plot(Nron_size_list_TANH, Val_error_list_ELU, 'cd-', label = "Validation(elu)")

plt.xlabel('Number of Neurons',fontsize=16)
plt.ylabel('Error rate',fontsize=16)
plt.legend(fontsize=16)


# In[32]:


Nron_size_list_RELU = []
Train_error_list_RELU = []
Val_error_list_RELU = []
Train_time_list_RELU = []
Val_time_list_RELU = []

from sklearn.model_selection import train_test_split
from datetime import datetime

for i in [300, 400, 500, 600, 700]:
    
    for j in [1000, 200, 3000, 4000, 5000]: 
    
    
    
        Nron_size_list_RELU.append((i,j))
        print('Neuron_set_size:', (i,j))
         
        X_train_scaled_partial, X_noneed, y_train_partial, y_noneed  = train_test_split(X_train_s, y_train, test_size = 0, random_state = 42)
    
    
        config = tf.contrib.learn.RunConfig(tf_random_seed=42) 
        feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train_scaled_partial)
    
        dnn_1_clf_nron = tf.contrib.learn.DNNClassifier(hidden_units=[i,j], n_classes=10, feature_columns=feature_columns, config=config)
        dnn_1_clf_nron = tf.contrib.learn.SKCompat(dnn_1_clf_nron) # if TensorFlow >= 1.1
        #dnn_1_clf.fit(X_train_partial, y_train_partial, batch_size=50, max_steps=40000)
    
        start=datetime.now()
        dnn_1_clf_nron.fit(X_train_scaled_partial, y_train_partial, batch_size=50, steps=((1-0)*1100) )
        Train_elapse = (datetime.now()-start).total_seconds()
        print('dnn_1_clf_nron training time', Train_elapse)
        Train_time_list_RELU.append(Train_elapse)
    
        y_train_partial_pred = dnn_1_clf_nron.predict(X_train_scaled_partial) #return dictionary
        Train_error = 1 - accuracy_score(y_train_partial, y_train_partial_pred['classes'])

   
        start=datetime.now()
        y_val_pred =  dnn_1_clf_nron.predict(X_val_s)
    
        Val_elapse = (datetime.now()-start).total_seconds()
        
        Val_time_list_RELU.append(Val_elapse)
    
        Val_error = 1 - accuracy_score(y_val, y_val_pred['classes'])
    
        print('Train error, Val_error',  Train_error, '/', Val_error)
                                   
        Train_error_list_RELU.append(Train_error)
        Val_error_list_RELU.append(Val_error)
    
print("Nron_size_list_RELU: ", Nron_size_list_RELU)
print("Train_error_list_RELU: ", Train_error_list_RELU)
print("Val_error_list_RELU: ", Val_error_list_RELU)
print("Train_time_list_RELU: ", Train_time_list_RELU)
print("Val_time_list_RELU: ", Val_time_list_RELU)


# In[59]:


Nron_size_list_FR =  [250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 725, 750, 784, 800]
Train_error_list_FR = [0.3676222222222222, 0.36693333333333333, 0.3551777777777778, 0.3620888888888889, 0.3538444444444444, 0.3578888888888889, 0.3599555555555556, 0.3583777777777778, 0.35873333333333335, 0.35302222222222224, 0.3562444444444445, 0.35144444444444445, 0.3538888888888889, 0.3551333333333333, 0.3464222222222222, 0.3559555555555556, 0.3476666666666667, 0.35097777777777783, 0.35368888888888894, 0.3469555555555556, 0.3549777777777777, 0.3499333333333333, 0.3532888888888889]
Val_error_list_FR = [0.5065999999999999, 0.4998, 0.4878, 0.5074000000000001, 0.5042, 0.4928, 0.5034000000000001, 0.499, 0.504, 0.503, 0.5045999999999999, 0.5084, 0.5022, 0.514, 0.49739999999999995, 0.5012, 0.5048, 0.5025999999999999, 0.5107999999999999, 0.5062, 0.5096, 0.5082, 0.5096]
Train_time_list_FR:  [54.802536, 139.621559, 58.740159, 63.958351, 69.348164, 77.49446, 74.198575, 85.024884, 85.83987, 89.370873, 91.838181, 95.535692, 96.803747, 105.256697, 105.012594, 117.005753, 115.989387, 118.883818, 120.898181, 126.453666, 129.265884, 132.712105, 138.658833]
Val_time_list_FR:  [0.562429, 0.673535, 0.713596, 0.579411, 0.534378, 0.685486, 0.613436, 0.681483, 0.616437, 0.643426, 0.705727, 0.625443, 0.712505, 0.667472, 0.73152, 0.698494, 0.700497, 0.828588, 0.765545, 0.769545, 0.819579, 0.78956, 0.940666]    


# In[60]:


#plt.figure(figsize = (20,16))
plt.plot(Nron_size_list_FR, Train_error_list_FR, 'go-', label = "Training(Relu)")
plt.plot(Nron_size_list_FR, Val_error_list_FR, 'ro-', label = "Validation(Relu)")


plt.xlabel('Number of Neurons')
plt.ylabel('Error rate')
plt.legend( fontsize=12)
plt.xlim(0, 1000)


# # FIXING RELU AND PLAYING AROUND WITH HYPERPARAMETERS [explained in report]

# In[181]:


#learning curve

Train_size_list_1ep = []
Train_error_list_1ep = []
Val_error_list_1ep = []
Train_time_list_1ep = []
Val_time_list_1ep = []

config = tf.contrib.learn.RunConfig(tf_random_seed=42) 
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train_s)
    
dnn_1_clf = tf.contrib.learn.DNNClassifier(activation_fn=tf.nn.relu, hidden_units=[1000,2000,1000], n_classes=10, feature_columns=feature_columns, config=config)
dnn_1_clf = tf.contrib.learn.SKCompat(dnn_1_clf) #### to be compatible with sklearn
    

from datetime import datetime
start=datetime.now()
dnn_1_clf.fit(X_train_s, y_train, batch_size=200, steps=(1100))
Train_elapse = (datetime.now()-start).total_seconds()
print('dnn_1_clf training time', Train_elapse)
Train_time_list_1ep.append(Train_elapse)
    
y_train_pred = dnn_1_clf.predict(X_train_s) #return dictionary
Train_error = 1 - accuracy_score(y_train, y_train_pred['classes'])

    
print(type(y_train_pred))

start=datetime.now()
y_val_pred =  dnn_1_clf.predict(X_val_s)
    
Val_elapse = (datetime.now()-start).total_seconds()

Val_time_list_1ep.append(Val_elapse)
Val_error = 1 - accuracy_score(y_val, y_val_pred['classes'])
      
    
print('Train error, Val_error',  Train_error, '/', Val_error)
                                   
Train_error_list_1ep.append(Train_error)
Val_error_list_1ep.append(Val_error)
    
print("Train_size: ", Train_size_list_1ep)
print("Train_error: ", Train_error_list_1ep)
print("Val_error: ", Val_error_list_1ep)
print("Train_time: ", Train_time_list_1ep)
print("Val_time: ", Val_time_list_1ep)  

accuracy = accuracy_score(y_val, y_val_pred['classes'])
accuracy


# # Changing the above code with the values below and observing validation accuracy results

# In[45]:


Neurons_1ep_1hiddenlayer_relu_50BS = [500, 1000, 1500, 2000, 3000, 4000]
Neurons_1ep_1hiddenlayer_relu_50BS


# In[46]:


accuracy_1ep_1hiddenlayer_relu = [0.4178, 0.4124, 0.4046, 0.3942, 0.3902, 0.3842]
accuracy_1ep_1hiddenlayer_relu


# In[47]:


Neurons_1ep_1hiddenlayer_relu_100BS = [500, 1000, 1500, 2000, 3000, 4000]
Neurons_1ep_1hiddenlayer_relu_100BS


# In[48]:


accuracy_1ep_1hiddenlayer_relu_100BS = [0.4876, 0.4796, 0.4752, 0.4654, 0.4646, 0.4554]
accuracy_1ep_1hiddenlayer_relu_100BS


# In[50]:


Neurons_1ep_1hiddenlayer_relu_200BS = [500, 1000, 1500, 2000, 3000, 4000]
Neurons_1ep_1hiddenlayer_relu_200BS


# In[51]:


accuracy_1ep_1hiddenlayer_relu_200BS = [0.508, 0.5094, 0.5096, 0.5084, 0.4972, 0.4950]
accuracy_1ep_1hiddenlayer_relu_200BS


# In[39]:


Neurons_1ep_2hiddenlayers_relu_200BS = [[500, 1000], [1000, 2000], [2000, 3000], [3000,4000], [4000,5000], [5000,4000], [6000,5000],[7000,6000]]
Neurons_1ep_2hiddenlayers_relu_200BS


# In[40]:


accuracy_1ep_2hiddenlayers_relu_200BS = [0.5186, 0.527, 0.5236, 0.5248, 0.5312, 0.5286, 0.5314, 0.5352, 0.5384]
accuracy_1ep_2hiddenlayers_relu_200BS


# In[52]:


plt.figure(figsize = (15,15))
plt.plot(Neurons_1ep_1hiddenlayer_relu_50BS, accuracy_1ep_1hiddenlayer_relu, 'g*-', label = "Batch Size = 50")
plt.plot(Neurons_1ep_1hiddenlayer_relu_100BS, accuracy_1ep_1hiddenlayer_relu_100BS, 'ro-', label = "Batch Size = 100")
plt.plot(Neurons_1ep_1hiddenlayer_relu_200BS, accuracy_1ep_1hiddenlayer_relu_200BS, 'yd-', label = "Batch Size = 200")

plt.xlabel('Number of Neurons')
plt.ylabel('Accuracy')
plt.legend( fontsize=12)
#plt.xlim(0, 1000)


# In[53]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
Order = [1, 2, 3, 4, 5, 6, 7]
Values = [0.5186, 0.527, 0.5236, 0.5248, 0.5314, 0.5352, 0.5384]

LABELS = ["[500,1000]", "[1000,2000]","[2000,3000]", "[3000,4000]", "[4000,5000]", "[6000,5000]", "[7000,6000]"]
plt.figure(figsize = (10,5))
plt.bar(Order, Values, align='center', width=0.3)
plt.xticks(Order, LABELS)
plt.xlabel('2 Hidden Layers and Neurons')
plt.ylabel('Validation Accuracy')
plt.ylim(0.51, 0.54)
plt.show()


# In[54]:


get_ipython().run_line_magic('matplotlib', 'inline')
Order = [1, 2, 3, 4]
Values = [0.5204, 0.5304, 0.5248, 0.5302]

LABELS = ["[500,1000,2000]", "[1000,2000,3000]", "[2000,3000,4000]","[4000,5000,4000]"]
plt.figure(figsize = (7,5))
plt.bar(Order, Values, align='center', width=0.3)
plt.xticks(Order, LABELS)
plt.xlabel('Three Hidden Layers and Neurons')
plt.ylabel('Validation Accuracy')
plt.ylim(0.51, 0.533)
plt.show()


# # Changing no. of hidden layers while fixing batch size -> 200, activation function -> relu

# In[213]:


Neurons_3layers_batchsize200_relu = [[500,1000,2000], [1000,2000,3000], [2000,3000,4000],[4000,5000,4000]]
Neurons_3layers_batchsize200_relu


# In[214]:


accuracy_3layers_batchsize200_relu =[0.5204, 0.5304, 0.5248, 0.5302]
accuracy_3layers_batchsize200_relu


# # FIXING 2 hidden layers -> [7000,6000], activation function -> relu, batch size -> 200, 1 EPOCH

# In[215]:


#learning curve

Train_size_list_1ep = []
Train_error_list_1ep = []
Val_error_list_1ep = []
Train_time_list_1ep = []
Val_time_list_1ep = []

config = tf.contrib.learn.RunConfig(tf_random_seed=42) 
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train_s)
    
dnn_1_clf = tf.contrib.learn.DNNClassifier(activation_fn=tf.nn.relu, hidden_units=[7000,6000], n_classes=10, feature_columns=feature_columns, config=config)
dnn_1_clf = tf.contrib.learn.SKCompat(dnn_1_clf) #### to be compatible with sklearn
    

from datetime import datetime
start=datetime.now()
dnn_1_clf.fit(X_train_s, y_train, batch_size=200, steps=(1100))
Train_elapse = (datetime.now()-start).total_seconds()
print('dnn_1_clf training time', Train_elapse)
Train_time_list_1ep.append(Train_elapse)
    
y_train_pred = dnn_1_clf.predict(X_train_s) #return dictionary
Train_error = 1 - accuracy_score(y_train, y_train_pred['classes'])

    
print(type(y_train_pred))

start=datetime.now()
y_val_pred =  dnn_1_clf.predict(X_val_s)
    
Val_elapse = (datetime.now()-start).total_seconds()

Val_time_list_1ep.append(Val_elapse)
Val_error = 1 - accuracy_score(y_val, y_val_pred['classes'])
      
    
print('Train error, Val_error',  Train_error, '/', Val_error)
                                   
Train_error_list_1ep.append(Train_error)
Val_error_list_1ep.append(Val_error)
    
print("Train_size: ", Train_size_list_1ep)
print("Train_error: ", Train_error_list_1ep)
print("Val_error: ", Val_error_list_1ep)
print("Train_time: ", Train_time_list_1ep)
print("Val_time: ", Val_time_list_1ep)  

accuracy = accuracy_score(y_val, y_val_pred['classes'])
accuracy


# # FIXING 2 hidden layers -> [7000,6000], activation function -> relu, batch size -> 200, 5 EPOCH

# In[221]:


#learning curve

Train_size_list_5epfix = []
Train_error_list_5epfix = []
Val_error_list_5epfix = []
Train_time_list_5epfix = []
Val_time_list_5epfix = []

config = tf.contrib.learn.RunConfig(tf_random_seed=42) 
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train_s)
    
dnn_1_clf = tf.contrib.learn.DNNClassifier(activation_fn=tf.nn.relu, hidden_units=[7000,6000], n_classes=10, feature_columns=feature_columns, config=config)
dnn_1_clf = tf.contrib.learn.SKCompat(dnn_1_clf) #### to be compatible with sklearn
    

from datetime import datetime
start=datetime.now()
dnn_1_clf.fit(X_train_s, y_train, batch_size=200, steps=(5*1100))
Train_elapse = (datetime.now()-start).total_seconds()
print('dnn_1_clf training time', Train_elapse)
Train_time_list_1ep.append(Train_elapse)
    
y_train_pred = dnn_1_clf.predict(X_train_s) #return dictionary
Train_error = 1 - accuracy_score(y_train, y_train_pred['classes'])

    
print(type(y_train_pred))

start=datetime.now()
y_val_pred =  dnn_1_clf.predict(X_val_s)
    
Val_elapse = (datetime.now()-start).total_seconds()

Val_time_list_1ep.append(Val_elapse)
Val_error = 1 - accuracy_score(y_val, y_val_pred['classes'])
      
    
print('Train error, Val_error',  Train_error, '/', Val_error)
                                   
Train_error_list_5epfix.append(Train_error)
Val_error_list_5epfix.append(Val_error)
    
print("Train_size: ", Train_size_list_5epfix)
print("Train_error: ", Train_error_list_5epfix)
print("Val_error: ", Val_error_list_5epfix)
print("Train_time: ", Train_time_list_5epfix)
print("Val_time: ", Val_time_list_5epfix)  

accuracy = accuracy_score(y_val, y_val_pred['classes'])
accuracy


# # FIXING 2 hidden layers -> [7000,6000], activation function -> relu, batch size -> 200, 10 EPOCHs

# In[222]:



Train_size_list_10epfix = []
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
dnn_1_clf.fit(X_train_s, y_train, batch_size=200, steps=(10*1100))
Train_elapse = (datetime.now()-start).total_seconds()
print('dnn_1_clf training time', Train_elapse)
Train_time_list_1ep.append(Train_elapse)
    
y_train_pred = dnn_1_clf.predict(X_train_s) #return dictionary
Train_error = 1 - accuracy_score(y_train, y_train_pred['classes'])

    
print(type(y_train_pred))

start=datetime.now()
y_val_pred =  dnn_1_clf.predict(X_val_s)
    
Val_elapse = (datetime.now()-start).total_seconds()

Val_time_list_1ep.append(Val_elapse)
Val_error = 1 - accuracy_score(y_val, y_val_pred['classes'])
      
    
print('Train error, Val_error',  Train_error, '/', Val_error)
                                   
Train_error_list_10epfix.append(Train_error)
Val_error_list_10epfix.append(Val_error)
    
print("Train_size: ", Train_size_list_10epfix)
print("Train_error: ", Train_error_list_10epfix)
print("Val_error: ", Val_error_list_10epfix)
print("Train_time: ", Train_time_list_10epfix)
print("Val_time: ", Val_time_list_10epfix)  

accuracy = accuracy_score(y_val, y_val_pred['classes'])
accuracy


# In[226]:


Accuracy_increase_with_epochs = [0.5384, 0.5868, 0.5886]
Accuracy_increase_with_epochs


# In[228]:


epochs = [1, 5, 10]
epochs


# In[ ]:


plt.figure(figsize = (5,5))
plt.plot(epochs, accuracy_increase_with_epochs, 'm*-')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend( fontsize=15)
#plt.xlim(0, 1000)

