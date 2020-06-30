# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
#import pathlib
import os
import pandas as pd
import numpy as np
#import time
np.set_printoptions(precision=4)

import matplotlib.pyplot as plt


#%%
gspc_to_2020_url="https://query1.finance.yahoo.com/v7/finance/download/%5EGSPC?period1=-252460800&period2=1577750400&interval=1d&events=history"
#gspc_url="file:///Users/n052328/Downloads/gspc.csv"
gspc_from_2020_url="https://query1.finance.yahoo.com/v7/finance/download/%5EGSPC?period1=1577836800&period2=1609372800&interval=1d&events=history"
gspc_to_2020_file = tf.keras.utils.get_file("gspc_to_2020.csv", gspc_to_2020_url)
df = pd.read_csv(gspc_to_2020_file,index_col='Date')
df = df.append(pd.read_csv(gspc_from_2020_url,index_col='Date'))
#df = pd.read_csv("gspc.csv",index_col='Date')
df.head()
df.tail()
print(df.dtypes)
df[['o','h','l','c','v']] = df[['Open','High','Low','Close','Volume']].apply(np.log)
#df.plot(subplots=True)
df = df[['o','h','l','c','v']]
df.plot(subplots=True)
df.describe()

df[['oo','hh','ll','cc','vv']] = df[['o','h','l','c','v']].sub(df[['o','h','l','c','v']].shift(axis=0))
df[['oo','hh','ll','cc','vv']].head()
df[['oo','hh','ll','cc','vv']].tail()
df['co'] = df['o'].sub(df.loc[:,'c'].shift(1))
df[['oh','ol','oc']] = df[['h','l','c']].sub(df['o'],axis=0)
df.iloc[:,-5:] = df.iloc[:,-5:].fillna(0.0)

dfmean = df.mean()
dfstd = df.std()
df = ( df - dfmean ) / dfstd

#%%
alpha= ((-1. + 5 ** .5)/2.0)**16;print(alpha)
df.ewm(alpha=alpha,axis=0).mean().plot(subplots=True)
df.ewm(alpha=alpha).std().plot(subplots=True)
alpha **=.5;print(alpha)
df.ewm(alpha=alpha,axis=0).mean().plot(subplots=True)
df.ewm(alpha=alpha).std().plot(subplots=True)
alpha **=.5;print(alpha)
df.ewm(alpha=alpha,axis=0).mean().plot(subplots=True)
df.ewm(alpha=alpha).std().plot(subplots=True)
alpha **=.5;print(alpha)
df.ewm(alpha=alpha,axis=0).mean().plot(subplots=True)
df.ewm(alpha=alpha).std().plot(subplots=True)
alpha **=.5;print(alpha)
df.ewm(alpha=alpha,axis=0).mean().plot(subplots=True)
df.ewm(alpha=alpha).std().plot(subplots=True)

df.ewm(alpha=.1).var()
df[['o','h','l','c']].ewm(alpha=.1).mean()

#%%
# df.iloc[-1024:,-5:].describe()
# df.iloc[-1024:,-5:].plot(subplots=True)
# df.iloc[:,-5:].plot(subplots=True)
# df.iloc[:,-5:].head()
# df.rolling(window=1)
#%%
df.iloc[:,-5:] = df.iloc[:,-5:].fillna(0.0)
dataset = tf.data.Dataset.from_tensor_slices(df.iloc[:,-5:].values)
dataset

#%% Mezcla de distintas características de la entrado con menos de la salida
feature_length = 5
label_length = 4
predict_size = 16
seq_size = 7*predict_size
window_size = seq_size + predict_size
batch_size = 2**14 //  window_size

feature_length = 14
label_length = 4
predict_size = 16
seq_size = 7*predict_size
window_size = seq_size + predict_size
batch_size = 2**14 //  window_size
timesteps = seq_size
latent_dim = label_length * predict_size

df.iloc[:,-feature_length:] = df.iloc[:,-feature_length:].fillna(0.0)
df.iloc[:,-feature_length:].values.shape

#range_ds = tf.data.Dataset.range(100000)
dataset1 = tf.data.Dataset.from_tensor_slices(df.iloc[:,-feature_length:].values)
#dataset2 = tf.data.Dataset.from_tensor_slices(df.iloc[:,-label_length:].values)

#%%  ok ventanas deslizantes superpuestas aprendizaje muestras independientes


def sub_to_batch(sub):
  return sub.batch(window_size, drop_remainder=True)

#windows = range_ds.window( window_size, shift=1).flat_map(sub_to_batch)
windows = dataset1.window( window_size, shift=1).flat_map(sub_to_batch)

def label_next_steps(batch):
  return (batch[:-predict_size],   # Take the first 5 steps
          batch[-predict_size:,-label_length:])   # take the remainder

def label_next_steps_s2s(batch):
  return (batch[:-predict_size-1],   # Take the first 5 steps
          batch[-predict_size-1:-1,-label_length:],
          batch[-predict_size:,-label_length:])   # take the remainder

predict_steps = windows.map(label_next_steps)
predict_steps_s2s = windows.map(label_next_steps_s2s)

for features, label in predict_steps.take(3):
  print(features.numpy(), " => ", label.numpy())
  
for features,features2, label in predict_steps_s2s.take(3):
  print(features.numpy(),features2.numpy(), " => ", label.numpy())
  
tf.data.Dataset.from_tensor_slices(label).unbatch()
  

hh = predict_steps.batch( batch_size )
hh_s2s = predict_steps_s2s.batch( batch_size )

# for features, label in hh.take(1):
#   print(features.numpy(), " => ", 
#         label.numpy()) 


#%% modelado

from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed

initializeri = tf.keras.initializers.Identity()
initializero = tf.keras.initializers.Orthogonal()

# Initializing the RNN
regressor = Sequential()
#regressor.add(Embedding(seq_size, feature_length))
# adding first layer
#regressor.add(LSTM(units = window_size * feature_length , return_sequences=True,
regressor.add(LSTM(units = latent_dim , return_sequences=True,
                   input_shape=(None, feature_length),
                   dropout=0.5,
                   recurrent_dropout=0.5))
#regressor.add(Dropout(0.1))
# adding second layer
# regressor.add(LSTM(units=20, return_sequences=True, stateful=True))
# regressor.add(LSTM(units = label_length * predict_size, return_sequences=True))
#regressor.add(LSTM(units = predict_size * label_length, return_sequences=False,
regressor.add(LSTM(units = latent_dim , return_sequences=False,
                   dropout=0.5,
                   recurrent_dropout=0.5))
# regressor.add(Dropout(0.2))
# adding third layer
# regressor.add(LSTM(units = predict_size, return_sequences=False,
#                    dropout=0.5,
#                    recurrent_dropout=0.5))
# adding output layer
regressor.add(Dense( units = latent_dim ,))
regressor.add(Dropout(0.1))
regressor.add(Reshape(( predict_size , label_length )))
df.shape
# compile model
regressor.compile( optimizer= 'adam', loss= 'mean_squared_error')
#regressor.compile( optimizer= 'adam', loss= 'mae')

#prueba inicialización
regressor.predict(df.iloc[-seq_size:,-feature_length:].values.reshape(1,seq_size,feature_length))
regressor.predict(hh).shape

regressor.summary()
#%%
model = Sequential()
model.add(LSTM(latent_dim, return_sequences=True, input_shape=(None, feature_length)))
model.add(Dropout(0.2))
model.add(LSTM(latent_dim, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(latent_dim, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(latent_dim))
model.add(Activation('tanh')) 
#model.add(Reshape(None,4))
model.add(RepeatVector(predict_size)) # 50 output points

model.add(LSTM(latent_dim, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(latent_dim, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(5, return_sequences=True))
model.add(TimeDistributed(Dense(4)))  # <--- This is new :)
model.add(Activation('linear'))
#model.add(Lambda(lambda x: x[:, -predict_size:, -4:])) # Grab only the last predict_size values
model.compile(loss='mean_squared_error',optimizer='adam')
print(model.summary())

#%% training and results
regressor.fit( hh , epochs=4)
model.fit(hh, epochs= 4)

regressor.save("regressor.h5")
model.save("model.h5")

regressor = tf.keras.models.load_model("regressor.h5")
model = tf.keras.models.load_model('model.h5')


#%%

encoder_inputs = Input(shape=(None,feature_length))
encoded = LSTM( latent_dim,  return_state=False, return_sequences=True)(encoder_inputs)
encoded = LSTM( latent_dim,  return_state=False, return_sequences=True)(encoded)
encoded = LSTM( latent_dim,  return_state=True, return_sequences=False)(encoded)
encoder = Model(inputs=encoder_inputs, outputs=encoded)
encoder.summary()
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None,label_length))
decoder_lstm = LSTM( latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(label_length)
decoder_outputs = decoder_dense(decoder_outputs)

model = Model(inputs=[encoder_inputs,decoder_inputs],outputs=decoder_outputs)
model.compile(loss='mean_squared_error',optimizer='adam')

model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
model.fit(hh_s2s, epochs=1)
# Save model
model.save('s2s.h5')


#%%
kk = model.predict(df.iloc[-seq_size:,-feature_length:].values.reshape(1,seq_size,feature_length))
#print(regressor.predict(df.iloc[-seq_size:,-feature_length:].values.reshape(1,seq_size,feature_length)))
kk = regressor.predict(df.iloc[-seq_size:,-feature_length:].values.reshape(1,seq_size,feature_length))
#kk = regressor.predict(df.iloc[-seq_size-1:-1,-feature_length:].values.reshape(1,seq_size,feature_length))
#kk = regressor.predict(df.iloc[-seq_size-(predict_size-2):-(predict_size-2),-feature_length:].values.reshape(1,seq_size,feature_length))
#kk = regressor.predict(df.iloc[-seq_size-predict_size:-predict_size,-feature_length:].values.reshape(1,seq_size,feature_length))
#kk = regressor.predict(df.iloc[-seq_size-predict_size*2:-predict_size*2,-feature_length:].values.reshape(1,seq_size,feature_length))

#%%
kk.shape
kk.cumsum()
kk = kk * dfstd[-4:].values + dfmean[-4:].values
kk[:,:,0::3].cumsum()
print(kk)

r=0.0

# print("mean one quarter of day                      ==> {}",kk.mean() )
# print("std one quarter of day                       ==> {}",kk.std() )
# print("kelly one quarter of day                     ==> {}",(kk.mean()-r)/kk.var() )
# print("kelly if kk.mean is neg one quarter of day   ==> {}",(-kk.mean()-r)/kk.var() )
# print("expected g at the end of pred period         ==> {}",(kk.mean()-kk.var()/2)*predict_size*label_length )
# print("expected std at the end of predicted period  ==> {}",(kk.var()*predict_size*label_length)**.5 )
kk.mean(),kk.std(),kk.mean()/kk.var(),(kk.mean()-kk.var()/2)*predict_size*label_length
-kk.mean(),kk.std(),(-kk.mean()-r)/kk.var(),(-kk.mean()-kk.var()/2)*predict_size*label_length
kk[:,:,1:3].std()*(predict_size*2)**.5

kk[:,:,0::3].cumsum()
kk[:,:,0::3].cumsum().shape
# set plot size for all plots that follow
plt.rcParams["figure.figsize"] = (8, 8)

# create the plot space upon which to plot the data
fig, ax = plt.subplots()

days_rg = np.arange(0,predict_size,0.5)
# add the x-axis and the y-axis to the plot
ax.bar(days_rg, kk[:,:,0::3].cumsum(), color="grey")

# set plot title
ax.set(title="expected patter for nex 20 days ")

# add labels to the axes
ax.set(xlabel="Month", ylabel="Cumsum occo 20 days"); 
plt.plot(days_rg,kk[:,:,0::3].cumsum(), 'r+')
plt.plot(days_rg,kk[:,:,0::3].cumsum())

kk.shape
kk = kk.reshape( predict_size, label_length)
futuredf = pd.DataFrame(data=kk[0:,0:],
                        index=[ i for i in range(0,kk.shape[0])],
                        columns=['co','oh','ol','oc'])

kkdf=futuredf[['co','oh']]
kkdf['hl'] = futuredf['ol'].sub(futuredf['oh'])
kkdf['lc'] = futuredf['oc'].sub(futuredf['ol'])
kkdf.plot(subplots=True)
plt.subplots()
kknew=kkdf.values.reshape(-1,predict_size*label_length)
kkcum=kknew.cumsum()
plt.plot(np.arange(0,predict_size,0.25), kkcum,'b-')
(df[['o','h','l','c']].tail(predict_size)-8.014723).plot()
df[['o','h','l','c']].tail(window_size).plot()
np.exp(df[['o','h','l','c']].tail(predict_size))
np.exp(kkcum)

np.percentile(kknew,50)
np.quantile(kknew,0.5)
r=0.0
print("mean one quarter of day                      ==> {}",kknew.mean() )
print("std one quarter of day                       ==> {}",kknew.std() )
print("kelly one quarter of day                     ==> {}",(kknew.mean()-r)/kknew.var() )
print("kelly if kk.mean is neg one quarter of day   ==> {}",(-kknew.mean()-r)/kknew.var() )
print("expected g at the end of pred period         ==> {}",(kknew.mean()-kknew.var()/2)*predict_size*label_length )
print("expected g if kk.mean is neg at the end of pred period   ==> {}",(-kknew.mean()-kknew.var()/2)*predict_size*label_length )
print("expected std at the end of predicted period  ==> {}",(kknew.var()*predict_size*label_length)**.5 )


#%%

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)

  
  
  

