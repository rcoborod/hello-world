# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 23:36:53 2020

@author: Lafayette
"""

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed,Lambda, Dropout, Activation ,RepeatVector
from tensorflow.keras.callbacks import ModelCheckpoint 
import numpy as np

from tensorflow.keras import backend as K

from tensorflow.keras.models import load_model

import os

#%%
features_num=14
latent_dim=40

#%%
encoder_inputs = Input(shape=(None, features_num))
encoded = LSTM(latent_dim, return_state=False ,return_sequences=True)(encoder_inputs)
encoded = LSTM(latent_dim, return_state=False ,return_sequences=True)(encoded)
encoded = LSTM(latent_dim, return_state=False ,return_sequences=True)(encoded)
encoded = LSTM(latent_dim, return_state=True)(encoded)

encoder = Model(inputs=encoder_inputs, outputs=encoded)
##

encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

#%%
decoder_inputs=Input(shape=(1, features_num))
decoder_lstm_1 = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_lstm_2 = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_lstm_3 = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_lstm_4 = LSTM(latent_dim, return_sequences=True, return_state=True)

decoder_dense = Dense(features_num)

all_outputs = []
inputs = decoder_inputs

# Place holder values:
states_1=encoder_states
states_2=states_1; states_3=states_1; states_4=states_1

for _ in range(1):
    # Run the decoder on one timestep
    outputs_1, state_h_1, state_c_1 = decoder_lstm_1(inputs, initial_state=states_1)
    outputs_2, state_h_2, state_c_2 = decoder_lstm_2(outputs_1)
    outputs_3, state_h_3, state_c_3 = decoder_lstm_3(outputs_2)
    outputs_4, state_h_4, state_c_4 = decoder_lstm_4(outputs_3)

    # Store the current prediction (we will concatenate all predictions later)
    outputs = decoder_dense(outputs_4)
    all_outputs.append(outputs)
    # Reinject the outputs as inputs for the next loop iteration
    # as well as update the states
    inputs = outputs
    states_1 = [state_h_1, state_c_1]
    states_2 = [state_h_2, state_c_2]
    states_3 = [state_h_3, state_c_3]
    states_4 = [state_h_4, state_c_4]


for _ in range(149):
    # Run the decoder on one timestep
    outputs_1, state_h_1, state_c_1 = decoder_lstm_1(inputs, initial_state=states_1)
    outputs_2, state_h_2, state_c_2 = decoder_lstm_2(outputs_1, initial_state=states_2)
    outputs_3, state_h_3, state_c_3 = decoder_lstm_3(outputs_2, initial_state=states_3)
    outputs_4, state_h_4, state_c_4 = decoder_lstm_4(outputs_3, initial_state=states_4)

    # Store the current prediction (we will concatenate all predictions later)
    outputs = decoder_dense(outputs_4)
    all_outputs.append(outputs)
    # Reinject the outputs as inputs for the next loop iteration
    # as well as update the states
    inputs = outputs
    states_1 = [state_h_1, state_c_1]
    states_2 = [state_h_2, state_c_2]
    states_3 = [state_h_3, state_c_3]
    states_4 = [state_h_4, state_c_4]


# Concatenate all predictions
decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)   

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

#model = load_model('pre_model.h5')


print(model.summary())


model.compile(loss='mean_squared_error', optimizer='adam')


def create_wavelength(min_wavelength, max_wavelength, fluxes_in_wavelength, category )  :         
#category :: 0 - train ; 2 - validate ; 4- test. 1;3;5 - dead space
    c=(category+np.random.random())/6         
    k = fluxes_in_wavelength
#
    base= (np.trunc(k*np.random.random()*(max_wavelength-min_wavelength))       +k*min_wavelength)  /k
    answer=base+c/k
    return (answer)       

def make_line(length,category):
    shift= np.random.random()
    wavelength = create_wavelength(30,10,1,category)
    a=np.arange(length)
    answer=np.sin(a/wavelength+shift)
    return answer

def make_data(seq_num,seq_len,dim,category):
    data=np.array([]).reshape(0,seq_len,dim)
    for i in range (seq_num):
        mini_data=np.array([]).reshape(0,seq_len)
        for j in range (dim):
            line = make_line(seq_len,category)
            line=line.reshape(1,seq_len)            
            mini_data=np.append(mini_data,line,axis=0)
        mini_data=np.swapaxes(mini_data,1,0)
        mini_data=mini_data.reshape(1,seq_len,dim)      
        data=np.append(data,mini_data,axis=0)
    return (data)


def train_generator():
    while True:
        sequence_length = np.random.randint(150, 300)+150       
        data=make_data(1000,sequence_length,features_num,0) # category=0 in train


    #   decoder_target_data is the same as decoder_input_data but offset by one timestep

        encoder_input_data = data[:,:-150,:] # all but last 150 

        decoder_input_data = data[:,-151,:] # the one before the last 150.
        decoder_input_data=decoder_input_data.reshape((decoder_input_data.shape[0],1,decoder_input_data.shape[1]))


        decoder_target_data = (data[:, -150:, :]) # last 150        
        yield [encoder_input_data, decoder_input_data], decoder_target_data
def val_generator():
    while True:

        sequence_length = np.random.randint(150, 300)+150       
        data=make_data(1000,sequence_length,features_num,2) # category=2 in val

        encoder_input_data = data[:,:-150,:] # all but last 150 

        decoder_input_data = data[:,-151,:] # the one before the last 150.
        decoder_input_data=decoder_input_data.reshape((decoder_input_data.shape[0],1,decoder_input_data.shape[1]))

        decoder_target_data = (data[:, -150:, :]) # last 150        
        yield [encoder_input_data, decoder_input_data], decoder_target_data

filepath_for_w= 'flux_p2p_s2s_model.h5' 
checkpointer=ModelCheckpoint(filepath_for_w, monitor='val_loss', verbose=0, save_best_only=True, mode='auto', period=1)     
model.fit_generator(train_generator(),callbacks=[checkpointer], steps_per_epoch=30, epochs=2000, verbose=1,validation_data=val_generator(),validation_steps=30)
model.save(filepath_for_w)




def predict_wave(input_wave,input_for_decoder):  # input wave= x[n,:,:], ie points except the last 150; each wave has feature_num features. run this function for all such instances (=n)   
    #print (input_wave.shape)
    #print (input_for_decoder.shape)
    pred= model.predict([input_wave,input_for_decoder])

    return pred

def predict_many_waves_from_input(x):   
    x, x2=x # x == encoder_input_data ; x==2 decoder_input_data

    instance_num= x.shape[0]


    multi_predict_collection=np.zeros((x.shape[0],150,x.shape[2]))

    for n in range(instance_num):
        input_wave=x[n,:,:].reshape(1,x.shape[1],x.shape[2])
        input_for_decoder=x2[n,:,:].reshape(1,x2.shape[1],x2.shape[2])
        wave_prediction=predict_wave(input_wave,input_for_decoder)
        multi_predict_collection[n,:,:]=wave_prediction
    return (multi_predict_collection)

def test_maker():
    if True:        
        sequence_length = np.random.randint(150, 300)+150       
        data=make_data(470,sequence_length,features_num,4) # category=4 in test

        encoder_input_data = data[:,:-150,:] # all but last 150 

        decoder_input_data = data[:,-151,:] # the one before the last 150.
        decoder_input_data=decoder_input_data.reshape((decoder_input_data.shape[0],1,decoder_input_data.shape[1]))

        decoder_target_data = (data[:, -150:, :]) # last 150        
        return [encoder_input_data, decoder_input_data],    decoder_target_data

x,y= test_maker()   



a=predict_many_waves_from_input (x) # is that right..?
x=x[0] # keep the wave (generated data except last 150 time points) 
print (x.shape)
print (y.shape)
print (a.shape)

np.save ('a.npy',a)
np.save ('y.npy',y)
np.save ('x.npy',x)



print (np.mean(np.absolute(y[:,:,0]-a[:,:,0])))
print (np.mean(np.absolute(y[:,:,1]-a[:,:,1])))
print (np.mean(np.absolute(y[:,:,2]-a[:,:,2])))
print (np.mean(np.absolute(y[:,:,3]-a[:,:,3])))
print (np.mean(np.absolute(y[:,:,4]-a[:,:,4])))