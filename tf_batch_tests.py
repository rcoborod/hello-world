# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 22:38:14 2020

@author: n052328
"""



#%%
batches = dataset.batch(25, drop_remainder=True)

def label_next_5_steps(batch):
  return (batch[:-5],   # Take the first 5 steps
          batch[-5:])   # take the remainder

predict_5_steps = batches.map(label_next_5_steps)

for features, label in predict_5_steps.take(3):
  print(features.numpy(), " => ", label.numpy())
  
#%%
range_ds = tf.data.Dataset.range(100000)
range_ds.numpy()
batches = range_ds.batch(10, drop_remainder=True)

for batch in batches.take(5):
  print(batch.numpy())

#%%
feature_length = 20
label_length = 5

features = range_ds.batch(feature_length, drop_remainder=True)
labels = range_ds.batch(feature_length).skip(1).map(lambda labels: labels[:-15])

predict_5_steps = tf.data.Dataset.zip((features, labels))

for features, label in predict_5_steps.take(3):
  print(features.numpy(), " => ", label.numpy())
  
#%%
feature_length = 20
label_length = 5

features = dataset.batch(feature_length, drop_remainder=True)
labels = dataset.batch(feature_length).skip(1).map(lambda labels: labels[:-15])

predict_5_steps = tf.data.Dataset.zip((features, labels))

for features, label in predict_5_steps.take(3):
  print(features.numpy(), " => ", label.numpy())  
  



#%%
batches = dataset.batch(100, drop_remainder=True)

for batch in batches.take(50):
  print(batch.numpy())
  
for batch in batches:
  print(batch.numpy())
  
#%%  
def plot_batch_sizes(ds):
  batch_sizes = [batch.shape[0] for batch in ds]
  plt.bar(range(len(batch_sizes)), batch_sizes)
  plt.xlabel('Batch number')
  plt.ylabel('Batch size')

#%%

batches = dataset.batch(1000).repeat(3)
plot_batch_sizes(batches)

batches = dataset.repeat(3).batch(1000 , drop_remainder=True)
plot_batch_sizes(batches)

for batch in batches.take(1):
  print(batch.numpy())
  
#%%
df.values.reshape(-1,1).shape