# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 19:37:13 2020

@author: n052328
"""


#%%
url="file:///Users/n052328/bvmnglpercod01_202005260021.out.gz"
percod_file = tf.keras.utils.get_file("bvmnglpercod01.csv", url)
df = pd.read_csv(percod_file,sep='\t',
                 names=['host','interval','timestamp','dev','metric','value'],
                 usecols=[0,2,3,4,5],
                 compression='gzip' )
df['timestamp'] = df['timestamp'] // 60 * 60
df.head(20)
df.dtypes
dfs = pd.pivot_table(df,values='value',index=['timestamp','metric','host','dev']).unstack(1)
dfs.head(20)
dfs.unstack(1).unstack()


#%%
df['host'] = pd.Categorical(df['host'])
df['dev'] = pd.Categorical(df['dev'])
df['metric'] = pd.Categorical(df['metric'])
df.dtypes
df['host'] = df.host.cat.codes
df['dev'] = df.dev.cat.codes
df['metric'] = df.metric.cat.codes
df.dtypes
df.describe()
df
tf.constant(df.values)

#%%
tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
                      'foo', 'foo', 'qux', 'qux'],
                     ['one', 'two', 'one', 'two',
                      'one', 'two', 'one', 'two']]))