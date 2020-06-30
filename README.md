# hello-world
My experiments with seq2seq, lstm's , encoders, decoders and tensorflow dataframes


I'm trying to take advantage of tensorflow.data.Dataset.batch and window for saving memory and automate data transformation .
Before trying attention and transformer models, i'm trying to get the best possible acurracy from encoder decoder with lstm based newral network.

Instead of using English to French translation, I'll use sp500 historical prices series.

First cells will preprocess pandas dataframe before it will be wrapped by a tf.data.Dataset.

first file will be gspc_preprodcessing.py

