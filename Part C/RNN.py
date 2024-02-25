import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from plot_keras_history import plot_history, show_history


print("downloading data...")
#fetching data, word limit = 20000 (maximum)
(x_train_imdb, y_train_imdb), (x_test_imdb, y_test_imdb) = tf.keras.datasets.imdb.load_data(num_words=20000)


#padding data, data is already in forms of numbers
x_train_imdb = tf.keras.preprocessing.sequence.pad_sequences(x_train_imdb, maxlen=100)

x_test_imdb = tf.keras.preprocessing.sequence.pad_sequences(x_test_imdb, maxlen=100)


print("creating rnn...")
#cresting rnn with embed size = 64 and word size = 20000 and input shape equal to the size of the dataset.
#we use a tanh LSTM layer with 64 units, a dropout layer and finally a sigmoid function  
imdb_rnn = tf.keras.models.Sequential([
      tf.keras.layers.Embedding(20000, 64, input_shape=(x_train_imdb.shape[1],)),
      tf.keras.layers.LSTM(units=64, activation='tanh'),
      tf.keras.layers.Dropout(rate=0.2),
      tf.keras.layers.Dense(units=1, activation='sigmoid')
])


print("compiling rnn...")
#in order to compile we use the rmsprop optimiser which works better than any other
imdb_rnn.compile(optimizer='rmsprop', loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])

print("fitting rnn...")
#5 epochs is enough and the batch size should be around 64. Less than 64 and it takes more time. 
#More than 64 and accuracy drops. Accuracy should be arround 93%
rnnHistory = imdb_rnn.fit(x_train_imdb, y_train_imdb, epochs=5, batch_size=64)


#printing and plotting results 
print(imdb_rnn.evaluate(x_test_imdb, y_test_imdb))

show_history(rnnHistory)
plt.close()


