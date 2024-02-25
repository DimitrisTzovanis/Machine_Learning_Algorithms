import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier




#fetching data 25000 training and 25000 testing
print("fetching train and test data...")
(x_train_imdb, y_train_imdb), (x_test_imdb, y_test_imdb) = tf.keras.datasets.imdb.load_data()


#translating data to words from numbers
print("translating data to words from numbers...")
word_index = tf.keras.datasets.imdb.get_word_index()
index2word = dict((i + 3, word) for (word, i) in word_index.items())
index2word[0] = '[pad]'
index2word[1] = '[bos]'
index2word[2] = '[oov]'
x_train_imdb = np.array([' '.join([index2word[idx] for idx in text]) for text in x_train_imdb])
x_test_imdb = np.array([' '.join([index2word[idx] for idx in text]) for text in x_test_imdb])


#creating binary voacbulary
print("creating binary vocabulary...")
binary_vectorizer = CountVectorizer(binary=True, min_df=100)
x_train_imdb_binary = binary_vectorizer.fit_transform(x_train_imdb)
x_test_imdb_binary = binary_vectorizer.transform(x_test_imdb)
print('Vocabulary size:', len(binary_vectorizer.vocabulary_))

x_train_imdb_binary = x_train_imdb_binary.toarray()
x_test_imdb_binary = x_test_imdb_binary.toarray()


#ID3 algorithm

depth = 5;
print("creating decision tree...")
print("Depth:", depth)
ID3 = DecisionTreeClassifier(max_depth=depth, criterion='entropy', max_features=0.6, splitter='best')
ID3.fit(x_train_imdb_binary, y_train_imdb)
y_pred = ID3.predict(x_test_imdb_binary)



#results
    
print("")
print("Accuracy:",accuracy_score(y_test_imdb, y_pred))


print(classification_report(y_test_imdb, ID3.predict(x_test_imdb_binary)))



#learning curves
def custom_learning_curve(x_train, y_train, x_test, y_test,
                          n_splits, ndepth):
  
  split_size = int(len(x_train) / n_splits)
  x_splits = np.split(x_train, n_splits) # must be equal division
  y_splits = np.split(y_train, n_splits)
  train_accuracies = list()
  test_accuracies = list()
  curr_x = x_splits[0]
  curr_y = y_splits[0]
  ID3 = DecisionTreeClassifier(max_depth=ndepth, criterion='entropy', max_features=0.6, splitter='best')
  ID3.fit(curr_x, curr_y)
  train_accuracies.append(accuracy_score(curr_y, ID3.predict(curr_x)))
  test_accuracies.append(accuracy_score(y_test, ID3.predict(x_test)))

  for i in range(1, len(x_splits)):
    ID3 = DecisionTreeClassifier(max_depth=ndepth, criterion='entropy', max_features=0.6, splitter='best')
    curr_x = np.concatenate((curr_x, x_splits[i]), axis=0)
    curr_y = np.concatenate((curr_y, y_splits[i]), axis=0)
    ID3.fit(curr_x, curr_y)

    train_accuracies.append(accuracy_score(curr_y, ID3.predict(curr_x)))

    test_accuracies.append(accuracy_score(y_test, ID3.predict(x_test)))

  plt.plot(list(range(split_size, len(x_train) + split_size, 
                      split_size)), train_accuracies, 'o-', color="b",
             label="Training accuracy")
  plt.plot(list(range(split_size, len(x_train) + split_size, 
                      split_size)), test_accuracies, 'o-', color="red",
           label="Testing accuracy")
  plt.legend(loc="lower right")
  plt.show()



print("creating learning curves...")
custom_learning_curve(x_train=x_train_imdb_binary, y_train=y_train_imdb, x_test=x_test_imdb_binary,
                      y_test=y_test_imdb, n_splits=5, ndepth = depth)







