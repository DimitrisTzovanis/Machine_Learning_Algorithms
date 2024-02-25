from collections import Counter
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm
import math



class RandomForest:
    def __init__(self, NumofTrees=10, min_samples=2, max_depth=5):
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.NumofTrees = NumofTrees
        self.nfeatures = None
        self.trees = []

    def fit(self, x, y):
        self.trees = []
        for _ in range(self.NumofTrees):    #loops as many times as the amount of trees we want to create
            tree = ID3(min_samples = self.min_samples ,max_depth=self.max_depth,) #creates id3 tree
            samples = x.shape[0]
            indexes = np.random.choice(samples, samples, replace=True)     #chooses random indexes of features
            xRandom = x[indexes]  #maps the randomly selected indexes
            yRandom = y[indexes]
            tree.fit(xRandom, yRandom)   #fits the tree of random features
            self.trees.append(tree)  #adds it to the table of trees

    def predict(self, x):
        predictions = []
        for tree in self.trees:     #loop for every tree
            predictions.append(tree.predict(x)) #call the prediction function for every id3 tree and store the result
        np.array(predictions)   #convert to np array  
        predictions = np.swapaxes(predictions, 0, 1)  #change the axe of the array to vertical for horizontal
        y_pred = []
        for prediction in predictions:  #loop for every prediction
            c = Counter(prediction)
            most = c.most_common(1)[0][0]  # find the most common prediction among all the trees
            y_pred.append(most)  #store result
        return np.array(y_pred)  #convert to np array



class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.left = left   #left child
        self.right = right   #right child
        self.feature = feature  #feature with the best information gain that will be used to create the children 
        self.threshold = threshold   #best split point (of feature) between the 2 children
        self.value = value   #only leaf nodes have value


class ID3:
    def __init__(self, min_samples=2, max_depth=100):
        self.nfeatures = None  #features of tree = feature of train/test data
        self.root = None   #root of tree
        self.min_samples = min_samples  #minimum samples at node
        self.max_depth = max_depth   #maximum depth of tree
        

    def fit(self, x, y):
         #grow tree
        self.nfeatures = x.shape[1]         #safety check, number of features
        self.root = self.grow_tree(x, y)     #create tree



    def grow_tree(self, x, y, depth=0):
        samples, numfeatures = x.shape   #take the number of features and samples   
        labels = len(np.unique(y))       #take all the unique labels

        # stop growing if
        if (depth >= self.max_depth         #reached maximum tree depth
            or labels == samples < self.min_samples       #too little samples to spli
            or labels == 1):      #too little labels, cant split
                c = Counter(y)
                most = c.most_common(1)[0][0]   #returns a 1d vector where samples are smaller 
                leaf_value = most
                return Node(value=leaf_value)   #then finds the most popular label

        featureIndexes = np.random.choice(numfeatures, self.nfeatures, replace=False)
        #creates an array of randomly selected features (takes every feature once)

        
        BestFeaure, BestThrehold = self.best_criteria(x, y, featureIndexes)  #find best feature and split using the IG function

        # grow the children that result from the split
        leftIndex, rightIndex = self.split(x[:, BestFeaure], BestThrehold) 
        left = self.grow_tree(x[leftIndex, :], y[leftIndex], depth + 1)    #call recursively function. Only left Indexes but all features
        right = self.grow_tree(x[rightIndex, :], y[rightIndex], depth + 1)
        return Node(BestFeaure, BestThrehold, left, right)

    def best_criteria(self, x, y, featureIdxs):
        maxgain = -1
        splitIndex, splitThreshold = None, None
        for featureIdx in featureIdxs:
            column = x[:, featureIdx]    #take the samples only at index i of x
            thresholds = np.unique(column)   #take all the unique samples
            for threshold in thresholds:
                gain = self.IG(y, column, threshold)  #calculate information gain

                if gain > maxgain:   #find max information gain
                    maxgain = gain
                    splitIndex = featureIdx
                    splitThreshold = threshold

        return splitIndex, splitThreshold

    def IG(self, y, column, threshold):
        # parent loss
        parent_entropy = entropy(y)  #calculate parent entropy

        # generate split
        left, right = self.split(column, threshold)

        if (len(left) == 0 or len(right) == 0):  #0 information gain
            return 0

        # compute the weighted avg. of the loss for the children
        Nsamples = len(y)   #number of samples
        NsamplesLeft = len(left)
        NsamplesRight = len(right)
        EntropyLeft = entropy(y[left])
        EntropyRight = entropy(y[right])
        child_entropy = (NsamplesLeft / Nsamples) * EntropyLeft + (NsamplesRight / Nsamples) * EntropyRight

        # information gain is difference in loss before vs. after split
        ig = parent_entropy - child_entropy
        return ig   #ig = entropy(parent) - [weightedAvg]*E(children)

    def split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs


    def predict(self, x):
        #traverse tree
        predict = []
        for i in x:  #for every test sample
            predict.append(self.traverse_tree(i, self.root)) #call traverse tree and store the returned value
        return np.array(predict)


    def traverse_tree(self, x, node):
        if (node.value is not None):   #we have reached a child
            return node.value
        if (x[node.feature] <= node.threshold) :     #if feature smaller than threshold go to the left child
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)


def entropy(y):
    occurences = np.bincount(y)
    Px = occurences / len(y)
    for x in Px:
        if x>0:
            return -np.sum(x * math.log2(x))





# Testing
if __name__ == "__main__":
    #fetching data 25000 training and 25000 testing
    print("fetching train and test data...")
    (x_train_imdb, y_train_imdb), (x_test_imdb, y_test_imdb) = tf.keras.datasets.imdb.load_data(num_words=4000)


    #translating data to words from numbers
    print("translating data to words from numbers...")
    word_index = tf.keras.datasets.imdb.get_word_index()
    index2word = dict((i + 3, word) for (word, i) in word_index.items())
    index2word[0] = '[pad]'
    index2word[1] = '[bos]'
    index2word[2] = '[oov]'
    x_train_imdb = np.array([' '.join([index2word[idx] for idx in text]) for text in x_train_imdb])
    x_test_imdb = np.array([' '.join([index2word[idx] for idx in text]) for text in x_test_imdb])


    #creating binary vocabulary
    print("creating binary vocabulary...")
    from sklearn.feature_extraction.text import CountVectorizer
    #min-df --> when building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.
    binary_vectorizer = CountVectorizer(binary=True, min_df=100)
    x_train_imdb_binary = binary_vectorizer.fit_transform(x_train_imdb)
    x_test_imdb_binary = binary_vectorizer.transform(x_test_imdb)
    print('Vocabulary size:', len(binary_vectorizer.vocabulary_))

    x_train_imdb_binary = x_train_imdb_binary.toarray()
    x_test_imdb_binary = x_test_imdb_binary.toarray()



    #Random Forest algorithm
    
    trees = 4  #change the values of the algorithm
    depth = 5
    
    print("Depth:", depth, ", Number of Trees:", trees)
    print("creating random forest...")
    RForest = RandomForest(NumofTrees = trees, max_depth = depth)


    print("fitting random forest...")
    RForest.fit(x_train_imdb_binary, y_train_imdb)
    y_pred = RForest.predict(x_test_imdb_binary)



    #results
    print("")
    print("Accuracy:",accuracy_score(y_test_imdb, y_pred))

    print(classification_report(y_test_imdb, RForest.predict(x_test_imdb_binary)))


    def custom_learning_curve(x_train, y_train, x_test, y_test,
                              n_splits, ndepth, ntrees):
      
      split_size = int(len(x_train) / n_splits)
      x_splits = np.split(x_train, n_splits) # must be equal division
      y_splits = np.split(y_train, n_splits)
      train_accuracies = list()
      test_accuracies = list()
      curr_x = x_splits[0]
      curr_y = y_splits[0]
      RForest = RandomForest(NumofTrees = ntrees, max_depth = ndepth)
      RForest.fit(curr_x, curr_y)
      train_accuracies.append(accuracy_score(curr_y, RForest.predict(curr_x)))
      test_accuracies.append(accuracy_score(y_test, RForest.predict(x_test)))

      for i in range(1, len(x_splits)):
        RForest = RandomForest(NumofTrees = ntrees, max_depth = ndepth)
        curr_x = np.concatenate((curr_x, x_splits[i]), axis=0)
        curr_y = np.concatenate((curr_y, y_splits[i]), axis=0)
        RForest.fit(curr_x, curr_y)

        train_accuracies.append(accuracy_score(curr_y, RForest.predict(curr_x)))

        test_accuracies.append(accuracy_score(y_test, RForest.predict(x_test)))

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
                          y_test=y_test_imdb, n_splits=5, ndepth = depth, ntrees = trees)






   


    
