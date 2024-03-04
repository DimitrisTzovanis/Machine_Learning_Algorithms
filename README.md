# Machine Learning Algorithms


## Members: Dimitris Tzovanis, Elpida Stasinou


### Parts
- We created two AI algorithms for part A, ID3 and random forest in python code, which uses id3 to generate trees.
- For part B we also created the id3 and random forest algorithms using the sklearn library to compare the results (accuracy, loss).
- For part C we created an RNN neural network.
- 
For part C code is delivered as a python file, but you can pass it to an ipynb file to run it faster
All algorithms use the imdb dataset to learn mechanically. That is, they use 25000 train sample reviews 
to train themselves to understand which reviews are positive and which are negative. 
We then test their performance by feeding them 25000 test samples, where they guess which ones are positive or negative.
After this process the results are printed: accuracy score, precision, recall, 
F1 score and training, testing accuracy in the form of a curve

### Part A


#### ID3:
This algorithm consists of 2 classes, Node and ID3 itself. Node is a helper class for creating the nodes of the 
binary tree that ID3 will create.
First init is called to create the id3 tree. There we pass variables like min_samples 
(the smallest possible number of samples a node of the tree can have in order to create the 2 
children under it) and max_depth (maximum the tree can reach)
Then the fit is called, which counts the number of properties and calls the grow_tree function, 
which is the most basic function of the algorithm.
The grow_tree first lists the properties and the samples of each property. 
Then it makes the following checks to see if it needs to stop: a) we have reached maximum depth, 
b) too few samples to break the node into children, 
c) all samples are identical on a node so we can't break. 
Then we take all the properties in random order and try to find the best property to break and the best 
point of that property by calling best_criteria ( which in turn calls information_gain).
When these values are found, we split the data based on the breakpoint in that property by calling the 
split function which returns the number of samples the left and right child of the node will have.
Finally, we recursively call grow_tree on each child and create an object of class node for the node we just analyzed
The best_criteria is called from grow_tree and it traces each property and each possible breakpoint to find the best combination. 
For each combination it calls IG to find the information gain and and to 
compare it with previous ones to find the maximum. It eventually returns the best breakpoint and the best property
IG finds the information gain. That is, for the given property and breakpoint, it calls split to split the data and 
calls the entropy function (calculates the entropy) 
on each child as well as on the node itself. The information gain is the result of: ig = entropy(parent) - [weightedAvg]*entropy(children)
Entropy simply calculates entropy via the math formula - np.sum(x * math.log2(x)) where x is each property (numpy library)
At this point the fit is finished and we can call predict which for each test sample calls traverse_tree
Traverse_tree starts at the root and looks to see if the sample will go to the left or right child depending on the breakpoint 
and the property it is testing, then recursively calls itself on the corresponding child until it reaches gender, i.e. we have a value


#### Random Forest:
We decided to create the random forest algorithm since it builds on id3.
Therefore we decided to decide to use random forest to build the random forest algorithm.
Similarly there is the init that stores values like number of trees, min samples, max depth as before
The fit creates as many id3 trees as NumofTrees and gives each child random properties (xtrain, ytrain), where it calls the fit of id3
Predict goes to all trees and calls the corresponding id3 predict and for each prediction finds the most frequent answer.
For the random forest algorithm we set the depth to 5 and the number of trees to 4 because it achieves maximum accuracy (0.68) in a reasonable amount of time
