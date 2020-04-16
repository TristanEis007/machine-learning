"""
The BRFA file contains objects that generate
a KNN and a Random Forest. See helper on each
object for more information.

This article was very helpful for creating
the DecisionTree and RandomForest classes:
https://towardsdatascience.com/random-forests-and-decision-trees-from-scratch-in-python-3e4fa5ae4249

Author: Tristan Eisenhart
"""

import numpy as np
import pandas as pd

class KNN():

    '''
    This object finds the K-nearest neighbors between
    a given observation x and a given data set. Use the
    find_knn function to get the results.

    Parameters
    ----------

    data_majority: numpy 2D array
        Data set with all majority observations

    k: int
        Number of nearest neighbors to return

    '''

    def __init__(self, data_majority, k):

        ''' Init func for class KNN '''

        self.data_majority = data_majority
        self.k = k


    def euclidean_distance(self, x, y):

        '''
        Returns the euclidean distance between
        rows x and y

        Parameters
        ----------

        x: Numpy array
            row of data x

        y: Numpy array
            row of data y

        '''

        return np.sqrt(sum(np.power(x - y, 2)))


    def find_knn(self, x):

        '''
        Computes the euclidean_distance function for x and all
        observations in the majority data set

        Parameters
        ----------

        x: Numpy array
            row of data x

        '''

        distances = list(map(lambda y: self.euclidean_distance(x, y),
                             np.array(self.data_majority)))
        nearest_index = np.argsort(distances)[0:self.k]

        return nearest_index


class RandomForest():

    '''

    This object creates a Random Forest made of
    a specified number of Decision Trees trained
    over random samples and features from the
    training set.

    The predict() function returns a prediction from
    each tree.

    Parameters
    ----------

    X : numpy 2D array
        Input data to train the forest on

    y: numpy array
        Output data from the training set

    n_trees: int
        Number of trees to generate in the forest

    n_features: int
        Number of features to randomely sample for
        in each generated tree

    sample_size: int
        Number of observations to randomely sample
        when training each tree

    depth: int
        Maximum number of splits to perform in
        each tree

    min_leaf: int
        Minimum number of observations required to
        make a split at a node for each tree

    debug: bool
        Ignore, this is used for debuging

    '''

    def __init__(self, X, y, n_trees, n_features,
                 sample_size=None, depth=10,
                 min_leaf=5, debug=False):

        ''' Init func for class RandomForest '''

        self.X = X
        self.y = np.array(y)

        assert self.X.shape[0] == self.y.shape[0]

        # Will tune those hyperparameters during training
        # in addition to n_trees (number of estimators)
        self.n_features = n_features
        self.depth = depth
        self.min_leaf = min_leaf
        # Using the data set size if no sized passed
        if sample_size == None:
            self.sample_size = X.shape[0]
        else:
            self.sample_size = sample_size

        self.i = 0
        self.debug = debug
        self.trees = [self.generate_tree() for i in range(n_trees)]


    def generate_tree(self):

        '''
        Function to initiate a decision tree. The `rand_ind` variable
        and `rand_features` variable are randomely sampled sets of
        observations and features, respectively, to train the tree on.

        DecisionTree is a Class of it's own, defined below.

        '''

        # Sampling observations with replacement
        rand_ind = np.random.choice(self.y.shape[0],
                                    size=self.sample_size,
                                    replace=True)

        rand_features = np.random.choice(self.X.shape[1],
                                         size=self.n_features,
                                         replace=False)

        self.i += 1

        print('Fitting tree num.{}, {:.1f}% observations with Ouctome = 1'\
              .format(self.i, 100*np.mean(self.y[rand_ind])))

        return DecisionTree(self.X.iloc[rand_ind],
                            self.y[rand_ind],
                            self.n_features,
                            rand_features,
                            ind=np.array(range(self.sample_size)),
                            depth=self.depth,
                            min_leaf=self.min_leaf,
                            debug=self.debug)


    def predict(self, x):

        ''' Predict an output for observation x '''

        return [t.predict(x) for t in self.trees]


class DecisionTree():

    '''

    This object creates Decision Trees every time
    a node creates a split, i.e. one Decision Tree
    for each split. The `run_split_logic()` function
    finds the optimal feature and threshold for each
    node. It also had built in logic to determine when
    a leaf node is reached. The function is ran in the
    init.

    The DecisionTree is called recursively anytime a
    leaf node is not reached. A leaf node is reached
    when there are less than min_leaf samples left
    or when the max depth is reached.

    The Gini Index is used to determine the optimal
    feature and threshold at every node. This loss
    function can easily be replaced by another one.

    Parameters
    ----------

    X : Pandas dataframe
        Input data to train the forest on

    y: Pandas Series
        Output data from the training set

    n_features: int
        Number of features to randomely sample for
        in each generated tree

    ind_features: numpy array
        Array containing the index of randomely
        sampled features for a specific decision tree

    ind: numpy array
        Array containing the index of randomely
        sampled observations from the original
        data set X to train a specific decision tree

    depth: int
        Maximum number of splits to perform in
        each tree

    min_leaf: int
        Minimum number of observations required to
        make a split at a node for each tree

    direction: str
        Ignore this argument. I used it do debug the
        tree

    debug: Bool
        Ignore this argument. I used it do debug the
        tree

    '''


    def __init__(self, X, y, n_features, ind_features,
                 ind, depth=10, min_leaf=5,
                 direction='C', debug=False):

        ''' Init func for class DecisionTree '''

        self.X = X
        self.y = y

        assert self.X.shape[0] == self.y.shape[0]

        self.ind = ind
        self.min_leaf = min_leaf
        self.ind_features = ind_features
        self.depth = depth
        self.direction = direction
        self.n_features = n_features
        self.val = np.argmax(np.bincount(np.array(y)[ind].astype(int)))
        self.score = np.inf
        self.debug = debug
        self.run_split_logic()


    def gini(self, y):

        '''
        Computes the Gini score for a given array of
        output. The Gini score is equal to:

            1 - sum(p_classes ^ 2)

        Where p_classes is an array containing the
        frequency of each class in the output data.

        Parameters
        ----------

        y: numpy array
            array of output over which to calculate the
            gini index

        '''

        _, counts = np.unique(y, return_counts = True)

        return 1 - np.sum(np.power(counts / np.sum(counts), 2))


    def run_split_logic(self):

        '''

        Function to run the split logic for a decision tree.
        The `find_optimal_split()` function performs a greedy 
        search, where it iteratively goes through each feature 
        in the set of randomely sampled features and computes
        the Gini score for each possible split threshold.

        Once the leaf node is reached, the functions returns. 

        '''

        if self.debug:
            print('This is node {}.{}'.format(self.depth, self.direction))

        # Starting by running the optimal split feature
        for feature in self.ind_features:
            self.find_optimal_split(feature)

        # Function returns if we've reached a leaf_node
        if self.is_leaf_node():

            if self.debug:
                print('Reached a leaf node with depth {}. Final score is {:.2f}'\
                      .format(self.depth, self.score))
            return

        if self.debug:
            print('Optimal split made on {} with threshold value {:.2f}'\
                  .format(self.X.columns[self.feature], self.split_threshold))

        # Pass the data that to the left / right branch
        # based on the threshold value
        x = self.X.values[self.ind, self.feature]
        lt_tree = np.where(x <= self.split_threshold)[0]
        rt_tree = np.where(x > self.split_threshold)[0]

        # Randomely sample features without replacement for 
        # initializing the next tree
        lt_rand_features = np.random.choice(self.X.shape[1],
                                            size=self.n_features,
                                            replace=False)
        rt_rand_features = np.random.choice(self.X.shape[1],
                                            size=self.n_features,
                                            replace=False)

        # Recursively calling the DecisionTree class on both
        # left and right branches as long as we haven't reached 
        # a leaf node
        self.lt_tree = DecisionTree(self.X,
                                    self.y,
                                    self.n_features, 
                                    lt_rand_features, 
                                    ind=self.ind[lt_tree],
                                    direction='L' + str(self.depth-1), # debug
                                    depth=self.depth - 1,
                                    min_leaf=self.min_leaf)
        self.rt_tree = DecisionTree(self.X,
                                    self.y,
                                    self.n_features,
                                    rt_rand_features,
                                    ind=self.ind[rt_tree],
                                    direction='R' + str(self.depth-1), # debug
                                    depth=self.depth - 1,
                                    min_leaf=self.min_leaf)
        

    def find_optimal_split(self, feature):

        '''
        This function iterates over all values for a 
        specified feature and updates the feature and
        threshold to use to make a split, if that split
        reduces the Gini value.

        Parameters
        ----------

        feature: int
            feature to run the split logic on

        '''

        x = np.array(self.X)[self.ind, feature]
        y = np.array(self.y)[self.ind]

        # Iterating over all observations in X
        for threshold in np.unique(x):

            lt_split_ind = np.where(x <= threshold)[0]
            rt_split_ind = np.where(x > threshold)[0]

            # If the split is smaller than min_leaf then use another
            # threshold value, i.e. skip this observation in the loop
            if len(lt_split_ind) < self.min_leaf or \
               len(rt_split_ind) < self.min_leaf:
                continue

            # Computing gini score for both left and right branches
            lt_gini = self.gini(y[lt_split_ind])
            rt_gini = self.gini(y[rt_split_ind])

            # Using a weighted gini score for making the split decision
            w_gini = (lt_gini * len(lt_split_ind) / len(self.ind)) + \
                     (rt_gini * len(rt_split_ind) / len(self.ind))

            # If the w_gini is < than the current score, then we
            # update the feature used to make the split as well as
            # the split_threshold
            if w_gini < self.score:
                self.feature = feature
                self.score = w_gini
                self.split_threshold = threshold


    def is_leaf_node(self):

        ''' Function to test if the node is a leaf node '''

        return self.depth <= 0 or self.score == np.inf


    def predict(self, X):

        ''' Function to run the prediction on all X values '''

        return np.array([self.predict_row(i) for i in X.values])


    def predict_row(self, i):

        ''' Function to run the prediction logice on a single value '''

        if self.is_leaf_node():
            return self.val
        else:
            if i[self.feature] <= self.split_threshold:
                t = self.lt_tree
            else:
                t = self.rt_tree

            return t.predict_row(i)


class BRAF_pipeline():

    '''
    Pipeline to run BRAF. This object runs the
    pseudo-code from "Biased Random Forest For
    Dealing With the Class Imbalance Problem".

    Please look at the helper for class KNN,
    DecisionTree and RandomForest for information
    on how each internal object is built.

    The  function `merge_predict()` is used to
    make a prediction using the trained Random
    Forests.

    Parameters
    ----------

    df: Pandas DataFrame
        Full data set to run BRAF on

    k: int
        Number of K-Nearest Neighbors to find

    p: float
        Proportion split to determine the number of
        estimators used for training each random
        forest

    s: int
        Number of estimators to use in the random forest.
        This number will be multiplied by p or (1-p)
    '''

    def __init__(self, df, k = 10, p = 0.5, s = 100):

        ''' Init funciton of BRAF_pipeline '''

        self.df = df
        self.k = k
        self.p = p
        self.s = s

        self.run_data_processing()
        self.run_random_forests()

    def run_data_processing(self):

        '''
        Function to process data as explained in
        the paper "Biased Random Forest For
        Dealing With the Class Imbalance Problem".

        '''

        self.X = self.df.loc[:, self.df.columns != 'Outcome']
        self.y = self.df.loc[:, 'Outcome']

        # Part 1: Split into a majority and
        # a minority set
        T_maj, T_min = self.split_maj_min()
        X_maj = T_maj.loc[:, T_maj.columns != 'Outcome']
        y_maj = T_maj.loc[:, 'Outcome']
        X_min = T_min.loc[:, T_min.columns != 'Outcome']
        y_min = T_min.loc[:, 'Outcome']

        del T_maj
        del T_min
        assert 'Outcome' not in X_maj.columns
        assert 'Outcome' not in X_min.columns

        # Part 2: For each observation from
        # T_min, find the K-nearest neighbors in
        # T_maj and add those unique neighbors to the
        # critical data set along with the minority
        # observation
        self.X_critical = self.generate_critical_set(X_maj, X_min)
        self.y_critical = self.df.loc[self.X_critical.index, 'Outcome'].astype(float)

        print('Critical data set generated using {} nearest neighbors. The set shape is {}'\
              .format(self.k, self.X_critical.shape))
        print('{:.1f}% observations with Outcome = 1 in the critical set'\
              .format(np.mean(self.y_critical) * 100))


    def run_random_forests(self):

        '''
        Function to run two random forests.

        The first random forest uses all observations
        from the original training set.

        The second random forest uses observations
        from the critical set.

        The number of estimators in each random forest
        is a function of s and p.

        '''

        # Part 3: Running a RF with observations from the
        # original data set. The size of the RF is equal
        # to s * (1-p).
        print('\nFitting RF with entire data set')
        self.rf_all = RandomForest(self.X, self.y,
                                   n_trees = int(self.s*(1-self.p)),
                                   n_features = len(self.X.columns)//3)

        # Part 4: Running a RF with observations from the
        # critical data set. The size of the RF is equal
        # to s * p.
        print('\nFitting RF with critical data set')
        self.rf_critical = RandomForest(self.X_critical, self.y_critical,
                                   n_trees = int(self.s*(self.p)),
                                   n_features = len(self.X.columns)//3)


    def split_maj_min(self, output='Outcome'):

        '''
        Split the data set into a majority and
        a minority set

        Parameters
        ----------

        output: String
            Column to use for the output

        '''

        assert output in self.df.columns

        maj = np.argmax(np.bincount(self.df['Outcome']))
        T_maj = self.df.loc[self.df[output]==maj, :]
        T_min = self.df.loc[self.df[output]!=maj, :]

        return T_maj, T_min


    def generate_critical_set(self, T_maj, T_min):

        '''
        For each observation from T_min,
        find the K-nearest neighbors in T_maj and
        add those unique neighbors to the critical
        data set along with the minority observation

        Parameters
        ----------

        T_maj: Pandas DataFrame
            Majority set

        T_min: Pandas DataFrame
            Minority set

        '''

        # Init KNN class
        knn = KNN(data_majority = T_maj, k = self.k)
        critical_ind = []
        T_critical = []
        # Looping over all minority observations
        for (n, minority_obs) in enumerate(np.array(T_min)):

            T_critical.append(T_min.iloc[n])
            # Finding k-nearest neighbors
            index_nn = knn.find_knn(minority_obs)

            # Looping over all nearest neighbors found
            for ind in index_nn:

                # Checking that neighbor is not already in
                # critical set
                if ind not in critical_ind:

                    # Adding unique majority observations to
                    # critical set
                    critical_ind.append(ind)
                    T_critical.append(T_maj.iloc[ind])

        return pd.DataFrame(T_critical)


    def merge_predict(self, x):

        '''
        Function to merge predictions from
        both random forest and to select the majority
        vote to determine the predicted class. Also
        returning the probability associated with
        the vote
        
        Parameters
        ----------

        x: Pandas DataFrame
            Data over which to run the prediction
            
        '''

        predict_all = self.rf_all.predict(x)
        predict_critical = self.rf_critical.predict(x)

        assert len(predict_all) == int(self.s * (1-self.p))
        assert len(predict_critical) == int(self.s * (self.p))

        all_ = predict_all, predict_critical

        majority_predict = [np.argmax(np.bincount(np.concatenate(all_)[:, i]))
                            for i in range(len(x))]
        majority_vote = [np.mean(np.concatenate(all_)[:, i])
                            for i in range(len(x))]

        proba = []
        for p, v in zip(majority_vote, majority_predict):
            if v == 0:
                proba.append(1-p)
            else:
                proba.append(p)

        return majority_predict, majority_vote
