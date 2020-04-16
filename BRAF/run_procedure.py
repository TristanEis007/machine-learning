'''
This script contains the main() function to run 
the BRAF procedure. 

Flags that can be used include k, p, s

`k`, `p`, and `s` are the hyperparameter described in the
"Biased Random Forest for Dealing with the Class
Imbalance Problem" paper.

For information on the KNN, DecisionTree, RandomForest, 
and BRAF_pipeline objects, please see BRAF.py

Author: Tristan Eisenhart
'''

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BRAF import KNN, DecisionTree, RandomForest, BRAF_pipeline

# Importing metrics from Sklearn for calculating AUC
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int)
parser.add_argument("--p", type=float)
parser.add_argument("--s", type=int)

args = parser.parse_args()

if args.k:
    K = args.k
else:
    K = 10
if args.p:
    P = args.p
else:
    P = 0.5
if args.s:
    S = args.s
else:
    S = 100

print('\n----------------------------------------')
print('\nRunning procedure with K={}, P={}, S={}\n'.format(K, P, S))

def read_data(path=''):
    
    ''' 
    Read csv from path into a pandas dataframe
    
    Parameters
    ----------
    
    path: String
        Path to the csv file
    
    '''
    
    return pd.read_csv(path)


def split_train_test(df, spl=0.8):
    
    ''' 
    Function to split df into a
    training and a test set
    
    Parameters
    ----------
    
    df: Pandas DataFrame
        DataFrame to split into train/test sets

    spl: float
        Split between training and test set

    '''
    
    # Spliting between training and test set
    ind = np.arange(df.shape[0])
    train = np.random.choice(ind, 
                             int(spl*df.shape[0]), 
                             replace=False)
    
    test = np.setdiff1d(ind, train)
    df_train = df.loc[train, :].reset_index(drop = True)
    df_test = df.loc[test, :].reset_index(drop = True)
    
    assert len(set(test) - set(train)) == len(set(test))
    
    return df_train, df_test


def cross_val_training(X, k_fold = 10):

    '''
    Function to do cross validation using the BRAF_pipeline
    class. This function performs k-fold cross validation
    and outputs key evaluation metrics after each fold

    Parameters
    ----------
    
    X: Pandas DataFrame
        Data Frame with all training data

    k_fold: int
        Number of folds to use when performing cross validation

    '''
    
    ind = np.random.permutation(X.index)
    i, j = 0, len(ind) // k_fold

    precision_l = []
    recall_l = []
    auprc_l = []
    auroc_l = []
    
    for k in range(k_fold):
        
        # Spliting data set into k folds
        ind_val = ind[i+k:j+k]
        ind_train = np.setdiff1d(ind, ind_val)
        
        # Using 1 fold for validation and k-1 fold for training
        val_set = X.loc[ind_val, :].reset_index(drop = True)
        train_set = X.loc[ind_train, :].reset_index(drop = True)
        
        print('----- Training on fold {} -----'.format(k))
        # Running the BRAF pipeline. This pipeline
        # is described in detail in the BRAF.py
        braf_algo = BRAF_pipeline(train_set, k=K, p=P, s=S)
        val_pred, val_prob = braf_algo.merge_predict(val_set.loc[:, val_set.columns != 'Outcome'])
        
        print('\nEvaluation Metrics for Fold {}'.format(k))
        # Computing evaluation metrics
        precision, recall, area, score = metrics(val_set['Outcome'], val_pred, val_prob)
        precision_l.append(precision)
        recall_l.append(recall)
        auprc_l.append(area)
        auroc_l.append(score)
        plot_ROC(val_set['Outcome'], val_prob, 'ROC_CVfold_{}'.format(k))
        plot_PRC(val_set['Outcome'], val_prob, 'PRC_CVfold_{}'.format(k))
    
    print('--------------------------------------------') 
    print('----------- METRICS FROM TRAINING ----------')
    print('\nMean CV Precision is {:.2f} and CV Recall is {:.2f}'\
          .format(np.mean(precision_l), np.mean(recall_l)))
    print('AUPRC is {:.2f}'.format(np.mean(auprc_l)))
    print('AUROC is {:.2f}'.format(np.mean(auroc_l)))
    print('--------------------------------------------')
    print('--------------------------------------------\n')
    print('Finished cross validation & training\n')


def metrics(true, pred, prob):

    '''
    Function to compute key evaluation metrics.

    Parameters
    ----------
    
    true: numpy array
        Array of true outputs

    pred: numpy array
        Array of predicted output

    prob: numpy array
        Array with class score

    '''
    
    t_pos = \
        len([a for a, p in zip(true, pred) if a == p and p == 1])
    t_neg = \
        len([a for a, p in zip(true, pred) if a == p and p == 0])
    f_pos = \
        len([a for a, p in zip(true, pred) if a != p and p == 1])
    f_neg = \
        len([a for a, p in zip(true, pred) if a != p and p == 0])
    
    precision = t_pos / (t_pos + f_pos)
    recall = t_pos / (t_pos + f_neg)
    
    print('Precision is {:.2f} and Recall is {:.2f}'.format(precision, recall))
            
    prec_c, rec_c, _ = precision_recall_curve(true, prob)
    area = auc(rec_c, prec_c)
    
    print('AUPRC is {:.2f}'.format(area))

    score = roc_auc_score(true, prob)
    print('AUROC is {:.2f}'.format(score))
    print('\n')

    return precision, recall, area, score

    
def plot_ROC(true, prob, name):
    
    '''
    Function to plot ROC and save it to disk.
    
    Parameters
    ----------
    
    true: numpy array
        Array of true outputs

    prob: numpy array
        Array with class score

    name: str
        Name of figure saved on disk
        
    '''
    
    false_pos, true_pos, _ = roc_curve(true, prob)
    roc_auc = auc(false_pos, true_pos)
    
    plt.title(name)
    plt.plot(false_pos, true_pos, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(name)
    plt.close()


def plot_PRC(true, prob, name):
    
    '''
    Function to plot PRC and save it to disk
    
    '''
    
    lr_precision, lr_recall, _ = precision_recall_curve(true, prob)
    
    plt.title(name)
    plt.plot(lr_precision, lr_recall, '-.')
    plt.xlim([0, 1])
    plt.ylim([0, 1.01])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.savefig(name)
    plt.close()


def main():

    ''' Main '''

    # Reading data
    df = read_data()
    
    # Split into Training and Test Sets
    df_train, df_test = split_train_test(df)
    
    # Performing CV training
    cross_val_training(df_train)

    # Retraining on full training dataset & testing on hold-out test set
    print('---------------------------------------------')
    print('\nTraining on entire training set and testing on hold-out test set')
    braf_algo = BRAF_pipeline(df_train, k=K, p=P, s=S)

    print('---------------------------------------------\n')
    print('Done Training. Testing on test set and outputing evaluation metrics')
    pred, prob = braf_algo.merge_predict(df_test.loc[:, df_test.columns != 'Outcome'])
    precision, recall, area, score = metrics(df_test['Outcome'], pred, prob)

    print('---------------------------------------------')
    print('----------- METRICS ON TEST SET -------------')
    print('Precision is {:.2f} and Recall is {:.2f}'\
          .format(np.mean(precision), np.mean(recall)))

    print('AUPRC is {:.2f}\n'.format(np.mean(area)))
    print('AUROC is {:.2f}'.format(np.mean(score)))
    print('---------------------------------------------')
    print('---------------------------------------------\n')
    plot_ROC(df_test['Outcome'], prob, 'ROC_test')
    plot_PRC(df_test['Outcome'], prob, 'PRC_test')
    print('\nAll done. The hyperparameters used were:')
    print('K={}, P={}, S={}\n'.format(K, P, S))


if __name__ == '__main__':
    main()
