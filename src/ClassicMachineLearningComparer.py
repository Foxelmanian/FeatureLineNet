from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_validate
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt

# Test algoirthms
class ClassicMachineLearningComparer(object):

    def __init__(self):
        pass
    @staticmethod
    def compare_machine_learning_alrogithm(df_y, df_x, percentage, plot_correlation_matrix=False):
        """ Compare several machine learning methods and plot the correlation matrix

            parm: Pandas Data Frame
            return: Pandas Data Frame
        """
        algorithms = [LinearSVC(),
                      #KNeighborsClassifier(3),
                      #SVC(kernel="linear", C=0.025),
                      #SVC(gamma=2, C=1),
                      DecisionTreeClassifier(max_depth=5),
                      #RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                      MLPClassifier(),
                      #AdaBoostClassifier(),
                      #GaussianNB(),
                      #QuadraticDiscriminantAnalysis()
                      ]

        if plot_correlation_matrix:
            correlation_matrix = df_x.corr()
            fig = plt.figure(figsize=(10, 7))
            ax = plt.axes()
            ax = sns.heatmap(correlation_matrix, mask=np.zeros_like(correlation_matrix, dtype=np.bool), cmap="bwr",
                             square=True, ax=ax)
            plt.show()
        # Sort to Feature and value
        n_perc = int(percentage * len(df_x[df_x.columns[1:]]))
        X = df_x
        y = df_y
        print(f'Used number of data: {n_perc}')
        def model_selection(X, y, n_splits=10, test_size=.2, train_size=.6, random_state=0):
            print('Start model selection')
            # Define the cross-validation split, leaving out 10%
            cv_split = ShuffleSplit(n_splits=n_splits, test_size=test_size,
                                    train_size=train_size, random_state=random_state)
            # Create a table to compare the algorithm's metrics and predictions
            columns = ['name', 'params', 'mean_train_accuracy', 'mean_test_accuracy', 'test_accuracy_3std', 'time']
            algorithm_comparison = pd.DataFrame(columns=columns)
            row_index = 0
            for alg in algorithms:
                print(alg)
                # Set name and parameters of the algorithm
                algorithm_name = alg.__class__.__name__
                algorithm_comparison.loc[row_index, 'name'] = algorithm_name
                algorithm_comparison.loc[row_index, 'params'] = str(alg.get_params())
                # Score model with cross validation using the accuracy metric
                cv_results = cross_validate(alg, X, y, cv=cv_split, scoring='accuracy')
                algorithm_comparison.loc[row_index, 'time'] = cv_results['fit_time'].mean()
                algorithm_comparison.loc[row_index, 'mean_train_accuracy'] = cv_results['train_score'].mean()
                algorithm_comparison.loc[row_index, 'mean_test_accuracy'] = cv_results['test_score'].mean()
                algorithm_comparison.loc[row_index, 'test_accuracy_3std'] = cv_results['test_score'].std() * 3
                row_index += 1

                print('Mean train accuracy', cv_results['train_score'].mean())
                print('mean_test_accuracy', cv_results['test_score'].mean())

            algorithm_comparison.sort_values(by=['mean_test_accuracy'], ascending=False, inplace=True)
            return algorithm_comparison
        dcsummary = model_selection(X, y)
        return dcsummary