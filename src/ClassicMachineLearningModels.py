
# 2. Run classic machine learning models for comparing the results
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
from .ClassicMachineLearningComparer import ClassicMachineLearningComparer


class ClassicMachineLearningModels(object):

    def __init__(self, train_data, train_labels, val_data, val_labels):
        self.__train_data = train_data
        self.__train_labels = train_labels
        self.__val_data = val_data
        self.__val_labels = val_labels

    def __compare_several_models(self):
        summary = ClassicMachineLearningComparer.compare_machine_learning_alrogithm(df=df_basic, percentage=1.0)
        summary.plot(kind='bar')
        print(summary)

    def train_model(self, model_type='KNN'):
        if model_type == 'KNN':
            model = KNeighborsClassifier(3)
        elif model_type == 'SVC':
            model = SVC(kernel="linear", C=0.025)
        else:
            raise AttributeError('Unsupported model type')
        model.fit(self.__train_data, self.__val_data)
        y_pred = model.predict(self.__val_data)
        print(f'Accuracy basic model: {accuracy_score(self.__val_labels, y_pred)}')
        return model

