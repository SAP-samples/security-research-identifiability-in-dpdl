from abc import ABC, abstractmethod
import numpy as np
from random import randrange
class DataSet(ABC):
    """
    Abstract base class that defines the interface for all data sets used in an MIApplication.
    """

    @abstractmethod
    def load_data(self):
        """
        Loads the training set and the test set. Training and test set must not be iterators.
        Instead they are supposed to be lists or numpy arrays
        :return: two tuples --> (x_train, y_train), (x_test, y_test)
        """
        pass


class TwoClassUnbalancedDataSet(DataSet):
    """
    Abstract class to create an unbalanced training data set that consists of 
    one instance from the minority class and a plurarlity of instances from the majority class.
    """

    def __init__(self, minority_class, majority_class, num_instances):
        """
        :param minority_class: class to be reduced
        :param majority_class: class without reduction
        :param num_instances: final number of instances in the training set of majority class + minority class
        """
        super().__init__()
        self.minority_class = minority_class
        self.majority_class = majority_class
        self.num_instances = num_instances

    def load_data(self):
        """
        based on the original data set that is supposed to be balanced, this method reduces the
        number of training instances by the desired amount provided in the constructor.
        :return: two tuples --> (x_train, y_train), (x_test, y_test)
        """
        (x_train, y_train), (x_test, y_test) = self.load_balanced_data()

        mask_majority = y_train == self.majority_class
        mask_minority = y_train == self.minority_class

        if type(self.num_instances) == int:
            n = self.num_instances-1
            m = 1
        else:
            n = int(np.sum(mask_majority+mask_minority) * float(self.num_instances))
            m = int(np.sum(mask_majority+mask_minority) * 1-float(self.num_instances))
        
        x_train_reduced = x_train[~mask_minority][:m]
        y_train_reduced = y_train[~mask_minority][:m]

        x_train = np.concatenate([x_train[mask_majority][:n+m]])
        y_train = np.concatenate([y_train[mask_majority][:n+m]])

        x_train_alt = np.concatenate([x_train_reduced, x_train[mask_majority][:n]])
        y_train_alt = np.concatenate([y_train_reduced, y_train[mask_majority][:n]])
        
        return (x_train, y_train), (x_train_alt, y_train_alt), (x_test, y_test)

    @abstractmethod
    def load_balanced_data(self):
        """
        Loads the training set and the test set. Training and test set must not be iterators.
        Instead they are supposed to be lists or numpy arrays
        :return: two tuples --> (x_train, y_train), (x_test, y_test)
        """
        pass


class DataGenerator(ABC):
    """
    Every data generator passed to mia.core.model.Model must implement this interface
    """

    @abstractmethod
    def set_data(self, X, y):
        """
        Sets the data used for training and the batch size
        :param X_train:
        :param y_train:
        :return:
        """
        pass

