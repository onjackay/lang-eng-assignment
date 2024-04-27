from __future__ import print_function
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye, Patrik Jonell and Dmytro Kalpakchi.
"""

class BinaryLogisticRegression(object):
    """
    This class performs binary logistic regression using batch gradient descent
    or stochastic gradient descent
    """

    #  ------------- Hyperparameters ------------------ #

    LEARNING_RATE = 1e0  # The learning rate.
    CONVERGENCE_MARGIN = 0.001  # The convergence criterion.
    MAX_ITERATIONS = 100 # Maximal number of passes through the datapoints in stochastic gradient descent.
    MINIBATCH_SIZE = 100 # Minibatch size (only for minibatch gradient descent)

    # ----------------------------------------------------------------------


    def __init__(self, x=None, y=None, theta=None, lr=None):
        """
        Constructor. Imports the data and labels needed to build theta.

        @param x The input as a DATAPOINT*FEATURES array.
        @param y The labels as a DATAPOINT array.
        @param theta A ready-made model. (instead of x and y)
        """
        if not any([x, y, theta]) or all([x, y, theta]):
            raise Exception('You have to either give x and y or theta')

        if theta:
            self.FEATURES = len(theta)
            self.theta = theta

        elif x and y:
            # Number of datapoints.
            self.DATAPOINTS = len(x)

            # Number of features.
            self.FEATURES = len(x[0]) + 1

            print('Number of datapoints:', self.DATAPOINTS)
            print('Number of features:', self.FEATURES)

            # Encoding of the data points (as a DATAPOINTS x FEATURES size array).
            self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(x)), axis=1)

            # Correct labels for the datapoints.
            self.y = np.array(y)

            self.weights = np.where(self.y == 1, (self.DATAPOINTS - np.sum(y)) / self.DATAPOINTS, np.sum(y) / self.DATAPOINTS)

            # The weights we want to learn in the training phase.
            self.theta = np.random.uniform(-1, 1, self.FEATURES)

            # The current gradient.
            self.gradient = np.zeros(self.FEATURES)

            if lr is not None:
                self.LEARNING_RATE = lr

    # ----------------------------------------------------------------------


    def sigmoid(self, z):
        """
        The logistic function.
        """
        return 1.0 / ( 1 + np.exp(-z) )


    def conditional_prob(self, label, datapoint):
        """
        Computes the conditional probability P(label|datapoint)
        """

        # REPLACE THE COMMAND BELOW WITH YOUR CODE
        if label == 1:
            return self.sigmoid(np.dot(self.theta, self.x[datapoint]))
        else:
            return 1 - self.sigmoid(np.dot(self.theta, self.x[datapoint]))


    def compute_gradient_for_all(self):
        """
        Computes the gradient based on the entire dataset
        (used for batch gradient descent).
        """

        # YOUR CODE HERE
        h = self.sigmoid(self.x @ self.theta)
        self.gradient = np.dot(self.x.T, (h - self.y) * self.weights) / self.DATAPOINTS


    def compute_gradient_minibatch(self, minibatch):
        """
        Computes the gradient based on a minibatch
        (used for minibatch gradient descent).
        """
        
        # YOUR CODE HERE
        h = self.sigmoid(self.x[minibatch] @ self.theta)
        self.gradient = np.dot(self.x[minibatch].T, (h - self.y[minibatch]) * self.weights[minibatch]) / len(minibatch)


    def compute_gradient(self, datapoint):
        """
        Computes the gradient based on a single datapoint
        (used for stochastic gradient descent).
        """

        # YOUR CODE HERE
        h = self.sigmoid(np.dot(self.theta, self.x[datapoint]))
        self.gradient = self.x[datapoint] * (h - self.y[datapoint]) * self.weights[datapoint]

    
    def loss(self):
        """
        Computes the loss based on the entire dataset
        """
        # YOUR CODE HERE
        h = self.sigmoid(self.x @ self.theta)
        # print(self.x)
        # print(self.theta)
        # print(h)
        return -np.sum(self.y * np.log(h) + (1 - self.y) * np.log(1 - h)) / self.DATAPOINTS


    def stochastic_fit(self):
        """
        Performs Stochastic Gradient Descent.
        """
        self.init_plot(self.FEATURES)

        # YOUR CODE HERE
        for it in tqdm(range(self.MAX_ITERATIONS * 100)):
            datapoint = np.random.randint(self.DATAPOINTS)

            self.compute_gradient(datapoint)

            self.theta -= self.LEARNING_RATE * self.gradient

            if it % 100 == 0:
                self.compute_gradient_for_all()
                # self.update_plot(np.sum(np.square(self.gradient)))
                self.update_plot(self.loss())

                # if all(np.abs(self.gradient) < self.CONVERGENCE_MARGIN):
                #     print('Converged after {} iterations'.format(it))
                #     break


    def minibatch_fit(self):
        """
        Performs Mini-batch Gradient Descent.
        """
        self.init_plot(self.FEATURES)

        # YOUR CODE HERE
        for it in tqdm(range(self.MAX_ITERATIONS * 10)):
            minibatch = np.random.choice(self.DATAPOINTS, self.MINIBATCH_SIZE, replace=False)

            self.compute_gradient_minibatch(minibatch)

            self.theta -= self.LEARNING_RATE * self.gradient

            if it % 10 == 0:
                self.compute_gradient_for_all()
                # self.update_plot(np.sum(np.square(self.gradient)))
                self.update_plot(self.loss())
                # if all(np.abs(self.gradient) < self.CONVERGENCE_MARGIN):
                #     print('Converged after {} iterations'.format(it))
                #     break


    def fit(self):
        """
        Performs Batch Gradient Descent
        """
        self.init_plot(self.FEATURES)

        # YOUR CODE HERE
        for it in tqdm(range(self.MAX_ITERATIONS)):
            self.compute_gradient_for_all()
            # if all(np.abs(self.gradient) < self.CONVERGENCE_MARGIN):
            #     print('Converged after {} iterations'.format(it))
            #     break

            self.theta -= self.LEARNING_RATE * self.gradient
            
            # self.update_plot(np.sum(np.square(self.gradient)))
            self.update_plot(self.loss())


    def classify_datapoints(self, test_data, test_labels):
        """
        Classifies datapoints
        """
        print('Model parameters:');

        print('  '.join('{:d}: {:.4f}'.format(k, self.theta[k]) for k in range(self.FEATURES)))

        self.DATAPOINTS = len(test_data)

        self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(test_data)), axis=1)
        self.y = np.array(test_labels)
        confusion = np.zeros((self.FEATURES, self.FEATURES))

        for d in range(self.DATAPOINTS):
            prob = self.conditional_prob(1, d)
            predicted = 1 if prob > .5 else 0
            confusion[predicted][self.y[d]] += 1

        print('                       Real class')
        print('                 ', end='')
        print(' '.join('{:>8d}'.format(i) for i in range(2)))
        for i in range(2):
            if i == 0:
                print('Predicted class: {:2d} '.format(i), end='')
            else:
                print('                 {:2d} '.format(i), end='')
            print(' '.join('{:>8.3f}'.format(confusion[i][j]) for j in range(2)))


    def print_result(self):
        print(' '.join(['{:.2f}'.format(x) for x in self.theta]))
        print(' '.join(['{:.2f}'.format(x) for x in self.gradient]))


    # ----------------------------------------------------------------------

    def update_plot(self, *args):
        """
        Handles the plotting
        """
        if self.i == []:
            self.i = [0]
        else:
            self.i.append(self.i[-1] + 1)

        for index, val in enumerate(args):
            self.val[index].append(val)
            self.lines[index].set_xdata(self.i)
            self.lines[index].set_ydata(self.val[index])

        self.axes.set_xlim(0, max(self.i) * 1.5)
        self.axes.set_ylim(0, max(max(self.val)) * 1.5)

        plt.draw()
        plt.pause(1e-3)


    def init_plot(self, num_axes):
        """
        num_axes is the number of variables that should be plotted.
        """
        self.i = []
        self.val = []
        plt.ion()
        self.axes = plt.gca()
        self.lines =[]

        for i in range(num_axes):
            self.val.append([])
            self.lines.append([])
            self.lines[i], = self.axes.plot([], self.val[0], '-', c=[random.random() for _ in range(3)], linewidth=1.5, markersize=4)

    # ----------------------------------------------------------------------


def main():
    """
    Tests the code on a toy example.
    """
    x = [
        [ 1,1 ], [ 0,0 ], [ 1,0 ], [ 0,0 ], [ 0,0 ], [ 0,0 ],
        [ 0,0 ], [ 0,0 ], [ 1,1 ], [ 0,0 ], [ 0,0 ], [ 1,0 ],
        [ 1,0 ], [ 0,0 ], [ 1,1 ], [ 0,0 ], [ 1,0 ], [ 0,0 ]
    ]

    #  Encoding of the correct classes for the training material
    y = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0]
    b = BinaryLogisticRegression(x, y, lr=1e1)
    print(b.weights)
    b.fit()
    b.print_result()
    input("Press Enter to continue...")


if __name__ == '__main__':
    main()
