import os
import pickle
from parse_dataset import Dataset
from dep_parser import Parser
from logreg import LogisticRegression
import numpy as np

class TreeConstructor:
    """
    This class builds dependency trees and evaluates using unlabeled arc score (UAS) and sentence-level accuracy
    """
    def __init__(self, parser):
        self.__parser = parser

    def build(self, model, words, tags, ds):
        """
        Build the dependency tree using the logistic regression model `model` for the sentence containing
        `words` pos-tagged by `tags`
        
        :param      model:  The logistic regression model
        :param      words:  The words of the sentence
        :param      tags:   The POS-tags for the words of the sentence
        :param      ds:     Training dataset instance having the feature maps
        """
        #
        # YOUR CODE HERE
        #
        i, stack, pred_tree = 0, [], [0] * len(words)

        while True:
            feature = ds.get_features(words, tags, i, stack)

            x = ds.features2array(feature)

            probs = model.get_log_probs(x)
            moves = self.__parser.valid_moves(i, stack, pred_tree)

            y = None
            for move in moves:
                if y is None or probs[move] > probs[y]:
                    y = move

            if y is None:
                break
            i, stack, pred_tree = self.__parser.move(i, stack, pred_tree, y)

        return pred_tree

    def evaluate(self, model, test_file, ds):
        """
        Evaluate the model on the test file `test_file` using the feature representation given by the dataset `ds`
        
        :param      model:      The model to be evaluated
        :param      test_file:  The CONLL-U test file
        :param      ds:         Training dataset instance having the feature maps
        """
        #
        # YOUR CODE HERE
        #
        with open(test_file) as f:
            n_sentences = 0
            correct_sentences = 0
            n_arcs = 0
            correct_arcs = 0

            for w, tags, tree, relations in self.__parser.trees(f):
                pred_tree = self.build(model, w, tags, ds)

                for i in range(len(tree)):
                    if pred_tree[i] == tree[i]:
                        correct_arcs += 1
                    n_arcs += 1

                if pred_tree == tree:
                    correct_sentences += 1
                n_sentences += 1
                
            print(f"Sentence-level accuracy: {correct_sentences/n_sentences}")
            print(f"UAS: {correct_arcs/n_arcs}")


if __name__ == '__main__':

    # Create parser
    p = Parser()

    # Create training dataset
    ds = p.create_dataset("en-ud-train-projective.conllu", train=True)

    # Train LR model
    if os.path.exists('model.pkl'):
        # if model exists, load from file
        print("Loading existing model...")
        lr = pickle.load(open('model.pkl', 'rb'))
        ds.to_arrays()
    else:
        # train model using minibatch GD
        lr = LogisticRegression()
        lr.fit(*ds.to_arrays())
        pickle.dump(lr, open('model.pkl', 'wb'))
    
    # Create test dataset
    test_ds = p.create_dataset("en-ud-dev-projective.conllu")
    # Copy feature maps to ensure that test datapoints are encoded in the same way
    test_ds.copy_feature_maps(ds)
    # Compute move-level accuracy
    lr.classify_datapoints(*test_ds.to_arrays())
    
    # Compute UAS and sentence-level accuracy
    t = TreeConstructor(p)
    t.evaluate(lr, 'en-ud-dev-projective.conllu', ds)
