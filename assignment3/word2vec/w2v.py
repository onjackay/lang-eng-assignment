import os
import time
import argparse
import string
from collections import defaultdict
import numpy as np
from sklearn.neighbors import NearestNeighbors

from tqdm import tqdm


"""
This file is part of the computer assignments for the course DD2418 Language engineering at KTH.
Created 2020 by Dmytro Kalpakchi.
"""


class Word2Vec(object):
    def __init__(self, filenames, dimension=300, window_size=2, nsample=10,
                 learning_rate=0.025, epochs=5, use_corrected=True, use_lr_scheduling=True):
        """
        Constructs a new instance.
        
        :param      filenames:      A list of filenames to be used as the training material
        :param      dimension:      The dimensionality of the word embeddings
        :param      window_size:    The size of the context window
        :param      nsample:        The number of negative samples to be chosen
        :param      learning_rate:  The learning rate
        :param      epochs:         A number of epochs
        :param      use_corrected:  An indicator of whether a corrected unigram distribution should be used
        """
        self.__pad_word = '<pad>'
        self.__sources = filenames
        self.__H = int(dimension)
        self.__lws = int(window_size)
        self.__rws = int(window_size)
        self.__C = self.__lws + self.__rws
        self.__init_lr = float(learning_rate)
        self.__lr = float(learning_rate)
        self.__nsample = int(nsample)
        self.__epochs = int(epochs)
        self.__nbrs = None
        self.__use_corrected = use_corrected
        self.__use_lr_scheduling = use_lr_scheduling


    def init_params(self, W, w2i, i2w):
        self.__W = W
        self.__w2i = w2i
        self.__i2w = i2w


    @property
    def vocab_size(self):
        return self.__V
        

    def clean_line(self, line):
        """
        The function takes a line from the text file as a string,
        removes all the punctuation and digits from it and returns
        all words in the cleaned line as a list
        
        :param      line:  The line
        :type       line:  str
        """
        line = "".join(list(filter(lambda x: x not in string.punctuation and x not in string.digits, line)))
        line = " ".join(line.split())
        return line


    def text_gen(self):
        """
        A generator function providing one cleaned line at a time

        This function reads every file from the source files line by
        line and returns a special kind of iterator, called
        generator, returning one cleaned line a time.

        If you are unfamiliar with Python's generators, please read
        more following these links:
        - https://docs.python.org/3/howto/functional.html#generators
        - https://wiki.python.org/moin/Generators
        """
        for fname in self.__sources:
            with open(fname, encoding='utf8', errors='ignore') as f:
                for line in f:
                    yield self.clean_line(line)


    def get_context(self, sent, i):
        """
        Returns the context of the word `sent[i]` as a list of word indices
        
        :param      sent:  The sentence
        :type       sent:  list
        :param      i:     Index of the focus word in the sentence
        :type       i:     int
        """
        context = []
        for j in range(max(0, i - self.__lws), min(len(sent), i + self.__rws + 1)):
            if j != i:
                context.append(self.__w2i[sent[i]])
        return context


    def skipgram_data(self):
        """
        A function preparing data for a skipgram word2vec model in 3 stages:
        1) Build the maps between words and indexes and vice versa
        2) Calculate the unigram distribution and corrected unigram distribution
           (the latter according to Mikolov's article)
        3) Return a tuple containing two lists:
            a) list of focus words
            b) list of respective context words
        """
        # Build the maps between words and indexes and vice versa
        self.__i2w = []
        self.__w2i = {}
        word_count = []

        for line in self.text_gen():
            for word in line.split():
                if word not in self.__w2i:
                    self.__i2w.append(word)
                    self.__w2i[word] = len(self.__i2w) - 1
                    word_count.append(0)
                idx = self.__w2i[word]
                word_count[idx] += 1
        self.__V = len(self.__i2w)

        # Calculate the unigram distribution and corrected unigram distribution
        self.__unigram = np.array(word_count) / np.sum(word_count)
        self.__corrected_unigram = self.__unigram ** 0.75 / np.sum(self.__unigram ** 0.75)

        # Build a list of focus words and a list of respective context words
        focus, context = [], []
        for line in self.text_gen():
            sent = line.split()
            for i in range(len(sent)):
                focus.append(self.__w2i[sent[i]])
                context.append(self.get_context(sent, i))

        return focus, context


    def sigmoid(self, x):
        """
        Computes a sigmoid function
        """
        return 1 / (1 + np.exp(-x))


    def negative_sampling(self, number, xb, pos):
        """
        Sample a `number` of negatives examples with the words in `xb` and `pos` words being
        in the taboo list, i.e. those should be replaced if sampled.
        
        :param      number:     The number of negative examples to be sampled
        :type       number:     int
        :param      xb:         The index of the current focus word
        :type       xb:         int
        :param      pos:        The index of the current positive example
        :type       pos:        int
        """
        valid_ids = np.setdiff1d(np.arange(self.__V), [xb, pos])
        p = self.__corrected_unigram[valid_ids]
        p /= np.sum(p)
        return np.random.choice(valid_ids, number, replace=False, p=p)


    def update(self, x, pos, negs):

        # Gradient for the input vector
        grad_in = self.__U[pos] * (self.sigmoid(np.dot(self.__U[pos], self.__W[x])) - 1) \
                + self.sigmoid(np.dot(self.__U[negs], self.__W[x])) @ self.__U[negs]
        
        # Gradient for the output vector of the positive example
        grad_out_pos = self.__W[x] * (self.sigmoid(np.dot(self.__U[pos], self.__W[x])) - 1)

        # Gradient for the output vector of the negative examples
        grad_out_negs =  np.outer(self.sigmoid(np.dot(self.__U[negs], self.__W[x])), self.__W[x])

        # Update
        self.__W[x] -= self.__lr * grad_in
        self.__U[pos] -= self.__lr * grad_out_pos
        self.__U[negs] -= self.__lr * grad_out_negs


    def train(self):
        """
        Performs the training of the word2vec skip-gram model
        """
        x, t = self.skipgram_data()
        N = len(x)
        print("Dataset contains {} datapoints".format(N))

        # REPLACE WITH YOUR RANDOM INITIALIZATION
        self.__W = np.random.normal(0, 0.1, (self.__V, self.__H))
        self.__U = np.random.normal(0, 0.1, (self.__V, self.__H))

        for ep in range(self.__epochs):
            for i in tqdm(range(N)):
                if self.__use_lr_scheduling:
                    self.__lr = self.__init_lr * max(1 - (ep * self.__epochs + i) / (N * self.__epochs + 1), 1e-4)

                for pos in t[i]:
                    negs = self.negative_sampling(self.__nsample, x[i], pos)
                    self.update(x[i], pos, negs)
            
            # Compute the loss
            loss = 0
            for i in range(N):
                for pos in t[i]:
                    loss += np.log(self.sigmoid(np.dot(self.__U[pos], self.__W[x[i]])))
            print("Epoch {}: loss = {}".format(ep, -loss / N))


    def find_nearest(self, words, metric, k=5):
        """
        Function returning k nearest neighbors with distances for each word in `words`
        
        We suggest using nearest neighbors implementation from scikit-learn 
        (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html). Check
        carefully their documentation regarding the parameters passed to the algorithm.
    
        To describe how the function operates, imagine you want to find 5 nearest neighbors for the words
        "Harry" and "Potter" using some distance metric `m`. 
        For that you would need to call `self.find_nearest(["Harry", "Potter"], k=5, metric='m')`.
        The output of the function would then be the following list of lists of tuples (LLT)
        (all words and distances are just example values):
    
        [[('Harry', 0.0), ('Hagrid', 0.07), ('Snape', 0.08), ('Dumbledore', 0.08), ('Hermione', 0.09)],
         [('Potter', 0.0), ('quickly', 0.21), ('asked', 0.22), ('lied', 0.23), ('okay', 0.24)]]
        
        The i-th element of the LLT would correspond to k nearest neighbors for the i-th word in the `words`
        list, provided as an argument. Each tuple contains a word and a similarity/distance metric.
        The tuples are sorted either by descending similarity or by ascending distance.
        
        :param      words:   Words for the nearest neighbors to be found
        :type       words:   list
        :param      metric:  The similarity/distance metric
        :type       metric:  string
        """
        results = []
        neigh = NearestNeighbors(n_neighbors=k, metric=metric)
        neigh = neigh.fit(self.__W)
        for word in words:
            if word not in self.__w2i:
                results.append([])
            else:
                distances, indices = neigh.kneighbors([self.__W[self.__w2i[word]]], n_neighbors=k, return_distance=True)
                results.append([(self.__i2w[idx], dis) for dis, idx in zip(distances[0], indices[0])])

        return results


    def write_to_file(self):
        """
        Write the model to a file `w2v.txt`
        """
        try:
            with open("w2v.txt", 'w') as f:
                W = self.__W
                f.write("{} {}\n".format(self.__V, self.__H))
                for i, w in enumerate(self.__i2w):
                    f.write(w + " " + " ".join(map(lambda x: "{0:.6f}".format(x), W[i,:])) + "\n")
        except:
            print("Error: failing to write model to the file")


    @classmethod
    def load(cls, fname):
        """
        Load the word2vec model from a file `fname`
        """
        w2v = None
        try:
            with open(fname, 'r') as f:
                V, H = (int(a) for a in next(f).split())
                w2v = cls([], dimension=H)

                W, i2w, w2i = np.zeros((V, H)), [], {}
                for i, line in enumerate(f):
                    parts = line.split()
                    word = parts[0].strip()
                    w2i[word] = i
                    W[i] = list(map(float, parts[1:]))
                    i2w.append(word)

                w2v.init_params(W, w2i, i2w)
        except:
            print("Error: failing to load the model to the file")
        return w2v


    def interact(self):
        """
        Interactive mode allowing a user to enter a number of space-separated words and
        get nearest 5 nearest neighbors for every word in the vector space
        """
        print("PRESS q FOR EXIT")
        text = input('> ')
        while text != 'q':
            text = text.split()
            neighbors = self.find_nearest(text, 'cosine')

            for w, n in zip(text, neighbors):
                print("Neighbors for {}: {}".format(w, n))
            text = input('> ')


    def train_and_persist(self):
        """
        Main function call to train word embeddings and being able to input
        example words interactively
        """
        self.train()
        self.write_to_file()
        self.interact()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='word2vec embeddings toolkit')
    parser.add_argument('-t', '--text', default='harry_potter_1.txt',
                        help='Comma-separated source text files to be trained on')
    parser.add_argument('-s', '--save', default='w2v.txt', help='Filename where word vectors are saved')
    parser.add_argument('-d', '--dimension', default=50, help='Dimensionality of word vectors')
    parser.add_argument('-ws', '--window-size', default=2, help='Context window size')
    parser.add_argument('-neg', '--negative_sample', default=10, help='Number of negative samples')
    parser.add_argument('-lr', '--learning-rate', default=0.025, help='Initial learning rate')
    parser.add_argument('-e', '--epochs', default=5, help='Number of epochs')
    parser.add_argument('-uc', '--use-corrected', action='store_true', default=True,
                        help="""An indicator of whether to use a corrected unigram distribution
                                for negative sampling""")
    parser.add_argument('-ulrs', '--use-learning-rate-scheduling', action='store_true', default=True,
                        help="An indicator of whether using the learning rate scheduling")
    args = parser.parse_args()

    if os.path.exists(args.save):
        w2v = Word2Vec.load(args.save)
        if w2v:
            w2v.interact()
    else:
        w2v = Word2Vec(
            args.text.split(','), dimension=args.dimension, window_size=args.window_size,
            nsample=args.negative_sample, learning_rate=args.learning_rate, epochs=args.epochs,
            use_corrected=args.use_corrected, use_lr_scheduling=args.use_learning_rate_scheduling
        )
        w2v.train_and_persist()
