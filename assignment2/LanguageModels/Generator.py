import math
import argparse
import codecs
from collections import defaultdict
import random
import numpy as np

"""
This file is part of the computer assignments for the course DD2417 Language engineering at KTH.
Created 2018 by Johan Boye and Patrik Jonell.
"""

class Generator(object) :
    """
    This class generates words from a language model.
    """
    def __init__(self):
    
        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = {}

        # The bigram log-probabilities.
        self.bigram_prob = defaultdict(dict)

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

        # The average log-probability (= the estimation of the entropy) of the test corpus.
        self.logProb = 0

        # The identifier of the previous word processed in the test corpus. Is -1 if the last word was unknown.
        self.last_index = -1

        # The fraction of the probability mass given to unknown words.
        self.lambda3 = 1e-8

        # The fraction of the probability mass given to unigram probabilities.
        self.lambda2 = 0.01 - self.lambda3

        # The fraction of the probability mass given to bigram probabilities.
        self.lambda1 = 0.99

        # The number of words processed in the test corpus.
        self.test_words_processed = 0


    def read_model(self,filename):
        """
        Reads the contents of the language model file into the appropriate data structures.

        :param filename: The name of the language model file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """

        try:
            with codecs.open(filename, 'r', 'utf-8') as f:
                self.unique_words, self.total_words = map(int, f.readline().strip().split(' '))
                # YOUR CODE HERE
                # Read the unigram probabilities
                for i in range(self.unique_words):
                    line = f.readline().strip().split(' ')
                    self.index[line[1]] = int(line[0])
                    self.word[int(line[0])] = line[1]
                    self.unigram_count[int(line[0])] = int(line[2])

                # Read the bigram probabilities
                for line in f.readlines():
                    if line.strip() == "-1":
                        break
                    line = line.strip().split(' ')
                    self.bigram_prob[int(line[0])][int(line[1])] = float(line[2])
                return True
        except IOError:
            print("Couldn't find bigram probabilities file {}".format(filename))
            return False

    def generate(self, w, n):
        """
        Generates and prints n words, starting with the word w, and sampling from the distribution
        of the language model.
        """ 
        # YOUR CODE HERE
        for i in range(n):
            if w not in self.index:
                print(w)
                w = np.random.choice(list(self.index.keys()))
            else:
                print(w)
                curr_index = self.index[w]

                # Randomly select the next word
                words = self.bigram_prob[curr_index]
                if len(words) == 0:
                    w = np.random.choice(list(self.index.keys()))
                else:
                    p = [math.exp(words[index]) for index in words]
                    p = [x/sum(p) for x in p]
                    w = self.word[np.random.choice(list(words.keys()), p=p)]

    def linear_interpolation(self, first_index, second_index):
        if first_index in self.bigram_prob and second_index in self.bigram_prob[first_index]:
            p1 = math.exp(self.bigram_prob[first_index][second_index])
        else:
            p1 = 0

        if second_index in self.unigram_count:
            p2 = self.unigram_count[second_index] / self.total_words
        else:
            p2 = 0

        return self.lambda1 * p1 + self.lambda2 * p2 + self.lambda3
        

def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BigramTester')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file with language model')
    parser.add_argument('--start', '-s', type=str, required=True, help='starting word')
    parser.add_argument('--number_of_words', '-n', type=int, default=100)

    arguments = parser.parse_args()

    generator = Generator()
    generator.read_model(arguments.file)
    generator.generate(arguments.start,arguments.number_of_words)

if __name__ == "__main__":
    main()
