#  -*- coding: utf-8 -*-
import math
import argparse
import nltk
import codecs
from collections import defaultdict

"""
This file is part of the computer assignments for the course DD2417 Language engineering at KTH.
Created 2017 by Johan Boye and Patrik Jonell.
"""

class BigramTester(object):
    def __init__(self):
        """
        This class reads a language model file and a test file, and computes
        the entropy of the latter. 
        """
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
        self.lambda3 = 0.000001

        # The fraction of the probability mass given to unigram probabilities.
        self.lambda2 = 0.01 - self.lambda3

        # The fraction of the probability mass given to bigram probabilities.
        self.lambda1 = 0.99

        # The number of words processed in the test corpus.
        self.test_words_processed = 0


    def read_model(self, filename):
        """
        Reads the contents of the language model file into the appropriate data structures.

        :param filename: The name of the language model file.
        :return: True if the entire file could be processed, False otherwise.
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


    def compute_entropy_cumulatively(self, word):
        # YOUR CODE HERE
        index = self.index.get(word, -1)
        self.logProb += math.log(self.linear_interpolation(self.last_index, index))
        self.last_index = index
        self.test_words_processed += 1

    def process_test_file(self, test_filename):
        """
        <p>Reads and processes the test file one word at a time. </p>

        :param test_filename: The name of the test corpus file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """
        try:
            with codecs.open(test_filename, 'r', 'utf-8') as f:
                self.tokens = nltk.word_tokenize(f.read().lower()) 
                for token in self.tokens:
                    self.compute_entropy_cumulatively(token)
                self.logProb /= -len(self.tokens)
            return True
        except IOError:
            print('Error reading testfile')
            return False

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
    parser.add_argument('--test_corpus', '-t', type=str, required=True, help='test corpus')

    arguments = parser.parse_args()

    bigram_tester = BigramTester()
    bigram_tester.read_model(arguments.file)
    bigram_tester.process_test_file(arguments.test_corpus)
    print('Read {0:d} words. Estimated entropy: {1:.2f}'.format(bigram_tester.test_words_processed, bigram_tester.logProb))

if __name__ == "__main__":
    main()