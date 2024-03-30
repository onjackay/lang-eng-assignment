from terminaltables import AsciiTable
import argparse

"""
The CKY parsing algorithm.

This file is part of the computer assignments for the course DD2418 Language engineering at KTH.
Created 2019 by Johan Boye.
"""

class CKY :

    # The unary rules as a dictionary from words to non-terminals,
    # e.g. { cuts : [Noun, Verb] }
    unary_rules = {}

    # The binary rules as a dictionary of dictionaries. A rule
    # S->NP,VP would result in the structure:
    # { NP : {VP : [S]}} 
    binary_rules = {}

    # The parsing table
    table = []

    # The backpointers in the parsing table
    backptr = []

    # The words of the input sentence
    words = []


    # Reads the grammar file and initializes the 'unary_rules' and
    # 'binary_rules' dictionaries
    def __init__(self, grammar_file) :
        stream = open( grammar_file, mode='r', encoding='utf8' )
        for line in stream :
            rule = line.split("->")
            left = rule[0].strip()
            right = rule[1].split(',')
            if len(right) == 2 :
                # A binary rule
                first = right[0].strip()
                second = right[1].strip()
                if first in self.binary_rules :
                    first_rules = self.binary_rules[first]
                else :
                    first_rules = {}
                    self.binary_rules[first] = first_rules
                if second in first_rules :
                    second_rules = first_rules[second]
                    if left not in second_rules :
                        second_rules.append( left )
                else :
                    second_rules = [left]
                    first_rules[second] = second_rules
            if len(right) == 1 :
                # A unary rule
                word = right[0].strip()
                if word in self.unary_rules :
                    word_rules = self.unary_rules[word]
                    if left not in word_rules :
                        word_rules.append( left )
                else :
                    word_rules = [left]
                    self.unary_rules[word] = word_rules

    # Parses the sentence a and computes all the cells in the
    # parse table, and all the backpointers in the table
    def parse(self, s) :
        self.words = s.split()        
        #
        #  YOUR CODE HERE
        #
        self.table = [[[] for i in range(len(self.words))] for j in range(len(self.words))]
        self.backptr = [[[] for i in range(len(self.words))] for j in range(len(self.words))]

        for i in range(len(self.words)):
            if self.words[i] in self.unary_rules:
                for key in self.unary_rules[self.words[i]]:
                    self.table[i][i].append(key)

        for k in range(1, len(self.words)):
            for i in range(len(self.words) - k):
                j = i + k
                for l in range(i, j):
                    for key1 in self.table[i][l]:
                        if key1 not in self.binary_rules:
                            continue
                        for key2 in self.table[l + 1][j]:
                            if key2 not in self.binary_rules[key1]:
                                continue
                            for key3 in self.binary_rules[key1][key2]:
                                self.table[i][j].append(key3)
                                self.backptr[i][j].append((l, key1, key2))

    # Prints the parse table
    def print_table( self ) :
        t = AsciiTable(self.table)
        t.inner_heading_row_border = False
        print( t.table )


    # Prints all parse trees derivable from cell in row 'row' and
    # column 'column', rooted with the symbol 'symbol'
    def print_trees( self, row, column, symbol ) :
        #
        #  YOUR CODE HERE
        #
        results = self.get_tree_string(row, column, symbol)
        for result in results:
            print(result)

    # Returns a list of string representation of the parse tree
    def get_tree_string( self, row, column, symbol ) :
        results = []
        if row == column:
            results.append(f'{symbol}({self.words[row]})')
        else:
            for i, key in enumerate(self.table[row][column]):
                if key == symbol:
                    k = self.backptr[row][column][i][0]
                    left = self.get_tree_string(row, k, self.backptr[row][column][i][1])
                    right = self.get_tree_string(k + 1, column, self.backptr[row][column][i][2])
                    for l in left:
                        for r in right:
                            results.append(f'{symbol}({l}, {r})')
        return results


def main() :

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CKY parser')
    parser.add_argument('--grammar', '-g', type=str,  required=True, help='The grammar describing legal sentences.')
    parser.add_argument('--input_sentence', '-i', type=str, required=True, help='The sentence to be parsed.')
    parser.add_argument('--print_parsetable', '-pp', action='store_true', help='Print parsetable')
    parser.add_argument('--print_trees', '-pt', action='store_true', help='Print trees')
    parser.add_argument('--symbol', '-s', type=str, default='S', help='Root symbol')

    arguments = parser.parse_args()

    cky = CKY( arguments.grammar )
    cky.parse( arguments.input_sentence )
    if arguments.print_parsetable :
        cky.print_table()
    if arguments.print_trees :
        # cky.print_trees( len(cky.words)-1, 0, arguments.symbol )
        cky.print_trees( 0, len(cky.words)-1, arguments.symbol )
    

if __name__ == '__main__' :
    main()    


                        
                        
                        
                    
                
                    

                
        
    
