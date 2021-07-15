from collections import defaultdict
import pickle
from conllu import parse_incr
class Trie:
    """
    Implement a trie with insert, search, and startsWith methods.
    """
    def __init__(self):
        self.root = defaultdict()

    # @param {string} word
    # @return {void}
    # Inserts a word into the trie.
    def insert(self, word):
        current = self.root
        for letter in word:
            current = current.setdefault(letter, {})
        current.setdefault("_end")

    # @param {string} word
    # @return {boolean}
    # Returns if the word is in the trie.
    def search(self, word):
        current = self.root
        for letter in word:
            if letter not in current:
                return False
            current = current[letter]
        if "_end" in current:
            return True
        return False

    # @param {string} prefix
    # @return {boolean}
    # Returns if there is any word in the trie
    # that starts with the given prefix.
    def startsWith(self, prefix):
        current = self.root
        for letter in prefix:
            if letter not in current:
                return False
            current = current[letter]
        return True

def create_dictionary(train_path, external_path, dict_path):
    tree = Trie()
    count = 0
    word_list = ()
    if train_path!=None:
        train_file = open(train_path, "r", encoding="utf-8")
        for tokenlist in parse_incr(train_file):
            for token in tokenlist:
                word = token['form']
                word = word.lower()
                if len(word)>1:
                    if not any(map(str.isdigit, word)):
                        tree.insert(word)
                        word_list.add(word)
    if external_path != None:
        external_file = open(external_path, "r", encoding="utf-8")
        lines = external_file.readlines()
        for line in lines:
            word = line.replace("\n","")
            if len(word)>1:
                if not any(map(str.isdigit, word)):
                    tree.insert(word)
                    word_list.add(word)

    if len(word_list)>0:
        with open(dict_path, 'wb') as config_dictionary_file:
            pickle.dump(tree, config_dictionary_file)
        config_dictionary_file.close()
        print("Succesfully generated dict file with total of ", len(word_list), " words.")
