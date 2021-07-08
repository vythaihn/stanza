from collections import defaultdict
import pickle

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

def main():
    tree_1 = Trie()
    tree_2 = Trie()
    f = open("dict.txt", "r")
    lines = f.readlines()


    start_syllable = set([word.split()[0] for word in lines])
    for syl in start_syllable:
        tree_1.insert(syl)

    end_syllable = set([word.split()[-1] for word in lines])
    for syl in end_syllable:
        tree_2.insert(syl)

    with open('vi-end.dictionary', 'wb') as config_dictionary_file:
        pickle.dump(tree_2, config_dictionary_file)
    with open('vi-start.dictionary', 'wb') as config_dictionary_file_2:
        pickle.dump(tree_1, config_dictionary_file_2)

    config_dictionary_file.close()
    config_dictionary_file_2.close()
    print("Succesfully generated dict files!")
    
if __name__=='__main__':
    main()
