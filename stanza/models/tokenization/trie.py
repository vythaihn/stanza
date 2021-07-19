from collections import defaultdict
import pickle
from conllu import parse_incr
import re
class Trie:
    """
    A simple Trie with add, search, and startsWith functions.
    """
    def __init__(self):
        self.root = defaultdict()

    def add(self, word):
        current = self.root
        for letter in word:
            current = current.setdefault(letter, {})
        current.setdefault("_end")

    def search(self, word):
        current = self.root
        for letter in word:
            if letter not in current:
                return False
            current = current[letter]
        if "_end" in current:
            return True
        return False

    def startsWith(self, prefix):
        current = self.root
        for letter in prefix:
            if letter not in current:
                return False
            current = current[letter]
        return True

def create_separabe_dict(lang, train_path, dict_path):
    tree = Trie()
    word_list = set()
    pattern_th = re.compile(r"(?:[^\d\W]+)|\s")
    dict_single = {}
    dict_multiple = {}
    #check if training file exists

    if train_path!=None:
        train_file = open(train_path, "r", encoding="utf-8")
        for tokenlist in parse_incr(train_file):
            for token in tokenlist:
                word = token['form'].lower()
                #check multiple_syllable word for vi
                if lang == "vi_vlsp":
                    if len(word.split(" "))>1 and any(map(str.isalpha, word)):
                        #do not include the words that contain numbers.
                        if not any(map(str.isdigit, word)):
                            #tree.add(word)
                            word_list.add(word)
                if lang == "th_orchid":
                    if len(word) > 1 and any(map(pattern_th.match, word)):
                        if not any(map(str.isdigit, word)):
                            #tree.add(word)
                            word_list.add(word)
                else:
                    if any(map(str.isalpha, word)) and not any(map(str.isdigit, word)):
                        if len(word) == 1:
                            dict_single[word] = dict_single.get(word,0)+1
                        if len(word) > 1:
                            dict_multiple[word[0]] = dict_multiple.get(word[0],0)+1

        margin = 5
        count = 0
        avg = 0
        for syllable in list(dict_single):
            if not (dict_single[syllable] > dict_multiple.get(syllable,0) + margin):
                del dict_single[syllable]
            else:
                dict_single[syllable] = dict_single.get(syllable) + dict_multiple.get(syllable,0)
                count += 1
                avg += dict_single[syllable]
        avg = avg/count
        count = 0
        for syllable in dict_single:
            if (dict_single[syllable] > avg):
                tree.add(syllable)
                count += 1
        print("Added ", count, " separable syllables found in training set to dictionary.")

    if count>0:
        with open(dict_path, 'wb') as config_dictionary_file:
            pickle.dump(tree, config_dictionary_file)
        config_dictionary_file.close()


def create_dictionary(lang, train_path, external_path, dict_path):
    tree = Trie()
    word_list = set()
    pattern_th = re.compile(r"(?:[^\d\W]+)|\s")

    #check if training file exists
    if train_path!=None:
        train_file = open(train_path, "r", encoding="utf-8")
        for tokenlist in parse_incr(train_file):
            for token in tokenlist:
                word = token['form'].lower()
                #check multiple_syllable word for vi
                if lang == "vi_vlsp":
                    if len(word.split(" "))>1 and any(map(str.isalpha, word)):
                        #do not include the words that contain numbers.
                        if not any(map(str.isdigit, word)):
                            tree.add(word)
                            word_list.add(word)
                if lang == "th_orchid":
                    if len(word) > 1 and any(map(pattern_th.match, word)):
                        if not any(map(str.isdigit, word)):
                            tree.add(word)
                            word_list.add(word)
                else:
                    if len(word)>1 and any(map(str.isalpha, word)):
                        if not any(map(str.isdigit, word)):
                            tree.add(word)
                            word_list.add(word)
        print("Added ", len(word_list), " words found in training set to dictionary.")
    if external_path != None:
        external_file = open(external_path, "r", encoding="utf-8")
        lines = external_file.readlines()
        for line in lines:
            word = line.lower()
            word = word.replace("\n","")
            # check multiple_syllable word for vi
            if lang == "vi_vlsp":
                if len(word.split(" "))>1 and any(map(str.isalpha, word)):
                    if not any(map(str.isdigit, word)):
                        tree.add(word)
                        word_list.add(word)
            if lang == "th_orchid":
                if len(word)>1 and any(map(pattern_th.match, word)):
                    if not any(map(str.isdigit, word)):
                        tree.add(word)
                        word_list.add(word)
            else:
                if len(word)>1 and any(map(str.isalpha, word)):
                    if not any(map(str.isdigit, word)):
                        tree.add(word)
                        word_list.add(word)

    if len(word_list)>0:
        with open(dict_path, 'wb') as config_dictionary_file:
            pickle.dump(tree, config_dictionary_file)
        config_dictionary_file.close()
        print("Succesfully generated dict file with total of ", len(word_list), " words.")

if __name__=='__main__':
    create_dictionary()
