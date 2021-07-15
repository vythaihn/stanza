"""
Very simple test of the mwt counting functionality in tokenization/data.py

TODO: could add a bunch more simple tests, including tests of reading
the data from a temp file, for example
"""

import pytest
import stanza

from stanza.tests import *
from stanza.models.tokenization.data import DataLoader
from stanza.models.tokenization.trie import Trie

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

# A single slice of the German tokenization data with no MWT in it
VI_DATA = [[('T', 0), ('h', 0), ('à', 0), ('n', 0), ('h', 0), (' ', 0), ('p', 0), ('h', 0), ('ố', 0), (' ', 1), ('H', 0), ('ồ', 0), (' ', 0), ('C', 0), ('h', 0), ('í', 0), (' ', 0), ('M', 0), ('i', 0), ('n', 0), ('h', 0), (' ', 1), ('t', 0), ('h', 0), ('ậ', 0), ('t', 0), (' ', 1), ('đ', 0), ('ẹ', 0), ('p', 0), (' ', 1), ('t', 0), ('u', 0), ('y', 0), ('ệ', 0), ('t', 0), (' ', 0), ('v', 0), ('ờ', 0), ('i', 0), ('!', 1)]]



FAKE_PROPERTIES_VI = {
    "lang":"vi",
    'feat_funcs': ("space_before","capitalized"),
    'max_seqlen': 300,
}

def test_vi_dict():

    """
    One dataset has no mwt, the other does
    """
    data = DataLoader(args=FAKE_PROPERTIES_VI, input_data=VI_DATA)
    words_check = ["nước hai","Nước Hai","Nguyễn Văn Đại","","","",""]
    #check for training dataset
    for word in words_check:
        assert data.dict_tree.search(word)

    #check for external training dataset
    words_check = ["","","","","","",""]
    for word in words_check:
        assert data.dict_tree.search(word)

