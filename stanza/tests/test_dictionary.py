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
VI_DATA = [[('T', 0), ('h', 0), ('à', 0), ('n', 0), ('h', 0), (' ', 0), ('p', 0), ('h', 0), ('ố', 1), (' ', 0), ('H', 0), ('ồ', 0), (' ', 0), ('C', 0), ('h', 0), ('í', 0), (' ', 0), ('M', 0), ('i', 0), ('n', 0), ('h', 1), (' ', 1), ('t', 0), ('h', 0), ('ậ', 0), ('t', 1), (' ', 0), ('đ', 0), ('ẹ', 0), ('p', 1), (' ', 0), ('t', 0), ('u', 0), ('y', 0), ('ệ', 0), ('t', 0), (' ', 0), ('v', 0), ('ờ', 0), ('i', 1), ('!', 1)]]



FAKE_PROPERTIES_VI = {
    "lang":"vi",
    'shorthand':'vi_vlsp',
    'feat_funcs': ("space_before","capitalized",'all_caps','numeric'),
    'max_seqlen': 300,
    'dict_feat':20,
}
FAKE_PROPERTIES_TH = {
    'lang':'th',
    'shorthand':'th_orchid',
    'feat_funcs':('space_before', "capitalized", 'numeric'),
    'max_seqlen':300,
    'dict_feat':20,
}
"""
def test_vi_dict():

   
    #One dataset has no mwt, the other does
    
    data = DataLoader(args=FAKE_PROPERTIES_VI, input_data=VI_DATA)
    words_check = ["nước hai","Nước Hai","Nguyễn Văn Đại", "500.000"]

    #check for training dataset
    for word in words_check:
        if data.dict_tree.search(word):
            print("These words are in dev set, not supposed in dict!")
        assert not data.dict_tree.search(word)

    words_check = ["20-9"]
    if data.dict_tree.search(words_check[0]):
        print("not training dict passed!")
    assert not data.dict_tree.search(words_check[0])

    #check for external training dataset
    words_check = ["chối bay chối"]
    for word in words_check:
        if not data.dict_tree.search(word):
            print("not external passed!")
        assert data.dict_tree.search(word)


    print("Passed the test!")
"""


def test_th_dict():
    data = DataLoader(args=FAKE_PROPERTIES_TH, input_data=VI_DATA)
    words_check = ["องค์การระหว่างประเทศว่าด้วยมาตรฐาน(ISO)","information coding","โปรแกรมมิงก์"]
    for word in words_check:
        if data.dict_tree.search(word):
            print("These words are in dev set, not supposed here!")
        assert not data.dict_tree.search(word)
    words_check = ["Cache Memory Controller"]
    if data.dict_tree.search(words_check[0]):
        print("not training dict passed")
    assert not data.dict_tree.search(words_check[0])

    words_check = ["มหาวิทยาลัยเทคโนโลยีสุรนารี","มหิตลาธิเบศรอดุลยเดชวิกรม"]
    for word in words_check:
        if not data.dict_tree.search(word):
            print("not external passed")
        assert data.dict_tree.search(word)
        
