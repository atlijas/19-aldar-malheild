from difflib import SequenceMatcher
import pickle
from islenska.bincompress import BinCompressed
import json
import os

current_path = os.path.basename(os.path.dirname(os.path.realpath(__file__)))


from .kvistur.kvistur import Kvistur

kv = Kvistur(f'{current_path}/kvistur/models/kvistur.hdf5', f'{current_path}/kvistur/models/chars.json')


bin_conn = BinCompressed()

def exists_in_old_words(token):
    return OLD_TREE.find(token, 0) != []


def exists_in_bin(token):
    return bin_conn.lookup(token) != [] or bin_conn.lookup(token.lower()) != [] or bin_conn.lookup(token.title()) != []

def exists_in_bin_or_old_words(token: str) -> bool:
    return (exists_in_bin(token)
            or exists_in_bin(token.lower())
            or exists_in_bin(token.title())
            or exists_in_old_words(token)
            or exists_in_old_words(token.lower())
            or exists_in_old_words(token.title()))

def all_parts_exist_in_bin(token: str) -> bool:
    if type(token) == str:
        token = [token]
    return (all([exists_in_bin(str(part)) for part in kv.split(token)[0].flatten()])
         or all([exists_in_bin(str(part)) for part in kv.split(token)[0].get_binary()]))

def load_tsv_as_dictionary(file):
    dictionary = {}
    with open(file, 'r', encoding='utf-8') as infile:
        split_lines = [line.split('\t') for line in infile.readlines()]
        for line in split_lines:
            if not any(char in line[2] for char in ['!', '[', 'None']):
                dictionary[line[0]] = line[2].strip()
    return dictionary

def load_pickle(file):
    with open(file, 'rb') as infile:
        return pickle.load(infile)

def get_similar_from_tree(token, tree, lev_dist=None):
    if lev_dist is None:
        lev_dist = 2 if len(token) > 12 else 1
    else:
        lev_dist = lev_dist
    return [tok for dist, tok in tree.find(token, n=lev_dist)]

def load_json(file):
    with open(file, 'r', encoding='utf-8') as infile:
        return json.load(infile)


JSON_EDITS = load_json(f'{current_path}/data/edits_10k.json')
DICTIONARY = load_tsv_as_dictionary(f'{current_path}/data/not_in_bin_10k.tsv')
BIN_TREE = load_pickle(f'{current_path}/data/bin_tree.pickle')
OLD_TREE = load_pickle(f'{current_path}/data/old_words.pickle')
DOUBLABLE_CONSONANTS = ['g', 'k', 'l', 'm', 'n']



if __name__ == '__main__':
    pass