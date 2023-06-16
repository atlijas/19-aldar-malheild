import difflib
from fuzzywuzzy import fuzz
from difflib import SequenceMatcher
from Levenshtein import distance
import re

def read_file(file):
    with open(file, 'r', encoding='utf-8') as infile:
        return infile.read()

def find_most_similar_sequence(correct_sentence, search_index_start, search_index_end, ocr_string):
    sequences = []
    for i in range(len(ocr_string) - len(correct_sentence) + 1):
        for j in range(5):
            sequences.append(ocr_string[i:i+len(correct_sentence)+j])
    return max(sequences, key=lambda x: fuzz.ratio(correct_sentence, x))


if __name__ == '__main__':
    ground_truth = read_file('output_files/modernized/Sudurland_1910-1912-06-08/10 - 8. tölublað/3.txt')
    ocr = read_file('output_files/original/Sudurland_1910-1912-06-08/10 - 8. tölublað/3.txt')
    ground_truth_lines = ground_truth.splitlines()
    for line in ground_truth_lines:
        start_index_in_ground_truth = ground_truth.find(line)
        end_index_in_ground_truth = start_index_in_ground_truth + len(line)
        ocr_search_window_indices = [start_index_in_ground_truth - 20, end_index_in_ground_truth + 20]
        most_similar_sentence = find_most_similar_sequence(line, ocr_search_window_indices[0], ocr_search_window_indices[1], ocr)
        print(line, '###', most_similar_sentence.replace('\n', ' '))
        print()
