from __future__ import annotations
from typing import TYPE_CHECKING
from string import punctuation as p
from .lexicon_lookup import exists_in_bin_or_old_words
from difflib import SequenceMatcher
from tokenizer import correct_spaces
import re
from .months import MONTHS
from wtpsplit import WtP
from torch import cuda

if TYPE_CHECKING:
    from fairseq.models.transformer import TransformerModel
    from collections.abc import Generator

extended_punctuation = p + "–„”“—«»"

HYPHENEND_TOKEN = '-HYPHENEND'
LINE_END_REGEX_PATTERN = re.compile('[^\dA-ZÁÉÍÓÚÝÞÆÖ]\.')
MONTHS_DATE_REGEX = re.compile('(?:\d{1,3}\.\s)(?:' + '|'.join(MONTHS) + ')\.{0,1}')

wtp = WtP('wtp-bert-mini')
if cuda.is_available():
    wtp.to('cuda')

def read_ocr_lines(file: str) -> list[str]:
    with open(file, 'r', encoding='utf-8') as infile:
        return infile.read().splitlines()

def correct_ocr_line_by_line(model: TransformerModel, file: list) -> Generator[str, None, None] :
    for line in model.translate(file):
        yield line

# TODO: Skoða þetta. Eitthvað skrítið.
def clean_token(token: str) -> str:
    token_out = token.strip(extended_punctuation.replace(HYPHENEND_TOKEN, ''))
    if token.endswith(HYPHENEND_TOKEN) and not token_out.endswith(HYPHENEND_TOKEN):
        token_out += '-'
    if token.startswith('--'):
        token_out = token_out[1:]
    return token_out

def merge_sentences(lines):
    """

    """
    sentences = []
    current_sentence = []
    for line in lines:
        line = line.strip()
        if line.endswith('.') and current_sentence and not line_ends_with_cardinal_or_abbreviation(line):
            current_sentence.append(line)
            sentences.append(current_sentence)
            current_sentence = []
        elif current_sentence:
            current_sentence.append(line)
        else:
            current_sentence = [line]
    if current_sentence:
        sentences.append(current_sentence)

    return sentences

# def merge_sentences(lines):
#     text = ' '.join(lines)  # Merge lines into a single string
#     sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z\u00C0-\u00FF])', text)
#     for sentence in sentences:
#         yield [sentence.strip()]

# TODO: Skoða þetta. Eitthvað skrítið.
def merge_words(sents: list[str]) -> Generator[str, None, None]:
    sents = [w+'HYPHENEND' if w.endswith('-') else w for w in sents]
    sents = merge_sentences(sents)
    # print(list(sents))
    for sent in sents:
        sent_out = ''
        # [(0, 'Leigjandinn: Eruð þér frá vitinu, herra'), (1, 'minn, jeg að flytja undir eins, áður en jeg hefi')
        enumerated_sent_parts = enumerate(sent)
        for sent_part_index, sent_part in enumerated_sent_parts:
            split_sent_part = sent_part.split()
            len_sent_part = len(split_sent_part)
            # [(0, 'Leigjandinn:'), (1, 'Eruð'), (2, 'þér'), (3, 'frá'), (4, 'vitinu,'), (5, 'herra')]
            enumerated_tokens = enumerate(split_sent_part)
            # Iterate over all tokens in split sent
            for token_index, token in enumerated_tokens:
                # If the token is not the last token in the sentence part,
                # we add it to the sentence with a space after it.
                if not token_index == len_sent_part-1:
                    sent_out += f'{token} '
                # If the token is the last token in the sentence part,
                # we check if it's a known word in the lexicon, if it ends with a hyphen, or if it's a number.
                else:
                    # We decide it exists if none of the above is true.
                    last_token_exists = exists_in_bin_or_old_words(clean_token(token)) and not token.endswith(HYPHENEND_TOKEN) or clean_token(token).isdigit()
                    # If it does not exist.
                    if not last_token_exists:
                        try:
                            # If the token ends with a hyphen
                            if token.endswith(HYPHENEND_TOKEN):
                                # We strip the hyphen-part from it
                                stripped_token = token[:-len(HYPHENEND_TOKEN)]
                            else:
                                stripped_token = token
                            out_cand = stripped_token + sent[sent_part_index+1].split()[0]
                            if exists_in_bin_or_old_words(clean_token(out_cand)) or token.endswith(HYPHENEND_TOKEN):
                                sent_out += f'{out_cand} '
                                sent[sent_part_index+1] = ' '.join(sent[sent_part_index+1].split()[1:])
                            else:
                                sent_out += f'{token} '
                        except IndexError:
                            sent_out += f'{token} '
                            continue                            
                    else:
                        sent_out += f'{token} '
        # print(sent_out)
        yield sent_out.replace('HYPHENEND', '')

# A function that combines elements of a list with the previous element if the previous element ends with a colon or semicolon.
def merge_on_colon_and_semicolon(sentences):
    whole_sentences = []
    current_sentence_being_merged = ''
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence.endswith(':') or sentence.endswith(';'):
            current_sentence_being_merged += f' {sentence} '
        elif current_sentence_being_merged:
            current_sentence_being_merged += f' {sentence} '
            whole_sentences.append(current_sentence_being_merged)
            current_sentence_being_merged = ''
        else:
            whole_sentences.append(sentence)
    if current_sentence_being_merged:
        whole_sentences.append(current_sentence_being_merged)
    return whole_sentences
    


def merge_and_format(text):
    sentence_string = ' '.join([sent for sent in merge_words(text)])
    sentences = wtp.split(sentence_string, lang_code='is', style='ud')
    sentences = merge_on_colon_and_semicolon(sentences)
    for i in sentences:
        yield correct_spaces(i)

def gen_overlapping_ngrams(chars: str, ngr: str) -> list:
    return [chars[i:i + ngr] for i in range(0, len(chars))]

def set_token_case(token, case):
    if case == 'lower':
        return token.lower()
    elif case == 'upper':
        return token.upper()
    elif case == 'title':
        return token.title()
    else:
        return token

def get_token_case(token):
    if token.islower():
        return 'lower'
    elif token.isupper():
        return 'upper'
    elif token.istitle():
        return 'title'
    else:
        return 'mixed'

def line_ends_with_cardinal_or_abbreviation(line):
    return not LINE_END_REGEX_PATTERN.search(line) and not MONTHS_DATE_REGEX.search(line)

def get_differences(original_token, edited_token):
    substitution_sequence = []
    original_token = original_token.lower()
    edited_token = edited_token.lower()
    sm = SequenceMatcher(None, original_token, edited_token).get_opcodes()
    for tag, i1, i2, j1, j2 in sm:
            if tag == 'replace':
                try:
                    original = original_token[i1:i2]
                    sub = edited_token[j1:j2]
                    change = original, sub
                    substitution_sequence.append(change)
                except IndexError:
                    pass
            elif tag == 'insert':
                try:
                    original = original_token[i1:i2]
                    sub = edited_token[j1:j2]
                    change = original, sub
                    substitution_sequence.append(change)
                except IndexError:
                    pass
    return substitution_sequence

if __name__ == '__main__':
    pass
