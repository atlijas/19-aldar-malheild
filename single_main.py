import os
import shutil
from glob import glob
from pathlib import Path
from argparse import ArgumentParser
from fairseq.models.transformer import TransformerModel
from utils.utility_functions import correct_ocr_line_by_line, merge_sentences, merge_and_format, read_ocr_lines
from utils.modernize import modernize_sentence


parser = ArgumentParser()
parser.add_argument('--file')
parser.add_argument('--transform-only', action='store_true')
parser.add_argument('--modernize-only', action='store_true')
args = parser.parse_args()


OCR_POST_PROCESSING_MODEL = TransformerModel.from_pretrained('frsq/models/',
                                            checkpoint_file='checkpoint_best.pt',
                                            data_name_or_path='frsq/data/data-bin.3000',
                                            bpe='sentencepiece',
                                            sentencepiece_model='frsq/data/sentencepiece/data/sentencepiece_3000.bpe.model')


OCR_POST_PROCESSING_MODEL.cuda(0)


read_lines = list(read_ocr_lines(args.file))

if args.transform_only:
    transformed_lines = list(correct_ocr_line_by_line(OCR_POST_PROCESSING_MODEL, read_lines))
    # merged_transformed = list(merge_and_format(list(transformed_lines)))
    for i in transformed_lines:
        if len(i) > 0:
            print(i)
    exit()

if args.modernize_only:
    merged_modernized = [modernize_sentence(line) for line in read_lines]
    for i in merged_modernized:
        if len(i) > 0:
            print(i)
    exit()


transformed_lines = list(correct_ocr_line_by_line(OCR_POST_PROCESSING_MODEL, read_lines))
merged_transformed = list(merge_and_format(list(transformed_lines)))
print(merged_transformed)
merged_modernized = [modernize_sentence(line) for line in merged_transformed]
print(merged_modernized)
# for i in merged_modernized:
#     if len(i) > 0:
#         print(i)
