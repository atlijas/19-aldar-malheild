import os
import shutil
from glob import glob
from pathlib import Path
from fairseq.models.transformer import TransformerModel
from utils.utility_functions import correct_ocr_line_by_line, merge_sentences, merge_and_format, read_ocr_lines
from utils.modernize import modernize_sentence

ALL_FILES = glob('all_txt/**/*txt', recursive=True)
FILE_LEN = len(ALL_FILES)

OCR_POST_PROCESSING_MODEL = TransformerModel.from_pretrained('frsq/models/',
                                            checkpoint_file='checkpoint_best.pt',
                                            data_name_or_path='frsq/data/data-bin.3000',
                                            bpe='sentencepiece',
                                            sentencepiece_model='frsq/data/sentencepiece/data/sentencepiece_3000.bpe.model')



OCR_POST_PROCESSING_MODEL.cuda(0)


counter = 0


for file in ALL_FILES[:100]:
    try:
        # Print this to see how much has been processed
        counter += 1
        print(f'[{counter}/{FILE_LEN}]')
        # Path object of the original OCRed file
        ocred_file = Path(file)
        # Name of the original OCRed file. The output files will have the same name.
        ocred_file_name = ocred_file.name

        # Path object of the directory the original OCRed file is in. 
        ocred_file_parent_dir = ocred_file.parent
        # Name of the directory the original OCRed file is in. The output files will
        # be in a directory of the same name.
        out_file_path_parent = ocred_file_parent_dir.parent.name

        # This json file contains information about the OCRed file,
        # such as publication date and type, synopsis, and description.
        json_info_file = Path(f'{ocred_file_parent_dir}/.issue.json')

        # Encode/decode the directory name because of some weird characters in the directory names.
        try:
            decoded_par_dir = ocred_file.parent.name.encode('ibm437').decode('utf-8')
        except UnicodeEncodeError:
            decoded_par_dir = ocred_file.parent.name.encode('ibm866').decode('utf-8')

        transformed_outfile_name = f'output_files/transformed/{out_file_path_parent}/{decoded_par_dir}/{ocred_file_name}'
        modernized_outfile_name = f'output_files/modernized/{out_file_path_parent}/{decoded_par_dir}/{ocred_file_name}'
        original_outfile_name = f'output_files/original/{out_file_path_parent}/{decoded_par_dir}/{ocred_file_name}'


        os.makedirs(os.path.dirname(transformed_outfile_name), exist_ok=True)
        os.makedirs(os.path.dirname(modernized_outfile_name), exist_ok=True)
        os.makedirs(os.path.dirname(original_outfile_name), exist_ok=True)

        transformed_new_json_info_file_path = f'output_files/transformed/{out_file_path_parent}/{decoded_par_dir}/.issue.json'
        modernized_new_json_info_file_path = f'output_files/modernized/{out_file_path_parent}/{decoded_par_dir}/.issue.json'
        original_new_json_info_file_path = f'output_files/original/{out_file_path_parent}/{decoded_par_dir}/.issue.json'


        read_lines = list(read_ocr_lines(file))
        transformed_lines = list(correct_ocr_line_by_line(OCR_POST_PROCESSING_MODEL, read_lines))
        merged_transformed = list(merge_and_format(list(transformed_lines)))
        merged_modernized = [modernize_sentence(line, 
                                check_similar_in_bin=False, 
                                check_parts_in_bin=False, 
                                check_modernized=True, 
                                check_yfirlestur=False, 
                                check_mask=False) 
                            for line in merged_transformed]

        if not Path(transformed_new_json_info_file_path).exists():
            shutil.copy(json_info_file, str(transformed_new_json_info_file_path))
        if not Path(modernized_new_json_info_file_path).exists():
            shutil.copy(json_info_file, str(modernized_new_json_info_file_path))
        if not Path(original_new_json_info_file_path).exists():
            shutil.copy(json_info_file, str(original_new_json_info_file_path))
        
        with open(transformed_outfile_name, 'w') as f:
            f.write('\n'.join(merged_transformed))

        with open(modernized_outfile_name, 'w') as f:
            f.write('\n'.join(merged_modernized))
        with open(original_outfile_name, 'w') as f:
            f.write('\n'.join(read_lines))

    except Exception as e:
        with open('error_log.txt', 'a') as f:
            f.write(f'{file}\n{e}\n\n')
        continue


