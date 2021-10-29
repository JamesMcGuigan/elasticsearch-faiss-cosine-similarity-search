#!/usr/bin/env python3

import glob
import os
import time

import numpy as np
from humanize import intcomma, naturalsize, precisedelta

from config import config
from src.segmentation.newline_segmenter import NewlineSegmenter
from src.sentence_encoding.laser_sentence_encoder import LaserSentenceEncoder
from src.utils.fasttest_model import language_detect
from src.utils.files import readfile, savefile
from src.utils.language_codes import get_language_name

dataset_names = [
    'europarl'
]
tokenizers = {
    # "punkt":   PunktSegmenter,
    "newline": NewlineSegmenter,
}
encoders = {
    "laser": LaserSentenceEncoder,
    # "labse": LabseSentenceEncoder,
}
if __name__ == '__main__':
    for dataset_name in dataset_names:
        glob_path  = config['datasets'][dataset_name]['input_glob']
        input_dir  = config['datasets'][dataset_name]['input_dir']
        output_dir = config['datasets'][dataset_name]['output_dir']
        filenames  = glob.glob(glob_path)
        for filename in filenames:
            for encoder_name, encoder_class in encoders.items():
                for tokenizer_name, tokenizer_class in tokenizers.items():
                    try:
                        output_file_txt   = filename.replace(input_dir, output_dir) + f'.{tokenizer_name}.txt'
                        output_file_embed = filename.replace(input_dir, output_dir) + f'.{tokenizer_name}.{encoder_name}.npy'
                        os.makedirs( os.path.dirname(output_file_embed), exist_ok=True )
                        if not os.path.exists(output_file_embed):
                            time_start = time.perf_counter()
                            text       = readfile(filename)
                            lang_code  = language_detect(text)
                            language   = get_language_name(lang_code)
                            sentences  = tokenizer_class().segment_document(text)
                            if not os.path.exists(output_file_txt):
                                savefile("\n".join(sentences), output_file_txt)
                                time_taken = time.perf_counter() - time_start
                                print(f'\nwrote: {output_file_txt} ({language}) =', naturalsize(os.path.getsize(output_file_txt)),
                                      f'\n{intcomma(len(sentences))} lines in {precisedelta(time_taken)}')

                            time_start = time.perf_counter()
                            embeddings = encoder_class().encode_sentences_cached(sentences)
                            np.save(output_file_embed, embeddings)
                            time_taken = time.perf_counter() - time_start

                            print(f'\nwrote: {output_file_embed} ({language}) =', naturalsize(os.path.getsize(output_file_embed)),
                                  f'\n{intcomma(len(embeddings))} embeddings in {precisedelta(time_taken)}')
                    except Exception as exception:
                        print('EXCEPTION:', exception)
