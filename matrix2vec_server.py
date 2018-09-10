import matrix_enhancer as me

# Generate cooccurrence_matrix and tokens of different window sizes from gow/output/intermediate data/graph/ to
# matrix2vec/input/
gow_folder_path = '~/Code/gow/output/intermediate data/'
for i in (2, 11):
    encoded_edges_count_file_path = gow_folder_path + 'graph/encoded_edges_count_window_size_' \
                                    + str(i) + '_undirected.txt'
    me.MatrixEnhancer.from_encoded_edges_count_file_path(encoded_edges_count_file_path=encoded_edges_count_file_path,
                                                         valid_wordIds_path=gow_folder_path + 'dicts_and_encoded_texts/valid_vocabulary_min_count_5_vocab_size_10000.txt',
                                                         merged_dict_path=gow_folder_path + 'dicts_and_encoded_texts/dict_merged.txt',
                                                         output_folder='input/')
