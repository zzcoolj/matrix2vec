import matrix_enhancer as me
import numpy as np
from nltk.corpus import stopwords
import string

import sys
sys.path.insert(0, '../common/')
import common
import multi_processing
sys.path.insert(0, '../word_embeddings_evaluator/')
import evaluator_server

'''
Generate cooccurrence_matrix and tokens of different window sizes from gow/output/intermediate data/graph/ to
matrix2vec/input/
'''

# gow_folder_path = '../gow/output/intermediate data/'
# for i in range(2, 11):
#     encoded_edges_count_file_path = gow_folder_path + 'graph/encoded_edges_count_window_size_' \
#                                     + str(i) + '_undirected.txt'
#     me.MatrixEnhancer.from_encoded_edges_count_file_path(encoded_edges_count_file_path=encoded_edges_count_file_path,
#                                                          valid_wordIds_path=gow_folder_path + 'dicts_and_encoded_texts/valid_vocabulary_min_count_5_vocab_size_10000.txt',
#                                                          merged_dict_path=gow_folder_path + 'dicts_and_encoded_texts/dict_merged.txt',
#                                                          output_folder='input/')

"""
Get matrix from input/ and generate rw0/ rw1/ rw2/ intermediate and firstOrder/ intermediate data
"""

# for window_size in range(2, 11):
#     m = me.MatrixEnhancer.from_storage(
#         matrix_path='input/encoded_edges_count_window_size_' + str(window_size) + '_undirected_matrix.npy',
#         tokens_path='input/encoded_edges_count_window_size_' + str(window_size) + '_undirected_tokens.pickle')
#     for matrix, step in m.zero_to_t_step_random_walk_stochastic_matrix_yielder(t=2):
#         me.save_enhanced_matrix(matrix, 'output/intermediate_data/rw' + str(step) + '/rw' + str(step) +
#                                 '_w' + str(window_size) + '.npy')
#
# for i in range(2, 11):
#     m = me.MatrixEnhancer.from_storage(
#         matrix_path='input/encoded_edges_count_window_size_' + str(i) + '_undirected_matrix.npy',
#         tokens_path='input/encoded_edges_count_window_size_' + str(i) + '_undirected_tokens.pickle')
#     firstOrder = m.raw2firstOrder(no_influence_of_stop_words_and_punctuation=True)
#     me.save_enhanced_matrix(firstOrder, 'output/intermediate_data/firstOrder_noStopWords/firstOrder_noStopWords_w'+str(i)+'.npy')

'''
From first_order/ intermediate data to firstOrder_normalized_svd/ (or firstOrder_normalized_smoothed_svd/)
ppmi_svd/
rw012_normalized_svd/
'''

# for i in range(2, 11):
#     mn = me.MatrixNormalization.from_storage('output/intermediate_data/firstOrder/firstOrder_w'+str(i)+'.npy')
#     matrix = mn.pmi_without_log()
#     matrix = me.MatrixSmoothing(matrix).log_shifted_positive(k_shift=None)
#
#     # # for firstOrder_noStopWords matrix, after ppmi, there are infinity values rows/columns. Change them to 0s.
#     # tokens = common.read_pickle('input/encoded_edges_count_window_size_' + str(i) + '_undirected_tokens.pickle')
#     # stop = stopwords.words('english') + list(string.punctuation)
#     # stop_indices = []
#     # for index in range(len(tokens)):
#     #     if tokens[index] in stop:
#     #         stop_indices.append(index)
#     # matrix[:, stop_indices] = 0  # replace corresponding columns to 0
#     # matrix[stop_indices, :] = 0  # replace corresponding rows to 0
#
#     # me.save_enhanced_matrix(matrix, 'output/intermediate_data/firstOrder_ppmied/firstOrder_ppmied_w'+ str(i) +'.npy')
#
#     for dimension in [500, 700, 1000]:
#         vectors = me.MatrixDimensionReducer.truncated_svd(matrix, dimension)
#         me.save_enhanced_matrix(vectors, 'output/vectors/firstOrder_svd/' + 'firstOrder_svd_w' +
#                                 str(i) + '_d' + str(dimension) + '.npy')

# for i in range(2, 11):
#     matrix = np.load('output/intermediate_data/firstOrder_noStopWords/firstOrder_noStopWords_w'+str(i)+'.npy')
#     for dimension in [500, 700, 1000]:
#         vectors = me.MatrixDimensionReducer.truncated_svd(matrix, dimension)
#         me.save_enhanced_matrix(vectors, 'output/vectors/firstOrder_noStopWords_svd/' + 'firstOrder_noStopWords_svd_w' +
#                                 str(i) + '_d' + str(dimension) + '.npy')

# for i in range(2, 11):
#     matrix = np.load('output/intermediate_data/ppmi/' + 'ppmi_w' + str(i) + '.npy')
#     for dimension in [500, 700, 1000]:
#         vectors = me.MatrixDimensionReducer.truncated_svd(matrix, dimension)
#         me.save_enhanced_matrix(vectors, 'output/vectors/ppmi_svd/' + 'ppmi_svd_w' + str(i) + '_d' + str(dimension) + '.npy')

# for rw in [0, 1, 2]:
#     for i in range(2, 11):
#         mn = me.MatrixNormalization.from_storage('output/intermediate_data/rw'+str(rw)+'/rw'+str(rw)+'_w'+str(i)+'.npy')
#         normalized_matrix = mn.pmi_without_log()
#         for dimension in [200, 500, 800, 1000]:
#             vectors = me.MatrixDimensionReducer.truncated_svd(normalized_matrix, dimension)
#             me.save_enhanced_matrix(vectors, 'output/vectors/rw' + str(rw) + '_normalized_svd/' + 'rw' + str(rw) +
#                                     '_normalized_svd_w' + str(i) + '_d' + str(dimension) + '.npy')

"""
Get matrix from ppmi/ and first_order/ and generate ppmi+firstOrder/ intermediate data and vectors of ppmi+firstOrder_svd/
Get matrix from ppmi/ and rw_1/ and generate ppmi+rw1/ intermediate data and vectors of ppmi+rw1_svd/
Get matrix from ppmi/ and rw_2/ and generate ppmi+rw2/ intermediate data and vectors of ppmi+rw2_svd/
"""

# m = me.MatrixMixer.from_storage(base_matrix_path='output/intermediate_data/ppmi/ppmi_w5.npy',
#                                 ingredient_matrix_path='output/intermediate_data/rw_2/rw2_w3.npy')
# for k, mixed_matrix in m.grid_search_k_yielder(ks=[0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, -0.1, -0.2, -0.5, -1, -2, -5, -10, -20, -50, -100],
#                                                output_path_prefix='output/intermediate_data/ppmi+rw2/ppmi_w5_rw2_w3_k'):
#     for dimension in [500]:
#         vectors = me.MatrixDimensionReducer.truncated_svd(mixed_matrix, dimension)
#         me.MatrixDimensionReducer.save_enhanced_matrix(vectors,
#                                                        'output/vectors/ppmi+rw2_svd/' +
#                                                        'ppmi_w5_+rw2_w3_k'+str(k)+'_svd_d'+str(dimension)+'.npy')

'''
From matrix from input/ and firstOrder/ to cooc_firstOrder_normalized_svd/
'''

# window_size = 5
# base_matrix = me.MatrixNormalization.from_storage('input/encoded_edges_count_window_size_' + str(window_size) + '_undirected_matrix.npy')
# normalized_base_matrix = base_matrix.pmi_without_log()
# ingredient_matrix = me.MatrixNormalization.from_storage('output/intermediate_data/firstOrder/firstOrder_w'+str(window_size)+'.npy')
# normalized_ingredient_matrix = ingredient_matrix.pmi_without_log()
# m = me.MatrixMixer(base_matrix=normalized_base_matrix, ingredient_matrix=normalized_ingredient_matrix,
#                    base_window_size=window_size, ingredient_window_size=window_size)
# for k, mixed_matrix in m.grid_search_k_yielder(ks=[0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, -0.1, -0.2, -0.5, -1, -2, -5, -10, -20, -50, -100],
#                                                output_path_prefix=''):
#     for dimension in [500, 800, 1000]:
#         vectors = me.MatrixDimensionReducer.truncated_svd(mixed_matrix, dimension)
#         me.save_enhanced_matrix(vectors, 'output/vectors/cooc_firstOrder_normalized_svd/' + 'cooc_w5_firstOrder_w5_normalized_k'+str(k)+'_svd_d'+str(dimension)+'.npy')

'''
Matrix concatenate idea1
'''

# # matrix1 = np.load('output/vectors/cooc_normalized_smoothed_svd/cooc_normalized_smoothed_svd_w5_d500.npy')
# # matrix2 = np.load('output/vectors/firstOrder_normalized_svd/firstOrder_normalized_svd_w5_d500.npy')
# # matrix_all = np.concatenate((matrix1, matrix2), axis=1)
# # me.save_enhanced_matrix(matrix_all, 'output/vectors/specific/test.npy')
#
# matrix_all = np.load('output/vectors/specific/test.npy')
# m2 = me.MatrixDimensionReducer.truncated_svd(matrix_all, 300)
# me.save_enhanced_matrix(m2, 'output/vectors/specific/m2_300.npy')
# m2 = me.MatrixDimensionReducer.truncated_svd(matrix_all, 500)
# me.save_enhanced_matrix(m2, 'output/vectors/specific/m2_500.npy')
# m2 = me.MatrixDimensionReducer.truncated_svd(matrix_all, 700)
# me.save_enhanced_matrix(m2, 'output/vectors/specific/m2_700.npy')
# m2 = me.MatrixDimensionReducer.truncated_svd(matrix_all, 900)
# me.save_enhanced_matrix(m2, 'output/vectors/specific/m2_900.npy')
#
# # matrix1 = np.load('output/vectors/cooc_normalized_smoothed_svd/cooc_normalized_smoothed_svd_w5_d500.npy')
# # matrix2 = np.load('output/vectors/firstOrder_normalized_smoothed_svd/firstOrder_normalized_smoothed_svd_w5_d500.npy')
# # m3 = np.concatenate((matrix1, matrix2), axis=1)
# # me.save_enhanced_matrix(m3, 'output/vectors/specific/m3.npy')
#
# m3 = np.load('output/vectors/specific/m3.npy')
# m4 = me.MatrixDimensionReducer.truncated_svd(m3, 300)
# me.save_enhanced_matrix(m4, 'output/vectors/specific/m4_300.npy')
# m4 = me.MatrixDimensionReducer.truncated_svd(m3, 500)
# me.save_enhanced_matrix(m4, 'output/vectors/specific/m4_500.npy')
# m4 = me.MatrixDimensionReducer.truncated_svd(m3, 700)
# me.save_enhanced_matrix(m4, 'output/vectors/specific/m4_700.npy')
# m4 = me.MatrixDimensionReducer.truncated_svd(m3, 900)
# me.save_enhanced_matrix(m4, 'output/vectors/specific/m4_900.npy')

"""
Matrix concatenate idea2
"""

# window_size = 5
#
# m1 = me.MatrixNormalization.from_storage('output/intermediate_data/firstOrder/firstOrder_w'+str(window_size)+'.npy')
# m2 = me.MatrixNormalization.from_storage('input/encoded_edges_count_window_size_' + str(window_size) + '_undirected_matrix.npy')
# # m1 = m1.line_normalization()
# # m1 = m1.pmi_without_log()
# m1 = me.MatrixSmoothing(m1.matrix).log_shifted_positive(k_shift=None)
# m1 = me.MatrixNormalization(m1).line_normalization()
# # m2 = m2.line_normalization()
# # m2 = m2.pmi_without_log()
# m2 = me.MatrixSmoothing(m2.matrix).log_shifted_positive(k_shift=None)
# m2 = me.MatrixNormalization(m2).line_normalization()
# m3 = np.concatenate((m1, m2), axis=1)
# m4 = me.MatrixDimensionReducer.truncated_svd(m3, 300)
# me.save_enhanced_matrix(m4, 'output/vectors/specific/m4_300.npy')
# m4 = me.MatrixDimensionReducer.truncated_svd(m3, 500)
# me.save_enhanced_matrix(m4, 'output/vectors/specific/m4_500.npy')
# m4 = me.MatrixDimensionReducer.truncated_svd(m3, 700)
# me.save_enhanced_matrix(m4, 'output/vectors/specific/m4_700.npy')
# m4 = me.MatrixDimensionReducer.truncated_svd(m3, 900)
# me.save_enhanced_matrix(m4, 'output/vectors/specific/m4_900.npy')
# m4 = me.MatrixDimensionReducer.truncated_svd(m3, 1000)
# me.save_enhanced_matrix(m4, 'output/vectors/specific/m4_1000.npy')


def super_concatenate(folder_a_path, folder_b_path):
    """

    :param folder_a_path: ends with '/'
    :param folder_b_path: ends with '/'
    :return:
    """
    max_window_size = 10
    dimensions = [500, 700, 1000]
    for i in range(2, max_window_size+1):
        folder_a_name = folder_a_path.split('/', -1)[-2]
        folder_b_name = folder_b_path.split('/', -1)[-2]
        matrix_a = np.load(folder_a_path + folder_a_name + '_w' + str(i) + '.npy')
        tokens_a = common.read_pickle('input/encoded_edges_count_window_size_' + str(i) + '_undirected_tokens.pickle')
        for j in range(2, max_window_size + 1):
            matrix_b = np.load(folder_b_path + folder_b_name + '_w' + str(j) + '.npy')
            tokens_b = common.read_pickle(
                'input/encoded_edges_count_window_size_' + str(j) + '_undirected_tokens.pickle')
            if i == j:
                matrix_b_reordered = matrix_b
            else:
                matrix_b_reordered = me.MatrixMixer._reorder_matrix(ingredient_matrix=matrix_b,
                                                                    ingredient_tokens=tokens_b, base_tokens=tokens_a)
            matrix_mix = np.concatenate((matrix_a, matrix_b_reordered), axis=1)
            for dimension in dimensions:
                vectors = me.MatrixDimensionReducer.truncated_svd(matrix_mix, dimension)
                me.save_enhanced_matrix(vectors,
                                        'output/vectors/to_delete/' + str(folder_a_name) + '_w' + str(i) + '_' + str(
                                            folder_b_name) + '_w' + str(j) + '_d' + str(dimension) + '.npy')

        # evaluate all combinations based on specific i
        evaluator_server.evaluate_folder_superConcatenate(files_prefix='output/vectors/to_delete/' + str(folder_a_name) + '_w' + str(i) + '_' + str(folder_b_name) + '_w',
                                                          base_window_size=i,
                                                          dimensions=dimensions,
                                                          window_size_max=max_window_size,
                                                          output_folder_path='output/vectors/to_delete/')
