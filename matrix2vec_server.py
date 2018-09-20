import matrix_enhancer as me

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
Get matrix from input/ and generate rw_0/ rw_1/ rw_2/ intermediate and first_order/ intermediate data
"""

# for window_size in range(2, 11):
#     m = me.MatrixEnhancer.from_storage(
#         matrix_path='input/encoded_edges_count_window_size_' + str(window_size) + '_undirected_matrix.npy',
#         tokens_path='input/encoded_edges_count_window_size_' + str(window_size) + '_undirected_tokens.pickle')
#     for matrix, step in m.zero_to_t_step_random_walk_stochastic_matrix_yielder(t=2):
#         me.save_enhanced_matrix(matrix, 'output/intermediate_data/rw' + str(step) + '/rw' + str(step) +
#                                 '_w' + str(window_size) + '.npy')

# for i in range(2, 11):
#     m = me.MatrixEnhancer.from_storage(
#         matrix_path='input/encoded_edges_count_window_size_' + str(i) + '_undirected_matrix.npy',
#         tokens_path='input/encoded_edges_count_window_size_' + str(i) + '_undirected_tokens.pickle')
#     firstOrder = m.raw2firstOrder()
#     me.save_enhanced_matrix(firstOrder, 'output/intermediate_data/first_order/firstOrder_w'+str(i)+'.npy')

'''
From first_order/ intermediate data to firstOrder_normalized_svd/ ,cooc_normalized_svd/ and rw012_normalized_svd/
'''

# for i in range(2, 11):
#     mn = me.MatrixNormalization.from_storage('output/intermediate_data/firstOrder/firstOrder_w'+str(i)+'.npy')
#     normalized_matrix = mn.pmi_without_log()
#     for dimension in [200, 500, 800, 1000]:
#         vectors = me.MatrixDimensionReducer.truncated_svd(normalized_matrix, dimension)
#         me.save_enhanced_matrix(vectors, 'output/vectors/firstOrder_normalized_svd/' + 'firstOrder_normalized_svd_w' +
#                                 str(i) + '_d' + str(dimension) + '.npy')
#
# for i in range(2, 11):
#     mn = me.MatrixNormalization.from_storage('input/encoded_edges_count_window_size_' + str(i) + '_undirected_matrix.npy')
#     normalized_matrix = mn.pmi_without_log()
#     for dimension in [200, 500, 800, 1000]:
#         vectors = me.MatrixDimensionReducer.truncated_svd(normalized_matrix, dimension)
#         me.save_enhanced_matrix(vectors, 'output/vectors/cooc_normalized_svd/' + 'cooc_normalized_svd_w' +
#                                 str(i) + '_d' + str(dimension) + '.npy')

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

"""
Specific case
"""
k = -1
base_matrix = me.MatrixNormalization.from_storage('input/encoded_edges_count_window_size_5_undirected_matrix.npy')
normalized_base_matrix = base_matrix.pmi_without_log()
ingredient_matrix = me.MatrixNormalization.from_storage('output/intermediate_data/firstOrder/firstOrder_w5.npy')
normalized_ingredient_matrix = ingredient_matrix.pmi_without_log()
m = me.MatrixMixer(base_matrix=normalized_base_matrix, ingredient_matrix=normalized_ingredient_matrix,
                   base_window_size=5, ingredient_window_size=5)
mm = m.mix(k)
for i in range(10000):
    for j in range(10000):
        if mm[i][j] < 0:
            print(i)
            print(j)
            print(mm[i][j])
            exit()
mms = me.MatrixSmoothing(mm).log_shifted_positive(k_shift=0)
for dimension in [500, 800, 1000]:
    vectors = me.MatrixDimensionReducer.truncated_svd(mms, dimension)
    me.save_enhanced_matrix(vectors, 'output/vectors/specific/' + 'specific_k' + str(k) +
                            '_svd_d' + str(dimension) + '.npy')