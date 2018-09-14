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
Get matrix from input/ and generate ppmi/ intermediate data and vectors of ppmi_svd
"""

# for i in range(2, 11):
#     m = me.MatrixEnhancer.from_storage(
#         matrix_path='input/encoded_edges_count_window_size_' + str(i) + '_undirected_matrix.npy',
#         tokens_path='input/encoded_edges_count_window_size_' + str(i) + '_undirected_tokens.pickle')
#     ppmi = m.raw2ppmi()
#     me.MatrixEnhancer.save_enhanced_matrix(ppmi, 'output/intermediate_data/ppmi/ppmi_w'+str(i)+'.npy')
#     for dimension in [200, 500, 800, 1000]:
#         vectors = me.MatrixEnhancer.truncated_svd(ppmi, dimension)
#         me.MatrixEnhancer.save_enhanced_matrix(vectors, 'output/vectors/ppmi_svd/' + 'ppmi_svd_w' + str(i) +
#                                                '_d' + str(dimension) + '.npy')

'''
Get matrix from input/ and generate first_order/ intermediate data and vectors of firstOrder_svd
'''

# for i in range(2, 11):
#     m = me.MatrixEnhancer.from_storage(
#         matrix_path='input/encoded_edges_count_window_size_' + str(i) + '_undirected_matrix.npy',
#         tokens_path='input/encoded_edges_count_window_size_' + str(i) + '_undirected_tokens.pickle')
#     firstOrder = m.raw2firstOrder()
#     me.MatrixEnhancer.save_enhanced_matrix(firstOrder, 'output/intermediate_data/first_order/firstOrder_w'+str(i)+'.npy')
#     for dimension in [200, 500, 800, 1000]:
#         vectors = me.MatrixEnhancer.truncated_svd(firstOrder, dimension)
#         me.MatrixEnhancer.save_enhanced_matrix(vectors, 'output/vectors/firstOrder_svd/' + 'firstOrder_svd_w' + str(i) +
#                                                '_d' + str(dimension) + '.npy')

"""
Get matrix from input/ and generate rw_0/ rw_1/ rw_2/ intermediate data and vectors of rw0_svd/ rw1_svd/ rw2_svd/
"""

# for window_size in range(2, 11):
#     m = me.MatrixEnhancer.from_storage(
#         matrix_path='input/encoded_edges_count_window_size_' + str(window_size) + '_undirected_matrix.npy',
#         tokens_path='input/encoded_edges_count_window_size_' + str(window_size) + '_undirected_tokens.pickle')
#     for matrix, step in m.zero_to_t_step_random_walk_stochastic_matrix_yielder(t=2):
#         me.MatrixEnhancer.save_enhanced_matrix(matrix, 'output/intermediate_data/rw_' + str(step) + '/rw' + str(step) +
#                                                '_w' + str(window_size) + '.npy')
#         for dimension in [200, 500, 800, 1000]:
#             vectors = me.MatrixEnhancer.truncated_svd(matrix, dimension)
#             me.MatrixEnhancer.save_enhanced_matrix(vectors,
#                                                    'output/vectors/rw' + str(step) + '_svd/' + 'rw' + str(step) +
#                                                    '_svd_w' + str(window_size) + '_d' + str(dimension) + '.npy')

'''
Get matrix from ppmi/ and first_order/ and generate ppmi+firstOrder/ intermediate data and vectors of ppmi+firstOrder_svd
'''

m = me.MatrixMixer.from_storage(base_matrix_path='output/intermediate_data/ppmi/ppmi_w5.npy',
                                ingredient_matrix_path='output/intermediate_data/first_order/firstOrder_w5.npy')
for k, mixed_matrix in m.grid_search_k_yielder(ks=[-0.1, -0.2, -0.5, -1, -2, -5, -10, -20, -50, -100],
                                               output_folder='output/intermediate_data/ppmi+firstOrder/'):
    for dimension in [500]:
        vectors = me.MatrixDimensionReducer.truncated_svd(mixed_matrix, dimension)
        me.MatrixDimensionReducer.save_enhanced_matrix(vectors,
                                                       'output/vectors/ppmi+firstOrder_svd/' +
                                                       'ppmi_w5_+firstOrder_w5_k'+str(k)+'_svd_d'+str(dimension)+'.npy')
