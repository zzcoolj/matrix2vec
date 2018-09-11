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

"""
Get matrix from input/ and generate first_order/ intermediate data and vectors of firstOrder_svd
"""

for i in range(2, 11):
    m = me.MatrixEnhancer.from_storage(
        matrix_path='input/encoded_edges_count_window_size_' + str(i) + '_undirected_matrix.npy',
        tokens_path='input/encoded_edges_count_window_size_' + str(i) + '_undirected_tokens.pickle')
    firstOrder = m.raw2firstOrder()
    me.MatrixEnhancer.save_enhanced_matrix(firstOrder, 'output/intermediate_data/first_order/firstOrder_w'+str(i)+'.npy')
    for dimension in [200, 500, 800, 1000]:
        vectors = me.MatrixEnhancer.truncated_svd(firstOrder, dimension)
        me.MatrixEnhancer.save_enhanced_matrix(vectors, 'output/vectors/firstOrder_svd/' + 'firstOrder_svd_w' + str(i) +
                                               '_d' + str(dimension) + '.npy')
