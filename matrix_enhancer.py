import numpy as np
from sklearn.decomposition import TruncatedSVD

import sys
sys.path.insert(0, '../common/')
import common
import multi_processing


class MatrixEnhancer:
    def __init__(self, matrix, tokens):
        self.matrix = matrix
        self.tokens = tokens

    @classmethod
    def from_storage(cls, matrix_path, tokens_path):
        matrix = np.load(matrix_path)
        tokens = common.read_pickle(tokens_path)
        return cls(matrix, tokens)

    @classmethod
    def from_encoded_edges_count_file_path(cls, encoded_edges_count_file_path, valid_wordIds_path, merged_dict_path,
                                           output_folder):
        def read_valid_vocabulary(file_path):
            result = []
            with open(file_path) as f:
                for line in f:
                    line_element = line.rstrip('\n')
                    result.append(line_element)
            return result

        def get_index2word(file, key_type=int, value_type=str):
            d = {}
            with open(file, encoding='utf-8') as f:
                for line in f:
                    (key, val) = line.rstrip('\n').split("\t")
                    d[key_type(val)] = value_type(key)
                return d

        # Code borrowed from the __init__ function of NoGraph in graph_builder.py
        name_prefix = multi_processing.get_file_name(encoded_edges_count_file_path).split('.')[0]
        valid_wordId = list(set(read_valid_vocabulary(valid_wordIds_path)))  # make sure no duplication
        # ATTENTION: graph_index2wordId should be a list of which the index order is from 0 to vocab_size-1
        graph_index2wordId = list(map(int, valid_wordId))
        vocab_size = len(valid_wordId)
        # ATTENTION: the index is of the type int, while the wordId is of the type str
        graph_wordId2index = dict(zip(valid_wordId, range(vocab_size)))
        # initialize numpy 2d array
        cooccurrence_matrix = np.zeros((vocab_size, vocab_size))
        # read encoded_edges_count_file
        for line in common.read_file_line_yielder(encoded_edges_count_file_path):
            # ATTENTION: line e.g. '17'  '57'  '10' or '57'   '17'  '10' (only one of them will appear in the file.)
            (source, target, weight) = line.split("\t")
            cooccurrence_matrix[graph_wordId2index[source]][graph_wordId2index[target]] = weight
            # undirected graph
            cooccurrence_matrix[graph_wordId2index[target]][graph_wordId2index[source]] = weight

        wordId2word = get_index2word(file=merged_dict_path)
        tokens = [wordId2word[wordId] for wordId in graph_index2wordId]

        # save into output folder
        np.save(output_folder + name_prefix + '_matrix.npy', cooccurrence_matrix, fix_imports=False)
        common.write_to_pickle(tokens, output_folder + name_prefix + '_tokens.pickle')

        return cls(cooccurrence_matrix, tokens)

    def raw2ppmi(self, k_shift=1.0):
        """
        Function from https://github.com/piskvorky/word_embeddings/blob/master/run_embed.py
        Convert raw counts from `get_coccur` into positive PMI values (as per Levy & Goldberg),
        in place.
        The result is an efficient stream of sparse word vectors (=no extra data copy).
        """

        cooccur = np.copy(self.matrix)
        # following lines a bit tedious, as we try to avoid making temporary copies of the (large) `cooccur` matrix
        marginal_word = cooccur.sum(axis=1)
        marginal_context = cooccur.sum(axis=0)
        cooccur /= marginal_word[:, None]  # #(w, c) / #w
        cooccur /= marginal_context  # #(w, c) / (#w * #c)
        cooccur *= marginal_word.sum()  # #(w, c) * D / (#w * #c)
        np.log(cooccur, out=cooccur)  # PMI = log(#(w, c) * D / (#w * #c))

        cooccur -= np.log(k_shift)  # shifted PMI = log(#(w, c) * D / (#w * #c)) - log(k)

        # clipping PMI scores to be non-negative PPMI
        cooccur.clip(0.0, out=cooccur)  # SPPMI = max(0, log(#(w, c) * D / (#w * #c)) - log(k))

        # # normalizing PPMI word vectors to unit length
        # for i, vec in enumerate(cooccur):
        #     cooccur[i] = matutils.unitvec(vec)

        # return matutils.Dense2Corpus(cooccur, documents_columns=False)
        return cooccur

    @staticmethod
    def truncated_svd(matrix, dimension):
        svd = TruncatedSVD(n_components=dimension)
        svd.fit(matrix)
        result = svd.transform(matrix)
        return result

    @staticmethod
    def save_enhanced_matrix(matrix, output_path):
        np.save(output_path, matrix, fix_imports=False)


if __name__ == '__main__':
    pass
    # MatrixEnhancer.from_encoded_edges_count_file_path(encoded_edges_count_file_path='/Users/zzcoolj/Desktop/GoW_new_ideas/input/cooccurrence matrix/encoded_edges_count_window_size_5_undirected.txt',
    #                                                   valid_wordIds_path='/Users/zzcoolj/Desktop/GoW_new_ideas/input/cooccurrence matrix/valid_vocabulary_min_count_5_vocab_size_10000.txt',
    #                                                   merged_dict_path='/Users/zzcoolj/Desktop/GoW_new_ideas/input/dict_merged.txt',
    #                                                   output_folder='output/')

    # m = MatrixEnhancer.from_storage(matrix_path='output/encoded_edges_count_window_size_5_undirected_matrix.npy',
    #                                 tokens_path='output/encoded_edges_count_window_size_5_undirected_tokens.pickle')
    # result = MatrixEnhancer.truncated_svd(m.raw2ppmi(), 1000)
    # MatrixEnhancer.save_enhanced_matrix(result, 'test.npy')

