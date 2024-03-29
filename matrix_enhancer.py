import numpy as np
from sklearn.decomposition import TruncatedSVD
import matplotlib
matplotlib.use('agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import string

import sys
sys.path.insert(0, '../common/')
import common
import multi_processing


class MatrixEnhancer(object):
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

    # def raw2ppmi(self, k_shift=None):
    #     """
    #     Function from https://github.com/piskvorky/word_embeddings/blob/master/run_embed.py
    #     Convert raw counts from `get_coccur` into positive PMI values (as per Levy & Goldberg),
    #     in place.
    #     The result is an efficient stream of sparse word vectors (=no extra data copy).
    #     """
    #
    #     cooccur = np.copy(self.matrix)
    #     # following lines a bit tedious, as we try to avoid making temporary copies of the (large) `cooccur` matrix
    #     marginal_word = cooccur.sum(axis=1)
    #     marginal_context = cooccur.sum(axis=0)
    #     cooccur /= marginal_word[:, None]  # #(w, c) / #w
    #     cooccur /= marginal_context  # #(w, c) / (#w * #c)
    #     cooccur *= marginal_word.sum()  # #(w, c) * D / (#w * #c)
    #     np.log(cooccur, out=cooccur)  # PMI = log(#(w, c) * D / (#w * #c))
    #
    #     if k_shift:
    #         cooccur -= np.log(k_shift)  # shifted PMI = log(#(w, c) * D / (#w * #c)) - log(k)
    #
    #     # clipping PMI scores to be non-negative PPMI
    #     cooccur.clip(0.0, out=cooccur)  # SPPMI = max(0, log(#(w, c) * D / (#w * #c)) - log(k))
    #
    #     # # normalizing PPMI word vectors to unit length
    #     # for i, vec in enumerate(cooccur):
    #     #     cooccur[i] = matutils.unitvec(vec)
    #
    #     # return matutils.Dense2Corpus(cooccur, documents_columns=False)
    #     return cooccur

    def _get_stochastic_matrix(self, remove_self_loops=False, change_zeros_to_minimum_positive_value=False):
        """
        Function from graph_builder.py of GNEG
        """
        vocab_size = self.matrix.shape[0]
        stochastic_matrix = self.matrix.copy()

        """ change all zeros values to the minimum positive value
        change in the co-occurrence stage or change after percentage will get same result.
        The only difference is:
             change in the co-occurrence stage then calculate percentage, stochastic matrix is already normalized.
             change after percentage, stochastic matrix need to be normalized again. So the previous one is better.
        """
        if change_zeros_to_minimum_positive_value:
            # find zero position in the matrix
            zero_indices_x, zero_indices_y = np.where(stochastic_matrix == 0)
            zeros_length = len(zero_indices_x)
            if zeros_length == 0:
                print('No zero cells in matrix.')
            else:
                # find the second minimum value in matrix, temp_matrix is used for that
                max_value = np.amax(stochastic_matrix)
                temp_matrix = np.copy(stochastic_matrix)
                for i in range(zeros_length):
                    temp_matrix[zero_indices_x[i]][zero_indices_y[i]] = max_value
                second_minimums = np.amin(temp_matrix, axis=1)  # Minima along the second axis
                # set all zeros to second minimum values
                for i in range(zeros_length):
                    stochastic_matrix[zero_indices_x[i]][zero_indices_y[i]] = second_minimums[zero_indices_x[i]]

        if remove_self_loops:
            for i in range(vocab_size):
                stochastic_matrix[i][i] = 0

        # print(stochastic_matrix)

        # calculate percentage
        matrix_sum_row = np.sum(stochastic_matrix, axis=1, keepdims=True)  # sum of each row and preserve the dimension
        stochastic_matrix /= matrix_sum_row
        return stochastic_matrix

    def zero_to_t_step_random_walk_stochastic_matrix_yielder(self, t, remove_self_loops=False,
                                                             change_zeros_to_minimum_positive_value=False):
        """
        Function from graph_builder.py of GNEG
        Instead of getting a specific t step random walk result, this method gets a dict of result from 1 step random
        walk to t step random walk. This method should be used for grid search.
        """
        transition_matrix = self._get_stochastic_matrix(remove_self_loops=remove_self_loops,
                                                        change_zeros_to_minimum_positive_value=change_zeros_to_minimum_positive_value)
        result = transition_matrix
        for t in range(t+1):
            if t != 0:
                result = np.matmul(result, transition_matrix)
            yield result, t

    def raw2firstOrder(self, no_influence_of_stop_words_and_punctuation):
        if no_influence_of_stop_words_and_punctuation:
            stop = stopwords.words('english') + list(string.punctuation)
            stop_indices = []
            for i in range(len(self.tokens)):
                if self.tokens[i] in stop:
                    stop_indices.append(i)
            m = np.copy(self.matrix)
            m[:, stop_indices] = 0  # replace corresponding columns to 0
            m[stop_indices, :] = 0  # replace corresponding rows to 0
            return np.dot(m, m.T)
        else:
            return np.dot(self.matrix, self.matrix.T)

    def draw_stopWords_and_punctuation(self, output_path):
        stopWords_indices = []
        stops = stopwords.words('english')
        punctuations_indices = []
        punctuations = list(string.punctuation)
        for i in range(len(self.tokens)):
            if self.tokens[i] in stops:
                stopWords_indices.append(i)
            elif self.tokens[i] in punctuations:
                punctuations_indices.append(i)
        m = np.zeros((len(self.tokens), len(self.tokens)))
        # set stop word token rows and columns to 1
        m[:, stopWords_indices] = 1
        m[stopWords_indices, :] = 1
        # set punctuation rows and columns to 2
        m[:, punctuations_indices] = 2
        m[punctuations_indices, :] = 2
        plt.imshow(m, cmap="nipy_spectral")  # plt.cm.BuPu_r, hot -> bad choices (no big difference)
        plt.colorbar()
        plt.savefig(output_path)
        plt.clf()


class MatrixMixer(object):
    def __init__(self, base_matrix, ingredient_matrix, base_window_size, ingredient_window_size):
        self.base_matrix = base_matrix
        self.ingredient_matrix = ingredient_matrix
        self.base_window_size = base_window_size
        self.ingredient_window_size = ingredient_window_size

    @classmethod
    def from_storage(cls, base_matrix_path, ingredient_matrix_path):
        base_matrix = np.load(base_matrix_path)
        ingredient_matrix = np.load(ingredient_matrix_path)
        if base_matrix_path.startswith('input'):
            base_window_size = int(base_matrix_path.split('window_size_')[1].split('_', 1)[0])
        else:
            base_window_size = int(base_matrix_path.rpartition('_w')[2].split('.npy')[0])
        if ingredient_matrix_path.startswith('input'):
            ingredient_window_size = int(ingredient_matrix_path.split('window_size_')[1].split('_', 1)[0])
        else:
            ingredient_window_size = int(ingredient_matrix_path.rpartition('_w')[2].split('.npy')[0])
        return cls(base_matrix, ingredient_matrix, base_window_size, ingredient_window_size)

    def mix(self, k):
        """
        :return: base_matrix + k * ingredient_matrix
        """
        if self.base_window_size == self.ingredient_window_size:
            # base_matrix and ingredient_matrix share the same tokens order <=> use the same tokens <=> same window size
            return np.add(self.base_matrix, k * self.ingredient_matrix)
        else:
            print('reorder matrix')
            ingredient_tokens_path = 'input/encoded_edges_count_window_size_'+str(self.ingredient_window_size)+'_undirected_tokens.pickle'
            base_tokens_path = 'input/encoded_edges_count_window_size_'+str(self.base_window_size)+'_undirected_tokens.pickle'
            ingredient_tokens = common.read_pickle(ingredient_tokens_path)
            base_tokens = common.read_pickle(base_tokens_path)
            return np.add(self.base_matrix, k * self._reorder_matrix(self.ingredient_matrix,
                                                                     ingredient_tokens, base_tokens))

    @staticmethod
    def _reorder_matrix(ingredient_matrix, ingredient_tokens, base_tokens):
        """e.g.
        ingredient_tokens: [windows, apple, ibm, tesla]
        base_tokens: [apple, tesla, ibm, windows] (what I want)
        new_index_order = [1, 3, 2, 0]
        1 means translated_matrix_order index 1 element apple is the first element in translated_reordered_matrix_order
        3 means translated_matrix_order index 3 element tesla is the second element in translated_reordered_matrix_order
        """
        new_index_order = [ingredient_tokens.index(token) for token in base_tokens]
        # reorder rows
        reordered_matrix = ingredient_matrix[new_index_order, :]
        # reorder columns
        reordered_matrix = reordered_matrix[:, new_index_order]
        return reordered_matrix

    def grid_search_k_yielder(self, ks, output_path_prefix):
        for k in ks:
            mixed_matrix = self.mix(k)
            # np.save(output_path_prefix + str(k) + '.npy', mixed_matrix, fix_imports=False)
            yield k, mixed_matrix


class MatrixNormalization(object):
    def __init__(self, matrix):
        self.matrix = matrix

    @classmethod
    def from_storage(cls, matrix_path):
        matrix = np.load(matrix_path)
        return cls(matrix)

    def pmi_without_log(self):
        normalized_matrix = np.copy(self.matrix)
        # following lines a bit tedious, as we try to avoid making temporary copies of the (large) `cooccur` matrix
        marginal_word = normalized_matrix.sum(axis=1)
        marginal_context = normalized_matrix.sum(axis=0)
        normalized_matrix /= marginal_word[:, None]  # #(w, c) / #w
        normalized_matrix /= marginal_context  # #(w, c) / (#w * #c)
        normalized_matrix *= marginal_word.sum()  # #(w, c) * D / (#w * #c)
        return normalized_matrix

    def line_normalization(self):
        normalized_matrix = np.copy(self.matrix)
        line_sum = np.sum(normalized_matrix, axis=1, keepdims=True)  # sum of each row and preserve the dimension
        normalized_matrix /= line_sum
        return normalized_matrix


class MatrixSmoothing(object):
    def __init__(self, matrix):
        self.matrix = matrix

    @classmethod
    def from_storage(cls, matrix_path):
        matrix = np.load(matrix_path)
        return cls(matrix)

    def log_shifted_positive(self, k_shift):
        smoothed_matrix = np.copy(self.matrix)
        np.log(smoothed_matrix, out=smoothed_matrix)  # PMI = log(#(w, c) * D / (#w * #c))

        if k_shift:
            smoothed_matrix -= np.log(k_shift)  # shifted PMI = log(#(w, c) * D / (#w * #c)) - log(k)

        # clipping PMI scores to be non-negative PPMI
        smoothed_matrix.clip(0.0, out=smoothed_matrix)  # SPPMI = max(0, log(#(w, c) * D / (#w * #c)) - log(k))

        # # normalizing PPMI word vectors to unit length
        # for i, vec in enumerate(cooccur):
        #     cooccur[i] = matutils.unitvec(vec)
        return smoothed_matrix


class MatrixDimensionReducer(object):
    @staticmethod
    def truncated_svd(matrix, dimension):
        svd = TruncatedSVD(n_components=dimension)
        svd.fit(matrix)
        result = svd.transform(matrix)
        return result


class MatrixMasker(object):
    def __init__(self, matrix):
        self.matrix = matrix

    def get_matrix_info(self, additional_matrix):
        zero_indices_x, zero_indices_y = np.where(self.matrix == 0)
        zero_cells_count = len(zero_indices_x)
        print('#zero cells:', zero_cells_count)

        count = 0
        for i in range(len(zero_indices_x)):
            if additional_matrix[zero_indices_x[i]][zero_indices_y[i]] != 0:
                count += 1
        print('#non-zero cells in additional matrix in the corresponding zero positions:', count,
              "{0:.0%}".format(count/zero_cells_count))

    def get_zero_positions(self):
        # find zero position in the matrix
        zero_indices_x, zero_indices_y = np.where(self.matrix == 0)
        temp_matrix = np.copy(self.matrix)
        for i in range(len(zero_indices_x)):
            temp_matrix[zero_indices_x[i]][zero_indices_y[i]] = max_value


def save_enhanced_matrix(matrix, output_path):
    np.save(output_path, matrix, fix_imports=False)


def matrix_vis(matrix, output_path):
    # # find zero position in the matrix
    # zero_indices_x, zero_indices_y = np.where(matrix == 0)
    # # find the second minimum value in matrix, temp_matrix is used for that
    # max_value = np.amax(matrix)
    # temp_matrix = np.copy(matrix)
    # for i in range(len(zero_indices_x)):
    #     temp_matrix[zero_indices_x[i]][zero_indices_y[i]] = max_value
    # second_minimum = np.amin(temp_matrix)  # first minimum is always 0
    # # set all zeros to second minimum value
    # for i in range(len(zero_indices_x)):
    #     matrix[zero_indices_x[i]][zero_indices_y[i]] = second_minimum

    # matrix = np.log10(matrix)  # Necessary for negative samples matrix, nearly all black if not.
    matrix_to_show = np.copy(matrix)
    for i in range(matrix_to_show.shape[0]):
        for j in range(matrix_to_show.shape[1]):
            if matrix_to_show[i][j] != 0:
                matrix_to_show[i][j] = 1

    plt.imshow(matrix_to_show, cmap="nipy_spectral")  # plt.cm.BuPu_r, hot -> bad choices (no big difference)
    plt.colorbar()
    # print(np.amax(matrix))
    # print(np.amin(matrix))
    # plt.show()
    plt.savefig(output_path)
    plt.clf()


if __name__ == '__main__':
    pass
    # MatrixEnhancer.from_encoded_edges_count_file_path(encoded_edges_count_file_path='/Users/zzcoolj/Desktop/GoW_new_ideas/input/cooccurrence matrix/encoded_edges_count_window_size_5_undirected.txt',
    #                                                   valid_wordIds_path='/Users/zzcoolj/Desktop/GoW_new_ideas/input/cooccurrence matrix/valid_vocabulary_min_count_5_vocab_size_10000.txt',
    #                                                   merged_dict_path='/Users/zzcoolj/Desktop/GoW_new_ideas/input/dict_merged.txt',
    #                                                   output_folder='input/')

    # m = MatrixEnhancer.from_storage(matrix_path='input/encoded_edges_count_window_size_5_undirected_matrix.npy',
    #                                 tokens_path='input/encoded_edges_count_window_size_5_undirected_tokens.pickle')
    # result = MatrixEnhancer.truncated_svd(m.raw2ppmi(), 1000)
    # MatrixEnhancer.save_enhanced_matrix(result, 'test.npy')

