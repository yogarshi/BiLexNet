import codecs
import itertools

import numpy as np


def get_id(corpus, key):
    """
    Get the corpus ID of a word
    (handle utf-8 encoding errors, following change
    https://github.com/vered1986/LexNET/pull/2 from @gossebouma)
    :param corpus: the corpus' resource object
    :param key: the word
    :return: the ID of the word or -1 if not found
    """
    id = -1  # the index of the unknown word
    try:
        id = corpus.get_id_by_term(key.encode('utf-8'))
    except UnicodeEncodeError:
        pass
    return id


def vectorize_path(path, lemma_index, pos_index, dep_index, dir_index):
    """
    Return a vector representation of the path
    :param path: the string representation of a path
    :param lemma_index: index to lemma dictionary
    :param pos_index: index to POS dictionary
    :param dep_index: index to dependency label dictionary
    :param dir_index: index to edge direction dictionary
    :return:
    """
    path_edges = [vectorize_edge(edge, lemma_index, pos_index, dep_index, dir_index) for edge in path.split('_')]

    if None in path_edges:
        return None

    return tuple(path_edges)


def vectorize_edge(edge, lemma_index, pos_index, dep_index, dir_index):
    """
    Return a vector representation of the edge: concatenate lemma/pos/dep and add direction symbols
    :param edge: the string representation of an edge
    :param lemma_index: index to lemma dictionary
    :param pos_index: index to POS dictionary
    :param dep_index: index to dependency label dictionary
    :param dir_index: index to edge direction dictionary
    :return:
    """
    try:
        lemma, pos, dep, direction = edge.split('/')
        lemma, pos, dep, direction = lemma_index.get(lemma.lower(), 0), pos_index[pos], dep_index[dep], dir_index[direction]
    except:
        return None

    return tuple([lemma, pos, dep, direction])


def reconstruct_edge(lemma, pos, dep, direction,
                     lemma_inverted_index, pos_inverted_index, dep_inverted_index, dir_inverted_index):
    """
    Return a string representation of the edge
    :param lemma_inverted_index: lemma to index dictionary
    :param pos_inverted_index: POS to index dictionary
    :param dep_inverted_index: dependency label to index dictionary
    :param dir_inverted_index: edge direction to index dictionary
    :return: The string representation of the edge
    """
    edge = '/'.join([lemma_inverted_index[lemma], pos_inverted_index[pos], dep_inverted_index[dep],
                     dir_inverted_index[direction]])
    return edge


def load_embeddings(file_name, vocabulary):
    """
    Load the pre-trained embeddings from a file
    :param file_name: the embeddings file
    :param vocabulary: limited vocabulary to load vectors for
    :return: the vocabulary and the word vectors
    """
    words = []
    vectors = []

    words_set = set()
    with codecs.open(file_name, 'r', 'utf-8') as f_in:


        # x = f_in.readline()
        for each_line in f_in:
           # print (each_line.encode('utf-8'))
           try:
               c_word, c_vector = each_line.strip().split(' ', 1)
           except ValueError:
               continue
           if len(c_vector.split()) != 300:
              continue
           if c_word not in vocabulary:
               continue
           if c_word in words_set:
                continue
           words.append(c_word)
           words_set.add(c_word)
           vectors.append(c_vector)
        # words, vectors = zip(*[line.strip().split(' ', 1) for line in f_in])
    wv = np.loadtxt(vectors)
    #print (wv.shape)
    # dim=wv.shape[1]
    dim = 300
    pad_embed = np.ones((dim,))
    wv = np.vstack((pad_embed,wv ))

    words = ['#PAD#'] + list(words)

    # Add the unknown words
    unknown_vector = np.random.random_sample((dim,))

    word_set = set(words)
    unknown_words = list(set(vocabulary).difference(set(words)))

    # Create vectors for MWEs - sum of word embeddings, and OOV words
    unknown_word_vectors = [np.add.reduce([wv[words.index(w)] if w in word_set else unknown_vector
                                           for w in word.split(' ')])
                            for word in unknown_words]

    wv = np.vstack((wv, unknown_word_vectors))
    words = list(words) + unknown_words

    print ('Known lemmas:', len(vocabulary) - len(unknown_words), '/', len(vocabulary))

    # Normalize each row (word vector) in the matrix to sum-up to 1
    row_norm = np.sum(np.abs(wv) ** 2, axis=-1) ** (1. / 2)
    wv /= row_norm[:, np.newaxis]

    word_index = {w: i for i, w in enumerate(words)}
    wv[0] = 0
    return wv, word_index


def load_test_dataset(dataset_file, relations):
    """
    Loads a dataset file
    :param dataset_file: the file path
    :return: a list of dataset instances, (x, y, relation)
    """
    with codecs.open(dataset_file, 'r', 'utf-8') as f_in:
        dataset = {}
        dataset_en_trans = {}
        dataset_hi_trans = {}
        for line in f_in:
            x, y, relation, x_hi_trans, y_en_trans = line.strip().split('\t')
            relation = relation.strip()
            x = x.strip()
            y = y.strip()
            x_hi_trans = x_hi_trans.strip()
            y_en_trans = y_en_trans.strip()
            if relation in relations:
                dataset[(x.lower(), y.lower())] = relation
                dataset_hi_trans[(x.lower(), y.lower())] = [
                    (x_hi.lower(), y.lower()) for x_hi in x_hi_trans.split(',')]
                dataset_en_trans[(x.lower(), y.lower())] = [
                    (x.lower(), y_en.lower()) for y_en in y_en_trans.split(',')]
                # dataset = [tuple(line.strip().split('\t')) for line in f_in]
                # dataset = {(x.lower(), y.lower()): relation for (x, y, relation) in dataset if relation in relations}

    return dataset, dataset_hi_trans, dataset_en_trans




def load_dataset(dataset_file, relations):
    """
    Loads a dataset file
    :param dataset_file: the file path
    :return: a list of dataset instances, (x, y, relation)
    """
    with codecs.open(dataset_file, 'r', 'utf-8') as f_in:
        dataset = {}
        dataset_en_trans = {}
        dataset_hi_trans = {}
        for line in f_in:
            x, y, relation, x_hi_trans, y_en_trans = line.strip().split('\t')
            relation = relation.strip()
            x = x.strip()
            y = y.strip()
            x_hi_trans = x_hi_trans.strip()
            y_en_trans = y_en_trans.strip()
            if relation in relations:
                dataset[(x.lower(),y.lower())] = relation
                dataset_hi_trans[(x.lower(), y.lower())] = [(x_hi.lower(), y.lower()) for x_hi in x_hi_trans.split(',')]
                dataset_en_trans[(x.lower(),y.lower())] = [ (x.lower(), y_en.lower()) for y_en in y_en_trans.split(',')]
                #dataset = [tuple(line.strip().split('\t')) for line in f_in]
                #dataset = {(x.lower(), y.lower()): relation for (x, y, relation) in dataset if relation in relations}
            else:
                print line

    return dataset, dataset_hi_trans, dataset_en_trans


def unique(lst):
    """
    :param lst: a list of lists
    :return: a unique list of items appearing in those lists
    """
    indices = sorted(range(len(lst)), key=lst.__getitem__)
    indices = set(next(it) for k, it in
                  itertools.groupby(indices, key=lst.__getitem__))
    return [x for i, x in enumerate(lst) if i in indices]


def get_paths(corpus, x, y):
    """
    Get the paths that connect x and y in the corpus
    :param corpus: the corpus' resource object
    :param x: x
    :param y: y
    :return:
    """
    x_to_y_paths = corpus.get_relations(x, y)
    y_to_x_paths = corpus.get_relations(y, x)
    paths = {corpus.get_path_by_id(path): count for (path, count) in x_to_y_paths.iteritems()}
    paths.update({corpus.get_path_by_id(path).replace('X/', '@@@').replace('Y/', 'X/').replace('@@@', 'Y/'): count
                  for (path, count) in y_to_x_paths.iteritems()})
    return paths
