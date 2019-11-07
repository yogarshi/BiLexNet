import sys

import argparse
import torch
import time
import random
import numpy as np
import cPickle as pickle

ap = argparse.ArgumentParser()
ap.add_argument('--dataset_prefix', help='path to the train/test/val/rel data', default='/Users/yogarshi/Google Drive/Acads/Research/tmp/data/training_data/train_xxl_zh')
ap.add_argument('--xlingual_path_file', default='/Users/yogarshi/Google Drive/Acads/Research/tmp/data/training_data/train_xxl/all.xlingual.count.small')
ap.add_argument('--mono_path_file_en', default='/Users/yogarshi/Google Drive/Acads/Research/tmp/data/training_data/train_xxl_zh/all.mono.count.small')
ap.add_argument('--mono_path_file_hi', default='/Users/yogarshi/Google Drive/Acads/Research/tmp/data/training_data/test_final/train+val+test.mono_hi.paths.count.small')
ap.add_argument('--embeddings_file_en', default='/Users/yogarshi/Google Drive/Acads/Research/tmp/data/training_data/wiki.en.align.vec.small')
ap.add_argument('--embeddings_file_hi', default='/Users/yogarshi/Google Drive/Acads/Research/tmp/data/training_data/train_xxl_zh/wiki.zh.align.vec.simp.small')
ap.add_argument('--model_prefix_file', help='where to store the result', default='./model')
ap.add_argument('--num_epochs', help='number of epochs to train', type=int, default=10)
ap.add_argument('-g', '--gpus', help='number of gpus to use [0,1], default=0', type=int, default=0, choices=[0,1])
ap.add_argument('--num_hidden_layers', help='number of hidden layers to use', type=int, default=0)
ap.add_argument('-s', '--seed', help='random seed', type=int, default=100)
ap.add_argument('-t', '--temp', help='student teacher training temperature', type=float, default=20)
ap.add_argument('--lr', help='learning rate',type=float,default=0.001)
ap.add_argument('--kdalpha', help='student teacher training alpha', type=float, default=0.5)
ap.add_argument('--fold',type=int, default=0)

print ap
args = ap.parse_args()
print args
sys.path.append('../common/')

from lstm_common import *
from itertools import count
from evaluation_common import *
from collections import defaultdict
from paths_student_teacher_pooling import PathLSTMClassifier

EMBEDDINGS_DIM = 50
MAX_PATHS_PER_PAIR = 20000 # Set to K > 0 if you want to limit the number of path per pair (for memory reasons)

def load_xlingual_paths_train(xlingual_path_file,term_pairs, trans_pairs):
    """
    Each line in xlingual_path_file is of the type
    w_e w_h path lang count
    term pairs = list of x,y pairse
    trans pairs = list of list of x,y pairs
    :param xlingual_path_file:
    :param term_pairs:
    :return:
    """
    all_paths_en = {}
    all_paths_hi = {}

    trans_pairs_flat =  set([item for sublist in trans_pairs for item in sublist])
    # print trans_pairs_flat[:50]
    sys.stdout.flush()
    with codecs.open(xlingual_path_file, 'r', 'utf-8') as f:
        for each_line in f:
            w_e, w_h, path, lang, count = each_line.strip().split("\t")
            key = (w_e, w_h)
            if key not in trans_pairs_flat:
                continue
            if lang == "en":
                if key not in all_paths_en:
                    all_paths_en[key] = {}
                all_paths_en[key][path] = int(count)
            elif lang == "hi":
                if key not in all_paths_hi:
                    all_paths_hi[key] = {}
                all_paths_hi[key][path] = int(count)

    all_paths_final_en = {}
    for i in range(len(term_pairs)):
        real_key = term_pairs[i]
        all_paths_final_en[real_key] = {}
        for fake_key in trans_pairs[i]:
            all_paths_final_en[real_key][fake_key] = {}
            if fake_key in all_paths_en:
                for path in all_paths_en[fake_key]:
                    count = all_paths_en[fake_key][path]
                    all_paths_final_en[real_key][fake_key][path] = count

    all_paths_final_hi = {}
    for i in range(len(term_pairs)):
        real_key = term_pairs[i]
        all_paths_final_hi[real_key] = {}
        for fake_key in trans_pairs[i]:
            all_paths_final_hi[real_key][fake_key] = {}
            if fake_key in all_paths_hi:
                for path in all_paths_hi[fake_key]:
                    count = all_paths_hi[fake_key][path]
                    all_paths_final_hi[real_key][fake_key][path] = count

    return all_paths_final_en, all_paths_final_hi

def load_monolingual_paths(monolingual_path_file,term_pairs, trans_pairs):
    """
    Each line in xlingual_path_file is of the type
    w_e w_h path lang count
    term pairs = list of x,y pairse
    trans pairs = list of list of x,y pairs
    :param xlingual_path_file:
    :param term_pairs:
    :return:
    """
    all_paths = {}
    trans_pairs_flat =  set([item for sublist in trans_pairs for item in sublist])
    # trans_pairs_rev = {}
    # for
    # print trans_pairs_flat[:50]
    sys.stdout.flush()
    with codecs.open(monolingual_path_file, 'r', 'utf-8') as f:
        for each_line in f:
            w_e, w_h, path, count = each_line.strip().split("\t")
            key = (w_e,w_h)
            if key not in trans_pairs_flat:
                continue
            if key not in all_paths:
                all_paths[key] = {}
            all_paths[key][path] = int(count)

    all_paths_final = {}
    for i in range(len(term_pairs)):
        real_key = term_pairs[i]
        all_paths_final[real_key] = {}
        for fake_key in trans_pairs[i]:
            all_paths_final[real_key][fake_key] = {}
            if fake_key in all_paths:
                for path in all_paths[fake_key]:
                    count = all_paths[fake_key][path]
                    all_paths_final[real_key][fake_key][path] = count

    return all_paths_final

def load_monolingual_paths_simple(xlingual_path_file,term_pairs):
    """
    Each line in xlingual_path_file is of the type
    w_e w_h path lang count
    :param xlingual_path_file:
    :param term_pairs:
    :return:
    """
    all_paths_en = {}
    with codecs.open(xlingual_path_file, 'r', 'utf-8') as f:
        for each_line in f:
            w_e, w_h, path, lang, count = each_line.strip().split("\t")
            key = (w_e,w_h)
            if key not in term_pairs:
                continue
            if key not in all_paths_en:
                all_paths_en[key] = {}
            all_paths_en[key][path] = int(count)

    return all_paths_en

def load_xlingual_paths(xlingual_path_file,term_pairs):
    """
    Each line in xlingual_path_file is of the type
    w_e w_h path lang count
    :param xlingual_path_file:
    :param term_pairs:
    :return:
    """
    all_paths_en = {}
    all_paths_hi = {}
    with codecs.open(xlingual_path_file, 'r', 'utf-8') as f:
        for each_line in f:
            w_e, w_h, path, lang, count = each_line.strip().split("\t")
            key = (w_e,w_h)
            if key not in term_pairs:
                continue
            if lang == "en":
                if key not in all_paths_en:
                    all_paths_en[key] = {}
                all_paths_en[key][path] = int(count)
            elif lang == "hi":
                if key not in all_paths_hi:
                    all_paths_hi[key] = {}
                all_paths_hi[key][path] = int(count)

    return all_paths_en, all_paths_hi

def tensorize_paths(paths_x_to_y, use_gpu=False):
    tensorized_paths = []
    for i in range(len(paths_x_to_y)):
        # if i % 100 == 0:
        #     print (i)
        wp_counts = []
        wp_paths = []

        iter = 0
        for p in paths_x_to_y[i]:
            if iter > MAX_PATHS_PER_PAIR:
                break
            iter += 1
            if p is None:
                continue
            c = paths_x_to_y[i][p]
            wp_counts.append(c)

            curr_words = []
            curr_dep = []
            curr_pos = []
            curr_dirs = []

            # For each element in the current path
            for word, pos, dep, dir in p:
                curr_words.append(word)
                curr_dep.append(dep)
                curr_pos.append(pos)
                curr_dirs.append(dir)

            curr_path = torch.stack(
                [torch.LongTensor(curr_words), torch.LongTensor(curr_dep),
                 torch.LongTensor(curr_pos), torch.LongTensor(curr_dirs)])
            curr_path = curr_path.transpose(0, 1)
            wp_paths.append(curr_path)
        # Sort paths by length to enable batching of LSTM
        path_lengths = torch.LongTensor([path.shape[0] for path in wp_paths])

        if wp_paths:
            lengths, perm_index = path_lengths.sort(0, descending=True)
            k = torch.nn.utils.rnn.pad_sequence(wp_paths, batch_first=True)
            k = k[perm_index]
            wp_counts = torch.FloatTensor(wp_counts)

            if use_gpu:
                k = k.cuda()
                wp_counts = wp_counts.cuda()
            tensorized_paths.append((k, lengths, wp_counts))
            # print ("Number of paths = {0}".format(len(wp_paths)))
        else:
            tensorized_paths.append(None)
            # print ("Number of paths = 0")

    return tensorized_paths




def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    sys.stdout.write("Using random seed = {0}\n".format(args.seed))


    if args.gpus == 1:
        device ="cuda:0"
        use_gpu = True
        print ("Using GPU")
        torch.cuda.manual_seed(args.seed)
    else:
        device = "cpu"
        use_gpu = False
        print ("Using CPU")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # Load the relations
    with codecs.open(args.dataset_prefix + '/relations.txt', 'r', 'utf-8') as f_in:
        relations = [line.strip() for line in f_in]
        relation_index = { relation : i for i, relation in enumerate(relations) }

    # Load the datasets
    print ('Loading the dataset...')
    train_set, train_set_trans_hi, train_set_trans_en = load_dataset(args.dataset_prefix + '/train-clean.tsv', relations)
    assert len(train_set) == len(train_set_trans_hi)
    assert len(train_set) == len(train_set_trans_en)
    val_set, val_set_trans_hi, val_set_trans_en  = load_dataset(args.dataset_prefix + '/val-entrans.tsv', relations)
    assert len(val_set) == len(val_set_trans_hi)
    assert len(val_set) == len(val_set_trans_en)
    test_set, test_set_trans_hi, test_set_trans_en = load_dataset(args.dataset_prefix + '/en-trans-sorted.tsv'.format(args.fold), relations)
    assert len(test_set) == len(test_set_trans_hi)
    assert len(test_set) == len(test_set_trans_en)

    print ("Training example = {0}, Validation exmaples = {1}, Test examples = {2}".format(len(train_set), len(val_set), len(test_set)))
    y_train = [relation_index[label] for label in train_set.values()]
    y_val = [relation_index[label] for label in val_set.values()]
    y_test = [relation_index[label] for label in test_set.values()]
    # dataset_keys = list(train_set.keys()) + list(val_set.keys()) + list(test_set.keys())
    dataset_trans_hi = [train_set_trans_hi[key] for key in train_set.keys()]  + [val_set_trans_hi[key] for key in val_set.keys()]  + [test_set_trans_hi[key] for key in test_set.keys()]
    dataset_trans_en = [train_set_trans_en[key] for key in train_set.keys()] + [val_set_trans_en[key] for key in val_set.keys()] + [test_set_trans_en[key] for key in test_set.keys()]
    print ('Done!')
    sys.stdout.flush()

    # COunt number of instances of each class
    count_instances = [0 for key in relation_index]
    for each_element in y_train:
        count_instances[each_element] += 1
    count_instances = [1.0/(x/float(len(y_train))) for x in count_instances]
    print relation_index
    # print count_instances




    # Load the paths and create the feature vectors
    print ('Loading path files...')
    start = time.time()
    #word_vectors_en, word_vectors_hi, x_y_vectors, trans_lemmas_vectors, dataset_instances_mono_en, dataset_instances_mono_hi, dataset_instances_xlingual_en, dataset_instances_xlingual_hi, \
    #word_index_en, word_index_hi, pos_index, dep_index, dir_index, word_inverted_index_en, word_inverted_index_hi, \
    #pos_inverted_index, dep_inverted_index, dir_inverted_index \
    #  = load_paths_and_word_vectors((train_set.keys(), val_set.keys(),test_set.keys()), dataset_trans_en, dataset_trans_hi,
    #                              args.mono_path_file_en, args.mono_path_file_hi, args.xlingual_path_file,
    #                                args.embeddings_file_en, args.embeddings_file_hi, use_gpu)

    word_vectors_en, word_vectors_hi, x_y_vectors, trans_lemmas_vectors, dataset_instances_mono_en, dataset_instances_mono_hi, dataset_instances_xlingual_en, dataset_instances_xlingual_hi, \
	word_index_en, word_index_hi, pos_index, dep_index, dir_index, word_inverted_index_en, word_inverted_index_hi, \
	pos_inverted_index, dep_inverted_index, dir_inverted_index  = pickle.load(open("/fs/clip-scratch/yogarshi/data-hi-en-new.pkl".format(args.fold)))

    # word_vectors_en, word_vectors_hi, x_y_vectors, trans_lemmas_vectors, dataset_instances_mono_en, dataset_instances_mono_hi, dataset_instances_xlingual_en, dataset_instances_xlingual_hi, \
    # word_index_en, word_index_hi, pos_index, dep_index, dir_index, word_inverted_index_en, word_inverted_index_hi, \
    # pos_inverted_index, dep_inverted_index, dir_inverted_index = pickle.load(
    #     open("/fs/clip-scratch/yogarshi/data-hi-en-transbaseline.pkl"))

    # x_y_vectors = torch.cat((x_y_vectors[:10],  x_y_vectors[len(train_set):]), dim=0)
    # trans_lemmas_vectors = trans_lemmas_vectors[:10] + trans_lemmas_vectors[len(train_set):]
    # dataset_instances_mono_en = dataset_instances_mono_en[:10] + dataset_instances_mono_en[len(train_set):]
    # dataset_instances_mono_hi = dataset_instances_mono_hi[:10]+ dataset_instances_mono_hi[len(train_set):]
    # dataset_instances_xlingual_en = dataset_instances_xlingual_en[:10] + dataset_instances_xlingual_en[len(train_set):]
    # dataset_instances_xlingual_hi = dataset_instances_xlingual_hi[:10] + dataset_instances_xlingual_hi[len(train_set):]
    # y_train = y_train[:10]

    # word_vectors_en = word_vectors_en[:10000]
    # word_vectors_hi = word_vectors_hi[:10000]
    # x_y_vectors = x_y_vectors[:10000]
    # trans_lemmas_vectors = trans_lemmas_vectors[:10000]
    # dataset_instances_mono_en = dataset_instances_mono_en[:10000]
    # dataset_instances_mono_hi = dataset_instances_mono_hi[:10000]
    # dataset_instances_xlingual_en = dataset_instances_xlingual_en[:10000]
    # dataset_instances_xlingual_hi = dataset_instances_xlingual_hi[:10000]

    #print (trans_lemmas_vectors[:10])
    #print (dataset_instances_xlingual_en[:10])
    #print (dataset_instances_xlingual_hi[:10])
    #print (dataset_instances_xlingual_en[0].shape)
    #print (dataset_instances_xlingual_hi[0].shape)
    #print (dataset_instances_xlingual_en[1].shape)
    #print (dataset_instances_xlingual_hi[1].shape)



    print ('# of English words =  {0}, # of Hindi words = {1}, # of pos tags =  {2}, # of dependency labels: {3}, # of directions: {4}'
    .format(len(word_index_en),len(word_index_hi), len(pos_index), len(dep_index), len(dir_index)))

    print (len(x_y_vectors))
    print ("Loaded everything. Time = {0}".format(time.time() - start))

    # frac = int(len(train_set)/1.5)
    frac = len(train_set)

    X_train_xlingual_en = dataset_instances_xlingual_en[:frac]
    X_val_xlingual_en = dataset_instances_xlingual_en[len(train_set):len(train_set)+len(val_set)]
    X_test_xlingual_en = dataset_instances_xlingual_en[len(train_set)+len(val_set):]

    #Pairs with cross-lingual paths


    X_train_xlingual_hi = dataset_instances_xlingual_hi[:frac]
    X_val_xlingual_hi = dataset_instances_xlingual_hi[len(train_set):len(train_set) + len(val_set)]
    X_test_xlingual_hi = dataset_instances_xlingual_hi[len(train_set) + len(val_set):]

    X_train_mono_en = dataset_instances_mono_en[:frac]
    X_val_mono_en = dataset_instances_mono_en[len(train_set):len(train_set) + len(val_set)]
    X_test_mono_en = dataset_instances_mono_en[len(train_set) + len(val_set):]

    X_train_mono_hi = dataset_instances_mono_hi[:frac]
    X_val_mono_hi = dataset_instances_mono_hi[len(train_set):len(train_set) + len(val_set)]
    X_test_mono_hi = dataset_instances_mono_hi[len(train_set) + len(val_set):]

    invalid_xling_test = [i for i in range(len(X_test_xlingual_en)) if X_test_xlingual_en[i] is not  None or X_test_xlingual_hi[i] is not None]
    y_test_invalid_xling = [y_test[i] for i in invalid_xling_test]

    x_y_vectors_train = x_y_vectors[:frac]
    x_y_vectors_val = x_y_vectors[len(train_set):len(train_set)+len(val_set)]
    x_y_vectors_test = x_y_vectors[len(train_set)+len(val_set):]


    #TODO - Trans lemma needs to be split
    trans_lemmas_vectors_train = trans_lemmas_vectors[:frac]
    trans_lemmas_vectors_val = trans_lemmas_vectors[len(train_set):len(train_set) + len(val_set)]
    trans_lemmas_vectors_test = trans_lemmas_vectors[len(train_set) + len(val_set):]

    y_train = y_train[:frac]

    count_instances = [0 for key in relation_index]
    for each_element in y_train:
        count_instances[each_element] += 1
    print (count_instances)


    sys.stdout.flush()

    # Tune the hyper-parameters using the validation set
    alphas = [args.lr]
    # word_dropout_rates = [0,0.1,0.2,0.3,0.4,0.5]
    word_dropout_rates = [0.1]
    model_types = ['xling_only']

    sys.stderr.write("Parameters : {0}\t{1}\n".format(args.temp,args.kdalpha))

    for model_type in model_types:
        f1_results = []
        models = []
        descriptions = []
        for alpha in alphas:
            for word_dropout_rate in word_dropout_rates:
                # for temperature in [2, 5,10,15, 20]:
                for temperature in [args.temp]:
                    for kd_alpha in [args.kdalpha]:

                # for temperature in [1,]:
                #     for kd_alpha in [0.1]:

                        # Create the classifier
                        classifier = PathLSTMClassifier(model_type = model_type,
                                                        num_lemmas_en=len(word_index_en), num_lemmas_hi=len(word_index_hi), num_pos=len(pos_index),
                                                        num_dep=len(dep_index), num_directions=len(dir_index),
                                                        n_epochs=args.num_epochs,
                                                        num_relations=len(relations),alpha=alpha,
                                                        lemma_embeddings_en=word_vectors_en,
                                                        lemma_embeddings_hi=word_vectors_hi,
                                                        en_vocab = word_inverted_index_en,
                                                        hi_vocab = word_inverted_index_hi,
                                                        label_dict= { relation_index[key] : key for key in relation_index},
                                                        relations = relations,
                                                        dropout=word_dropout_rate,  use_xy_embeddings=True,
                                                        num_hidden_layers=args.num_hidden_layers, use_gpu=use_gpu,project_embeds=False, temperature=temperature, kd_alpha=kd_alpha)
                        if use_gpu:
                            classifier.cuda()

                        print ('Training with learning rate = %f, dropout = %f...' % (alpha, word_dropout_rate))
                        sys.stdout.flush()
                        classifier.fit(X_train_xlingual_en, X_train_xlingual_hi, X_train_mono_en,X_train_mono_hi, y_train, x_y_vectors_train, trans_lemmas_vectors_train, count_instances,
                                       X_val_xlingual_en, X_val_xlingual_hi, X_val_mono_en, X_val_mono_hi,x_y_vectors_val,trans_lemmas_vectors_val, y_val)
                        pred,vecs, probs= classifier.predict(X_val_xlingual_en, X_val_xlingual_hi, X_val_mono_en,X_val_mono_hi, x_y_vectors_val, trans_lemmas_vectors_val)
                        precision, recall, f1, support = evaluate(y_val, pred, relations, do_full_reoprt=False)
                        print ('Model type = %s, KD_alpha = %f, temperature = %f, Precision: %.3f, Recall: %.3f, F1: %.3f' % \
                              (model_type, kd_alpha, temperature, precision, recall, f1))
                        sys.stdout.flush()
                        f1_results.append(f1)
                        models.append(classifier)
                        # print (classifier.mono_w, classifier.xling_w)

                        pred, vecs, probs = classifier.predict(X_test_xlingual_en,
                                                               X_test_xlingual_hi,
                                                               X_test_mono_en,
                                                               X_test_mono_hi,
                                                               x_y_vectors=x_y_vectors_test,
                                                               trans_lemmas_vectors=trans_lemmas_vectors_test,
                                                               test_time=True)

                        precision, recall, f1, support = evaluate(y_test, pred, relations,do_full_reoprt=True)
                        print ('Precision: %.3f, Recall: %.3f, F1: %.3f' % (precision, recall, f1))

                        # Save intermediate models
                        # torch.save(classifier,)
                        # classifier.save_model(args.model_prefix_file + '.' + str(word_dropout_rate),
                        #                       [word_index, pos_index, dep_index, dir_index])
                        descriptions.append('KD alpha = {0}, temperature = {1}, model type = {2}'.format(kd_alpha, temperature, model_type))
                        sys.stdout.flush()

        best_index = np.argmax(f1_results)
        classifier = models[best_index]
        description = descriptions[best_index]
        print ('Best hyper-parameters for model {0}, {1} : '.format(description, model_type))
        print ('Evaluation:')
        pred, vecs, probs = classifier.predict(X_test_xlingual_en, X_test_xlingual_hi,
                                  X_test_mono_en, X_test_mono_hi, x_y_vectors_test, trans_lemmas_vectors_test, test_time=True)

        pred_val, vecs_val, probs_val = classifier.predict(X_val_xlingual_en,
                                               X_val_xlingual_hi,
                                               X_val_mono_en, X_val_mono_hi,
                                               x_y_vectors_val,
                                               trans_lemmas_vectors_val,
                                               test_time=True)

        if model_type == "joint":
            with open("paths.joint.test", 'w') as f_out:
                num_vecs = len(vecs[0])
                num_pairs = len(vecs)
                for i in range(num_pairs):
                    for j in range(num_vecs):
                        f_out.write("{0}\t".format(' '.join(map(str,vecs[i][j]))))
                    f_out.write(str(y_test[i]))
                    f_out.write("\n")

        pred_invalid_xling = [pred[i] for i in invalid_xling_test]
        # print pred_invalid_xling
        # print y_test_invalid_xling
        # precision, recall, f1, support = evaluate(y_test, pred, relations,
        #                                           do_full_reoprt=True)
        # print (
        #     'Precision: %.3f, Recall: %.3f, F1: %.3f' % (precision, recall, f1))
        # precision, recall, f1, support = evaluate(y_test_invalid_xling, pred_invalid_xling, relations,
        #                                           do_full_reoprt=True)

        print (
            'Invalid - Precision: %.3f, Recall: %.3f, F1: %.3f' % (precision, recall, f1))


        # Save the best model to a file
    #         print ('Saving the model...')
    #         torch.save(classifier.state_dict(),)
    #         classifier.save_model(args.model_prefix_file, [word_index, pos_index, dep_index, dir_index])
    # dir_index
    # Evaluate on the test se

    # Write the predictions to a file
    output_predictions(args.model_prefix_file + '.predictions_zh-en.{0}.{1}.{2}.valtest-toupload'.format(args.seed,args.temp,args.kdalpha), relations, pred, probs, test_set.keys(), y_test)
    output_predictions(args.model_prefix_file + '.predictions_zh-en.{0}.{1}.{2}.test-toupload'.format(args.seed, args.temp, args.kdalpha), relations, pred_val, probs_val,val_set.keys(), y_val)

    # Retrieve k-best scoring paths for each class
    # all_paths = unique([path for path_list in dataset_instances for path in path_list])
    # top_k = classifier.get_top_k_paths(all_paths, relation_index, 0.7)
	#
    # for i, relation in enumerate(relations):
    #     with codecs.open(args.model_prefix_file + '.paths.' + relation, 'w', 'utf-8') as f_out:
    #         for path, score in top_k[i]:
    #             path_str = '_'.join([reconstruct_edge(edge, word_inverted_index, pos_inverted_index,
    #                                                   dep_inverted_index, dir_inverted_index) for edge in path])
    #             print >> f_out, '\t'.join([path_str, str(score)])

def load_paths_and_word_vectors_transbase(dataset_keys, dataset_trans_en,
                                dataset_trans_hi, mono_en_path_file,
                                mono_hi_path_file,
                                xlingual_path_file, embeddings_file_en,
                                embeddings_file_hi, use_gpu):
    '''
    Load the paths and the word vectors for this dataset
    :param corpus: the corpus object
    :param dataset_keys: the word pairs in the dataset
    :param word_index: the index of words for the word embeddings
    :return:
    '''

    # Define the dictionaries
    train_set_keys, val_set_keys, test_set_keys = dataset_keys
    dataset_keys = train_set_keys + val_set_keys + test_set_keys

    pos_index = defaultdict(count(0).next)
    dep_index = defaultdict(count(0).next)
    dir_index = defaultdict(count(0).next)

    # Padding is 0
    _ = pos_index['#PAD#']
    _ = dep_index['#PAD#']
    _ = dir_index['#PAD#']

    # Unknowns are 1
    _ = pos_index['#UNKNOWN#']
    _ = dep_index['#UNKNOWN#']
    _ = dir_index['#UNKNOWN#']

    # Vectorize the paths
    keys = []
    en_vocabulary = set()
    hi_vocabulary = set()
    for (x, y) in train_set_keys + val_set_keys + test_set_keys:
        keys.append((x, y))
        en_vocabulary.add(x)
        en_vocabulary.add(y)
    # keys = [(get_id(corpus, x), get_id(corpus, y)) for (x, y) in dataset_keys]

    start = time.time()
    print ('Loading cross-lingual paths')
    sys.stdout.flush()

    dataset_trans_en_temp = dataset_trans_en[:len(train_set_keys)] + [
        [(x, y)] for x, y in val_set_keys + test_set_keys]
    print dataset_trans_en_temp[:10]
    print dataset_trans_en_temp[-10:]
    xlingual_en_paths, xlingual_hi_paths = load_xlingual_paths_train(xlingual_path_file, keys, dataset_trans_en_temp)
    string_paths_x_e = []
    string_paths_x_h = []
    trans_lemmas = []
    for (x, y) in keys:
        curr_x_e_paths = {}
        curr_string_paths_x_e = []
        curr_trans_lemmas = []
        string_paths_x_e.append(curr_string_paths_x_e)
        # for fake_key in curr_x_e_paths.keys():
        #     for path in curr_x_e_paths[fake_key]:
        #         for edge in path.split('_'):
        #             en_vocabulary.add(edge.split('/')[0].lower())
        trans_lemmas.append(curr_trans_lemmas)
        curr_x_h_paths = {}
        curr_string_paths_x_h = []
        string_paths_x_h.append(curr_string_paths_x_h)

    print ("Done! Time taken = {0}".format(time.time() - start))

    trans_lemmas2 = []
    start = time.time()
    print ('Loading monolingual english paths')
    sys.stdout.flush()
    string_paths_e = []
    # Switch this for mono baseline
    # dataset_en_pairs = [ [(x,y)] for x,y in train_set_keys] + dataset_trans_en[(len(train_set_keys)):]
    dataset_en_pairs = [[(x, y)] for x, y in
                        train_set_keys + val_set_keys + test_set_keys]
    mono_en_paths = load_monolingual_paths(mono_en_path_file, keys,
                                           dataset_en_pairs)
    for (x, y) in keys:
        try:
            curr_e_paths = mono_en_paths[(x, y)]
        except KeyError:
            curr_e_paths = {}
        curr_string_paths = []
        curr_trans_lemmas = []
        for fake_key in curr_e_paths:
            curr_trans_lemmas.append(fake_key[1])
            en_vocabulary.add(fake_key[1])
            curr_string_paths.append(curr_e_paths[fake_key].items())
        string_paths_e.append(curr_string_paths)
        for fake_key in curr_e_paths.keys():
            for path in curr_e_paths[fake_key]:
                for edge in path.split('_'):
                    en_vocabulary.add(edge.split('/')[0].lower())

        trans_lemmas2.append(curr_trans_lemmas)

    trans_lemmas = trans_lemmas[:len(train_set_keys)] + [x[:1] for x in
                                                         trans_lemmas2[len(
                                                             train_set_keys):]]
    print (len(string_paths_e))
    print ("Done! Time taken = {0}".format(time.time() - start))

    # start = time.time()
    # print ('Loading monolingual hindi paths')
    # sys.stdout.flush()
    # string_paths_h = []
    # mono_hi_paths = load_monolingual_paths(mono_hi_path_file, keys,
    #                                        dataset_trans_hi)
    # # print mono_hi_paths
    # for (x, y) in keys:
    #     try:
    #         curr_h_paths = mono_hi_paths[(x, y)]
    #     except KeyError:
    #         curr_h_paths = {}
    #     curr_string_paths = []
    #     for fake_key in curr_h_paths:
    #         # print (curr_h_paths[fake_key].items())
    #         curr_string_paths.append(curr_h_paths[fake_key].items())
    #     string_paths_h.append(curr_string_paths)
    #     for fake_key in curr_h_paths.keys():
    #         for path in curr_h_paths[fake_key]:
    #             for edge in path.split('_'):
    #                 hi_vocabulary.add(edge.split('/')[0])
    # print ("Done! Time taken = {0}".format(time.time() - start))

    start = time.time()
    print ('Loading Hindi word embeddings')
    sys.stdout.flush()
    # word_vectors_hi, lemma_index_hi = load_embeddings(embeddings_file_hi,
    #                                                   hi_vocabulary)
    # word_inverted_index_hi = {i: w for w, i in lemma_index_hi.iteritems()}
    print ("Done! Time taken = {0}".format(time.time() - start))
    sys.stdout.flush()

    start = time.time()
    print ('Loading English word embeddings')
    sys.stdout.flush()
    word_vectors_en, lemma_index_en = load_embeddings(embeddings_file_en,
                                                      en_vocabulary)
    word_inverted_index_en = {i: w for w, i in lemma_index_en.iteritems()}
    print ("Done! Time taken = {0}".format(time.time() - start))
    sys.stdout.flush()

    # paths_x_to_y_xlingual_en = [{vectorize_path(path, lemma_index_en, pos_index, dep_index,
    #                                 dir_index): count
    #                  for path, count in curr_paths}
    #                 for curr_paths in string_paths_x_e]

    paths_x_to_y_mono_en = []
    for path_sets in string_paths_e:
        l = [{vectorize_path(path, lemma_index_en, pos_index, dep_index,
                             dir_index): count
              for path, count in curr_paths}
             for curr_paths in path_sets]
        paths_x_to_y_mono_en.append(l)

    paths_x_to_y_xlingual_en = [[{}]] * len( paths_x_to_y_mono_en)
    # for path_sets in string_paths_x_e:
    #     l = [{vectorize_path(path, lemma_index_en, pos_index, dep_index,
    #                          dir_index): count
    #           for path, count in curr_paths}
    #          for curr_paths in path_sets]
    #     paths_x_to_y_xlingual_en.append(l)
    # paths_xlingual_en = [{p: c for p, c in paths_x_to_y_xlingual_en[i].iteritems() if p is not None} for
    #          i in range(len(keys))]

    paths_x_to_y_xlingual_hi = [[{}]] * len(paths_x_to_y_mono_en)
    # for path_sets in string_paths_x_h:
    #     l = [{vectorize_path(path, lemma_index_hi, pos_index, dep_index,
    #                          dir_index): count
    #           for path, count in curr_paths}
    #          for curr_paths in path_sets]
    #     paths_x_to_y_xlingual_hi.append(l)



    paths_x_to_y_mono_hi = []
    # print len(string_paths_x_h)
    # for path_sets in string_paths_h:
    #     # print (path_sets)
    #     l = [{vectorize_path(path, lemma_index_hi, pos_index, dep_index,
    #                          dir_index): count
    #           for path, count in curr_paths}
    #          for curr_paths in path_sets]
    #     paths_x_to_y_mono_hi.append(l)

    # Get the word embeddings for x and y (get a lemma index)
    start = time.time()
    print ('Getting word vectors for the terms...')
    # unk_token =
    if use_gpu:
        x_y_vectors = torch.cuda.LongTensor(
            [(lemma_index_en.get(x, 0), lemma_index_en.get(y, 0)) for (x, y)
             in
             train_set_keys] + [
                (lemma_index_en.get(x, 0), lemma_index_en.get(y, 0)) for
                (x, y) in
                val_set_keys + test_set_keys])
        trans_lemmas_vectors = []
        # for curr_trans_lemmas in trans_lemmas[:len(train_set_keys)]:
        #     curr_vectors = torch.cuda.LongTensor(
        #         [lemma_index_hi.get(x, 0) for x in curr_trans_lemmas])
        #     trans_lemmas_vectors.append(curr_vectors)
        # for curr_trans_lemmas in trans_lemmas[len(train_set_keys):]:
        #     curr_vectors = torch.cuda.LongTensor(
        #         [lemma_index_en.get(x, 0) for x in curr_trans_lemmas])
        #     trans_lemmas_vectors.append(curr_vectors)


    else:
        x_y_vectors = torch.LongTensor(
            [(lemma_index_en.get(x, 0), lemma_index_en.get(y, 0)) for (x, y)
             in
             train_set_keys] + [
                (lemma_index_en.get(x, 0), lemma_index_en.get(y, 0)) for
                (x, y) in  val_set_keys + test_set_keys])
        trans_lemmas_vectors = []
        # for curr_trans_lemmas in trans_lemmas[:len(train_set_keys)]:
        #     curr_vectors = torch.LongTensor(
        #         [lemma_index_hi.get(x, 0) for x in curr_trans_lemmas])
        #     trans_lemmas_vectors.append(curr_vectors)
        # for curr_trans_lemmas in trans_lemmas[len(train_set_keys):]:
        #     curr_vectors = torch.LongTensor(
        #         [lemma_index_en.get(x, 0) for x in curr_trans_lemmas])
        #     trans_lemmas_vectors.append(curr_vectors)

    print ("Done! Time taken = {0}".format(time.time() - start))
    sys.stdout.flush()

    start = time.time()
    print ("Tensorizing paths")
    tensorized_paths_xlingual_en = [
        tensorize_paths(paths_x_to_y_xlingual_en[i], use_gpu) for i in
        range(len(paths_x_to_y_xlingual_en))]
    tensorized_paths_xlingual_hi = [
        tensorize_paths(paths_x_to_y_xlingual_hi[i], use_gpu) for i in
        range(len(paths_x_to_y_xlingual_hi))]

    tensorized_paths_mono_en = [
        tensorize_paths(paths_x_to_y_mono_en[i], use_gpu) for i in
        range(len(paths_x_to_y_mono_en))]
    tensorized_paths_mono_hi = [
        tensorize_paths(paths_x_to_y_mono_hi[i], use_gpu) for i in
        range(len(paths_x_to_y_mono_hi))]
    print ("Done! Time taken = {0}".format(time.time() - start))

    # print "{0} word pairs, {1} have no xlingual-en paths, {2} have no xlingual-hi paths." \
    #       " {3} have no xlingual paths, {4} have no mono-en paths, {5} have no mono-hi paths, {6} have no mono paths, {7} have no paths".\
    #     format(len(keys), len(empty_xlingual_en), len(empty_xlingual_hi), len(empty_xlingual_both), len(empty_mono_en),
    #            len(empty_mono_hi), len(empty_mono_both), len(empty_all))
    # sys.stdout.flush()

    pos_inverted_index = {i: p for p, i in pos_index.iteritems()}
    dep_inverted_index = {i: p for p, i in dep_index.iteritems()}
    dir_inverted_index = {i: p for p, i in dir_index.iteritems()}
    tensorized_paths_mono_hi = []
    word_vectors_hi = None
    word_inverted_index_hi = {}
    lemma_index_hi = {}
    giant_pickle_obj = [word_vectors_en, word_vectors_hi, x_y_vectors,
                        trans_lemmas_vectors, tensorized_paths_mono_en,
                        tensorized_paths_mono_hi,
                        tensorized_paths_xlingual_en,
                        tensorized_paths_xlingual_hi, \
                        lemma_index_en, lemma_index_hi, dict(pos_index),
                        dict(dep_index), dict(dir_index), \
                        word_inverted_index_en, word_inverted_index_hi,
                        pos_inverted_index, dep_inverted_index,
                        dir_inverted_index]

    pickle.dump(giant_pickle_obj,open("/fs/clip-scratch/yogarshi/data-en-en-transbaseline.pkl",'w'))

    return word_vectors_en, word_vectors_hi, x_y_vectors, trans_lemmas_vectors, tensorized_paths_mono_en, tensorized_paths_mono_hi, tensorized_paths_xlingual_en, tensorized_paths_xlingual_hi, \
           lemma_index_en, lemma_index_hi, pos_index, dep_index, dir_index, \
           word_inverted_index_en, word_inverted_index_hi, pos_inverted_index, dep_inverted_index, dir_inverted_index

def load_paths_and_word_vectors(dataset_keys, dataset_trans_en, dataset_trans_hi, mono_en_path_file, mono_hi_path_file,
                                xlingual_path_file, embeddings_file_en,embeddings_file_hi, use_gpu):
    '''
    Load the paths and the word vectors for this dataset
    :param corpus: the corpus object
    :param dataset_keys: the word pairs in the dataset
    :param word_index: the index of words for the word embeddings
    :return:
    '''

    # Define the dictionaries
    train_set_keys, val_set_keys, test_set_keys = dataset_keys
    dataset_keys = train_set_keys + val_set_keys+ test_set_keys

    pos_index = defaultdict(count(0).next)
    dep_index = defaultdict(count(0).next)
    dir_index = defaultdict(count(0).next)

    # Padding is 0
    _ = pos_index['#PAD#']
    _ = dep_index['#PAD#']
    _ = dir_index['#PAD#']

    # Unknowns are 1
    _ = pos_index['#UNKNOWN#']
    _ = dep_index['#UNKNOWN#']
    _ = dir_index['#UNKNOWN#']

    # Vectorize the paths
    keys = []
    en_vocabulary = set()
    hi_vocabulary = set()
    for (x,y) in train_set_keys :
        keys.append((x,y))
        en_vocabulary.add(x)
        en_vocabulary.add(y)
    for (x,y) in val_set_keys + test_set_keys:
        keys.append((x,y))
        en_vocabulary.add(x)
        hi_vocabulary.add(y)
    # keys = [(get_id(corpus, x), get_id(corpus, y)) for (x, y) in dataset_keys]

    start = time.time()
    print ('Loading cross-lingual paths')
    sys.stdout.flush()

    dataset_trans_en_temp = dataset_trans_en[:len(train_set_keys)] + [[(x,y)] for x,y in val_set_keys + test_set_keys]
    print dataset_trans_en_temp[:10]
    print dataset_trans_en_temp[-10:]
    xlingual_en_paths, xlingual_hi_paths = load_xlingual_paths_train(xlingual_path_file,keys,dataset_trans_en_temp )
    string_paths_x_e = []
    string_paths_x_h = []
    trans_lemmas = []
    for (x, y) in keys:
        try:
            curr_x_e_paths = xlingual_en_paths[(x,y)]
        except KeyError:
            curr_x_e_paths = {}
        curr_string_paths_x_e = []
        curr_trans_lemmas = []
        for fake_key in curr_x_e_paths:
            curr_trans_lemmas.append(fake_key[1])
            hi_vocabulary.add(fake_key[1])
            curr_string_paths_x_e.append(curr_x_e_paths[fake_key].items())
        string_paths_x_e.append(curr_string_paths_x_e)
        for fake_key in curr_x_e_paths.keys():
            for path in curr_x_e_paths[fake_key]:
                for edge in path.split('_'):
                    en_vocabulary.add(edge.split('/')[0].lower())

        trans_lemmas.append(curr_trans_lemmas)

        try:
            curr_x_h_paths = xlingual_hi_paths[(x, y)]
        except KeyError:
            curr_x_h_paths = {}
        curr_string_paths_x_h = []
        for fake_key in curr_x_h_paths:
            curr_string_paths_x_h.append(curr_x_h_paths[fake_key].items())
        string_paths_x_h.append(curr_string_paths_x_h)
        for fake_key in curr_x_h_paths.keys():
            for path in curr_x_h_paths[fake_key]:
                for edge in path.split('_'):
                    hi_vocabulary.add(edge.split('/')[0])


    print ("Done! Time taken = {0}".format(time.time()- start))


    trans_lemmas2 = []
    start = time.time()
    print ('Loading monolingual english paths')
    sys.stdout.flush()
    string_paths_e =[]
    # Switch this for mono baseline
    dataset_en_pairs = [ [(x,y)] for x,y in train_set_keys] + dataset_trans_en[(len(train_set_keys)):]
    # dataset_en_pairs = [ [(x,y)] for x,y in train_set_keys + val_set_keys + test_set_keys]
    mono_en_paths = load_monolingual_paths(mono_en_path_file, keys, dataset_en_pairs)
    for (x, y) in keys:
        try:
            curr_e_paths = mono_en_paths[(x,y)]
        except KeyError:
            curr_e_paths = {}
        curr_string_paths = []
        curr_trans_lemmas = []
        for fake_key in curr_e_paths:
            curr_trans_lemmas.append(fake_key[1])
            en_vocabulary.add(fake_key[1])
            curr_string_paths.append(curr_e_paths[fake_key].items())
        string_paths_e.append(curr_string_paths)
        for fake_key in curr_e_paths.keys():
            for path in curr_e_paths[fake_key]:
                for edge in path.split('_'):
                    en_vocabulary.add(edge.split('/')[0].lower())

        trans_lemmas2.append(curr_trans_lemmas)

    trans_lemmas = trans_lemmas[:len(train_set_keys)] + [ x[:1] for x in trans_lemmas2[len(train_set_keys):]]
    print (len(string_paths_e))
    print ("Done! Time taken = {0}".format(time.time() - start))

    # start = time.time()
    # print ('Loading monolingual hindi paths')
    # sys.stdout.flush()
    # string_paths_h = []
    # mono_hi_paths = load_monolingual_paths(mono_hi_path_file, keys,
    #                                        dataset_trans_hi)
    # # print mono_hi_paths
    # for (x, y) in keys:
    #     try:
    #         curr_h_paths = mono_hi_paths[(x, y)]
    #     except KeyError:
    #         curr_h_paths = {}
    #     curr_string_paths = []
    #     for fake_key in curr_h_paths:
    #         # print (curr_h_paths[fake_key].items())
    #         curr_string_paths.append(curr_h_paths[fake_key].items())
    #     string_paths_h.append(curr_string_paths)
    #     for fake_key in curr_h_paths.keys():
    #         for path in curr_h_paths[fake_key]:
    #             for edge in path.split('_'):
    #                 hi_vocabulary.add(edge.split('/')[0])
    # print ("Done! Time taken = {0}".format(time.time() - start))

    start = time.time()
    print ('Loading Hindi word embeddings')
    sys.stdout.flush()
    word_vectors_hi, lemma_index_hi = load_embeddings(embeddings_file_hi, hi_vocabulary)
    word_inverted_index_hi = {i: w for w, i in lemma_index_hi.iteritems()}
    print ("Done! Time taken = {0}".format(time.time() - start))
    sys.stdout.flush()

    start = time.time()
    print ('Loading English word embeddings')
    sys.stdout.flush()
    word_vectors_en, lemma_index_en = load_embeddings(embeddings_file_en, en_vocabulary)
    word_inverted_index_en = {i: w for w, i in lemma_index_en.iteritems()}
    print ("Done! Time taken = {0}".format(time.time() - start))
    sys.stdout.flush()

    # paths_x_to_y_xlingual_en = [{vectorize_path(path, lemma_index_en, pos_index, dep_index,
    #                                 dir_index): count
    #                  for path, count in curr_paths}
    #                 for curr_paths in string_paths_x_e]

    paths_x_to_y_xlingual_en = []
    for path_sets in string_paths_x_e:
        l = [{vectorize_path(path, lemma_index_en, pos_index, dep_index,
                             dir_index): count
              for path, count in curr_paths}
             for curr_paths in path_sets]
        paths_x_to_y_xlingual_en.append(l)
    # paths_xlingual_en = [{p: c for p, c in paths_x_to_y_xlingual_en[i].iteritems() if p is not None} for
    #          i in range(len(keys))]

    paths_x_to_y_xlingual_hi = []
    for path_sets in string_paths_x_h:
        l = [{vectorize_path(path, lemma_index_hi, pos_index, dep_index,
                             dir_index): count
              for path, count in curr_paths}
             for curr_paths in path_sets]
        paths_x_to_y_xlingual_hi.append(l)

    paths_x_to_y_mono_en = []
    for path_sets in string_paths_e:
        l = [{vectorize_path(path, lemma_index_en, pos_index, dep_index,
                             dir_index): count
              for path, count in curr_paths}
             for curr_paths in path_sets]
        paths_x_to_y_mono_en.append(l)

    paths_x_to_y_mono_hi = []
    # print len(string_paths_x_h)
    # for path_sets in string_paths_h:
    #     # print (path_sets)
    #     l = [{vectorize_path(path, lemma_index_hi, pos_index, dep_index,
    #                          dir_index): count
    #           for path, count in curr_paths}
    #          for curr_paths in path_sets]
    #     paths_x_to_y_mono_hi.append(l)

    # Get the word embeddings for x and y (get a lemma index)
    start = time.time()
    print ('Getting word vectors for the terms...')
    # unk_token =
    if use_gpu:
        x_y_vectors = torch.cuda.LongTensor(
            [(lemma_index_en.get(x, 0), lemma_index_en.get(y, 0)) for (x, y) in
             train_set_keys] + [(lemma_index_en.get(x, 0), lemma_index_hi.get(y, 0)) for (x, y) in
             val_set_keys + test_set_keys])
        trans_lemmas_vectors = []
        for curr_trans_lemmas in trans_lemmas[:len(train_set_keys)]:
            curr_vectors = torch.cuda.LongTensor([lemma_index_hi.get(x, 0) for x in curr_trans_lemmas])
            trans_lemmas_vectors.append(curr_vectors)
        for curr_trans_lemmas in trans_lemmas[len(train_set_keys):]:
            curr_vectors = torch.cuda.LongTensor([lemma_index_en.get(x, 0) for x in curr_trans_lemmas])
            trans_lemmas_vectors.append(curr_vectors)


    else:
        x_y_vectors = torch.LongTensor(
            [(lemma_index_en.get(x, 0), lemma_index_en.get(y, 0)) for (x, y) in
             train_set_keys] + [(lemma_index_en.get(x, 0), lemma_index_hi.get(y, 0)) for (x, y) in
             val_set_keys + test_set_keys])
        trans_lemmas_vectors = []
        for curr_trans_lemmas in trans_lemmas[:len(train_set_keys)]:
            curr_vectors = torch.LongTensor([lemma_index_hi.get(x, 0) for x in curr_trans_lemmas])
            trans_lemmas_vectors.append(curr_vectors)
        for curr_trans_lemmas in trans_lemmas[len(train_set_keys):]:
            curr_vectors = torch.LongTensor([lemma_index_en.get(x, 0) for x in curr_trans_lemmas])
            trans_lemmas_vectors.append(curr_vectors)


    print ("Done! Time taken = {0}".format(time.time() - start))
    sys.stdout.flush()

    start = time.time()
    print ("Tensorizing paths")
    tensorized_paths_xlingual_en = [tensorize_paths(paths_x_to_y_xlingual_en[i], use_gpu) for i in range(len(paths_x_to_y_xlingual_en))]
    tensorized_paths_xlingual_hi = [tensorize_paths(paths_x_to_y_xlingual_hi[i], use_gpu) for i in range(len(paths_x_to_y_xlingual_hi))]

    tensorized_paths_mono_en = [tensorize_paths(paths_x_to_y_mono_en[i], use_gpu) for i in range(len(paths_x_to_y_mono_en))]
    tensorized_paths_mono_hi = [tensorize_paths(paths_x_to_y_mono_hi[i], use_gpu) for i in range(len(paths_x_to_y_mono_hi))]
    print ("Done! Time taken = {0}".format(time.time() - start))

    empty_xlingual_en = set([i for i in range(len(tensorized_paths_xlingual_en)) if not tensorized_paths_xlingual_en[i]])
    empty_xlingual_hi = set([i for i in range(len(tensorized_paths_xlingual_hi)) if not tensorized_paths_xlingual_hi[i]])
    empty_xlingual_both = empty_xlingual_hi.intersection(empty_xlingual_en)

    # print "{0} word pairs, {1} have no xlingual-en paths, {2} have no xlingual-hi paths." \
    #       " {3} have no xlingual paths, {4} have no mono-en paths, {5} have no mono-hi paths, {6} have no mono paths, {7} have no paths".\
    #     format(len(keys), len(empty_xlingual_en), len(empty_xlingual_hi), len(empty_xlingual_both), len(empty_mono_en),
    #            len(empty_mono_hi), len(empty_mono_both), len(empty_all))
    # sys.stdout.flush()

    pos_inverted_index = {i: p for p, i in pos_index.iteritems()}
    dep_inverted_index = {i: p for p, i in dep_index.iteritems()}
    dir_inverted_index = {i: p for p, i in dir_index.iteritems()}
    tensorized_paths_mono_hi = []

    giant_pickle_obj = [word_vectors_en, word_vectors_hi, x_y_vectors, trans_lemmas_vectors,tensorized_paths_mono_en, tensorized_paths_mono_hi, tensorized_paths_xlingual_en, tensorized_paths_xlingual_hi, \
           lemma_index_en, lemma_index_hi, dict(pos_index), dict(dep_index), dict(dir_index), \
           word_inverted_index_en, word_inverted_index_hi, pos_inverted_index, dep_inverted_index, dir_inverted_index]

    pickle.dump(giant_pickle_obj,open("/fs/clip-scratch/yogarshi/data-hi-en-test.pkl".format(args.fold),'w'))

    return word_vectors_en, word_vectors_hi, x_y_vectors, trans_lemmas_vectors, tensorized_paths_mono_en, tensorized_paths_mono_hi, tensorized_paths_xlingual_en, tensorized_paths_xlingual_hi, \
           lemma_index_en, lemma_index_hi, pos_index, dep_index, dir_index, \
           word_inverted_index_en, word_inverted_index_hi, pos_inverted_index, dep_inverted_index, dir_inverted_index

if __name__ == '__main__':
    main()
