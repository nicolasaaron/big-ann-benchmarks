import pdb
import pickle
import numpy as np
import os

from multiprocessing.pool import ThreadPool

import faiss
from faiss.contrib.inspect_tools import get_invlist
from neurips23.filter.base import BaseFilterANN
from benchmark.datasets import DATASETS
from benchmark.dataset_io import download_accelerated

import bow_id_selector

def csr_get_row_indices(m, i):
    """ get the non-0 column indices for row i in matrix m """
    return m.indices[m.indptr[i] : m.indptr[i + 1]]

def make_id_selector_ivf_two(docs_per_word):
    sp = faiss.swig_ptr
    return faiss.IDSelectorIVFTwo(sp(docs_per_word.indices), sp(docs_per_word.indptr))

def make_id_selector_cluster_aware(indices, limits, clusters, cluster_limits):
    sp = faiss.swig_ptr
    return faiss.IDSelectorIVFClusterAware(sp(indices), sp(limits), sp(clusters), sp(cluster_limits))

def prepare_filter_by_cluster(docs_per_word, index):
    print('creating filter cluster')
    inverted_lists = index.invlists
    from_id_to_map = dict()
    for i in range(inverted_lists.nlist):
        list_ids, _ = get_invlist(inverted_lists, i)
        for id in list_ids:
            from_id_to_map[id] = i
    print('loaded the mapping with {} entries'.format(len(from_id_to_map)))

    ## reorganize the docs per word
    #
    cluster_limits = [0]
    clusters = list()
    limits = list()

    indices = np.array(docs_per_word.indices)
    indptr = docs_per_word.indptr
    for word in range(docs_per_word.shape[0]):
        start = indptr[word]
        end = indptr[word + 1]
        if word % 100 == 0:
            print('processed {} words'.format(word))
        array_ind_cluster = [(id, from_id_to_map[id]) for id in indices[start:end]]
        array_ind_cluster.sort(key=lambda x: x[1])

        local_clusters = []
        local_limits = []
        current_cluster = -1
        for pos, arr in enumerate(array_ind_cluster):
            id, cluster = arr
            if current_cluster == -1 or cluster != current_cluster:
                current_cluster = cluster
                local_clusters.append(cluster)
                local_limits.append(start + pos)
            indices[start + pos] = id

        clusters.extend(local_clusters)
        limits.extend(local_limits)
        cluster_limits.append(len(local_clusters))
    limits.append(len(indices))

    clusters = np.array(clusters, dtype=np.int16)
    limits = np.array(limits, dtype=np.int32)
    cluster_limits = np.array(cluster_limits, dtype=np.int32)

    return indices, limits, clusters, cluster_limits


class FAISS(BaseFilterANN):

    def __init__(self,  metric, index_params):
        self._index_params = index_params
        self._metric = metric
        print(index_params)
        self.train_size = index_params.get('train_size', None)
        self.indexkey = index_params.get("indexkey", "IVF32768,SQ8")
        self.metadata_threshold = 1e-3
        self.nt = index_params.get("threads", 1)
    

    def fit(self, dataset):
        faiss.omp_set_num_threads(self.nt)
        ds = DATASETS[dataset]()

        print('the size of the index', ds.d)
        index = faiss.index_factory(ds.d, self.indexkey)
        xb = ds.get_dataset()

        print("train")
        print('train_size', self.train_size)
        if self.train_size is not None:
            x_train = xb[:self.train_size]
        else:
            x_train = xb
        index.train(x_train)
        print("populate")

        bs = 1024
        for i0 in range(0, ds.nb, bs):
            index.add(xb[i0: i0 + bs])


        print('ids added')
        self.index = index
        self.nb = ds.nb
        self.xb = xb
        self.ps = faiss.ParameterSpace()
        self.ps.initialize(self.index)
        print("store", self.index_name(dataset))
        faiss.write_index(index, self.index_name(dataset))

        if ds.search_type() == "knn_filtered":
            words_per_doc = ds.get_dataset_metadata()
            words_per_doc.sort_indices()
            self.docs_per_word = words_per_doc.T.tocsr()
            self.docs_per_word.sort_indices()
            self.ndoc_per_word = self.docs_per_word.indptr[1:] - self.docs_per_word.indptr[:-1]
            self.freq_per_word = self.ndoc_per_word / self.nb
            del words_per_doc

            self.indices, self.limits, self.clusters, self.cluster_limits = prepare_filter_by_cluster(self.docs_per_word, self.index)
            print('dumping cluster map')
            pickle.dump((self.indices, self.limits, self.clusters, self.cluster_limits), open(self.cluster_sig_name(dataset), "wb"), -1)

    
    def index_name(self, name):
        return f"data/{name}.{self.indexkey}_wm.faissindex"


    def cluster_sig_name(self, name):
        return f"data/{name}.{self.indexkey}_cluster_wm.pickle"





    def load_index(self, dataset):
        """
        Load the index for dataset. Returns False if index
        is not available, True otherwise.

        Checking the index usually involves the dataset name
        and the index build paramters passed during construction.
        """
        if not os.path.exists(self.index_name(dataset)):
            if 'url' not in self._index_params:
                return False

            print('Downloading index in background. This can take a while.')
            download_accelerated(self._index_params['url'], self.index_name(dataset), quiet=True)

        print("Loading index")
        ds = DATASETS[dataset]()
        self.nb = ds.nb
        self.xb = ds.get_dataset()

        if ds.search_type() == "knn_filtered":
            words_per_doc = ds.get_dataset_metadata()
            words_per_doc.sort_indices()
            self.docs_per_word = words_per_doc.T.tocsr()
            self.docs_per_word.sort_indices()
            self.ndoc_per_word = self.docs_per_word.indptr[1:] - self.docs_per_word.indptr[:-1]
            self.freq_per_word = self.ndoc_per_word / self.nb
            del words_per_doc

        self.index = faiss.read_index(self.index_name(dataset))

        if ds.search_type() == "knn_filtered":
            if  os.path.isfile( self.cluster_sig_name(dataset)):
                print('loading cluster file')
                self.indices, self.limits, self.clusters, self.cluster_limits = pickle.load(open(self.cluster_sig_name(dataset), "rb"))
            else:
                print('cluster file not found')
                self.indices, self.limits, self.clusters, self.cluster_limits = prepare_filter_by_cluster(self.docs_per_word, self.index)
                pickle.dump((self.indices, self.limits, self.clusters, self.cluster_limits), open(self.cluster_sig_name(dataset), "wb"), -1)

        self.ps = faiss.ParameterSpace()
        self.ps.initialize(self.index)

        return True

    def index_files_to_store(self, dataset):
        """
        Specify a triplet with the local directory path of index files,
        the common prefix name of index component(s) and a list of
        index components that need to be uploaded to (after build)
        or downloaded from (for search) cloud storage.

        For local directory path under docker environment, please use
        a directory under
        data/indices/track(T1 or T2)/algo.__str__()/DATASETS[dataset]().short_name()
        """
        raise NotImplementedError()
    
    def query(self, X, k):
        nq = X.shape[0]
        self.I = -np.ones((nq, k), dtype='int32')        
        bs = 1024

        try:
            print('k_factor', self.index.k_factor)
            self.index.k_factor = self.k_factor
        except:
            pass
        for i0 in range(0, nq, bs):
            _, self.I[i0:i0+bs] = self.index.search(X[i0:i0+bs], k)

    
    def filtered_query(self, X, filter, k):
        print('running filtered query')
        nq = X.shape[0]
        self.I = -np.ones((nq, k), dtype='int32')

        meta_q = filter
        docs_per_word = self.docs_per_word
        ndoc_per_word = self.ndoc_per_word
        freq_per_word = self.freq_per_word
        
        def process_one_row(q):
            faiss.omp_set_num_threads(1)
            qwords = csr_get_row_indices(meta_q, q)
            assert qwords.size in (1, 2)
            w1 = qwords[0]
            freq = freq_per_word[w1]
            if qwords.size == 2:
                w2 = qwords[1]
                freq *= freq_per_word[w2]
            else:
                w2 = -1

            if freq < self.metadata_threshold:
                # metadata first
                docs = csr_get_row_indices(docs_per_word, w1)
                if w2 != -1:
                    docs = bow_id_selector.intersect_sorted(
                        docs, csr_get_row_indices(docs_per_word, w2))

                assert len(docs) >= k#, pdb.set_trace()
                xb_subset = self.xb[docs]
                _, Ii = faiss.knn(X[q : q + 1], xb_subset, k=k)
 
                self.I[q, :] = docs[Ii.ravel()]
            else:
                # IVF first, filtered search
                #sel = make_id_selector_ivf_two(self.docs_per_word)
                sel = make_id_selector_cluster_aware(self.indices, self.limits, self.clusters, self.cluster_limits)
                sel.set_words(int(w1), int(w2))


                if hasattr(self, 'k_factor') and self.k_factor > 0:
                    params = faiss.SearchParametersIVF(sel=sel, nprobe=self.nprobe, k_factor=self.k_factor)
                else:
                    params = faiss.SearchParametersIVF(sel=sel, nprobe=self.nprobe)

                _, Ii = self.index.search( X[q:q+1], k, params=params )
                Ii = Ii.ravel()
                self.I[q] = Ii


        if self.nt <= 1:
        #if True:
            for q in range(nq):
                process_one_row(q)

        else:
            faiss.omp_set_num_threads(self.nt)
            pool = ThreadPool(self.nt)
            list(pool.map(process_one_row, range(nq)))

    def get_results(self):
        return self.I

    def set_query_arguments(self, query_args):
        faiss.cvar.indexIVF_stats.reset()
        if "nprobe" in query_args:
            self.nprobe = query_args['nprobe']
            self.ps.set_index_parameters(self.index, f"nprobe={query_args['nprobe']}")
            self.qas = query_args
        else:
            self.nprobe = 1
        if "k_factor" in query_args:
            self.k_factor = query_args['k_factor']
            self.qas = query_args

        if "mt_threshold" in query_args:
            self.metadata_threshold = query_args['mt_threshold']
        else:
            self.metadata_threshold = 1e-3

    def __str__(self):
        return f'Faiss({self.indexkey, self.qas})'

   