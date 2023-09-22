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

def make_id_selector_cluster_aware_intersect(indices, limits, clusters, cluster_limits):
    sp = faiss.swig_ptr
    return faiss.IDSelectorIVFClusterAwareIntersect(sp(indices), sp(limits), sp(clusters), sp(cluster_limits))

def make_id_selector_cluster_aware_direct(id_position_in_cluster, limits, clusters,  cluster_limits):
    sp = faiss.swig_ptr
    return faiss.IDSelectorIVFClusterAwareIntersectDirect(sp(id_position_in_cluster), sp(limits), sp(clusters), sp(cluster_limits))



def spot_check_filter(docs_per_word, index, indices, limits, clusters, cluster_limits):
    print('running spot check')
    inverted_lists = index.invlists
    from_id_to_map = dict()
    for i in range(inverted_lists.nlist):
        list_ids, _ = get_invlist(inverted_lists, i)
        for id in list_ids:
            from_id_to_map[id] = i

    indptr = docs_per_word.indptr

    ## lets' run some spot check
    for word in [0, 5000, 12124, 151123]:
    #for word in range(docs_per_word.shape[0]):
    #for word in [docs_per_word.shape[0]-1 ]:
        c_start = cluster_limits[word]
        c_end = cluster_limits[word + 1]
        assert c_end >= c_start

        start = indptr[word]
        end = indptr[word + 1]
        ids_in_word = {id for id in docs_per_word.indices[start:end]}

        cluster_base = -1
        for pos, cluster in enumerate(clusters[c_start: c_end]):
            if cluster_base == -1:
                cluster_base = cluster
            else:
                assert cluster != cluster_base
                cluster_base = cluster
            for id in indices[limits[c_start + pos]: limits[c_start + pos + 1]]:
                assert from_id_to_map[id] == cluster
                assert id in ids_in_word
                ids_in_word.remove(id)
        assert len(ids_in_word) == 0  # we should have covered all the ids in the word with the clusters

def find_max_interval(limits):

    out = -1
    for i in range(len(limits)-1):
        delta = limits[i+1] - limits[i]
        if delta > out:
            out = delta
    return out


def prepare_filter_by_cluster(docs_per_word, index):
    print('creating filter cluster')
    inverted_lists = index.invlists
    from_id_to_map = dict()
    from_id_to_pos = dict()
    for i in range(inverted_lists.nlist):
        list_ids, _ = get_invlist(inverted_lists, i)
        for pos, id in enumerate(list_ids):
            from_id_to_map[id] = i
            from_id_to_pos[id] = pos
    print('loaded the mapping with {} entries'.format(len(from_id_to_map)))

    ## reorganize the docs per word
    #
    cluster_limits = [0]
    clusters = list()
    limits = list()
    id_position_in_cluster = list()

    indices = np.array(docs_per_word.indices)
    indptr = docs_per_word.indptr
    for word in range(docs_per_word.shape[0]):
        start = indptr[word]
        end = indptr[word + 1]
        if word % 100 == 0:
            print('processed {} words'.format(word))
        array_ind_cluster = [(id, from_id_to_map[id]) for id in indices[start:end]]
        array_ind_cluster.sort(key=lambda x: x[1])

        if len(array_ind_cluster) == 0:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!there is am empty word!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
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
            id_position_in_cluster.append(from_id_to_pos[id])

        clusters.extend(local_clusters)
        limits.extend(local_limits)
        new_cluster_limit = len(local_clusters) + cluster_limits[-1]
        cluster_limits.append( new_cluster_limit)
    limits.append(len(indices))

    clusters = np.array(clusters, dtype=np.int16)
    limits = np.array(limits, dtype=np.int32)
    cluster_limits = np.array(cluster_limits, dtype=np.int32)
    id_position_in_cluster = np.array(id_position_in_cluster, dtype=np.int32)

    return indices, limits, clusters, cluster_limits, id_position_in_cluster


class FAISS(BaseFilterANN):

    def __init__(self,  metric, index_params):
        self._index_params = index_params
        self._metric = metric
        print(index_params)
        self.train_size = index_params.get('train_size', None)
        self.indexkey = index_params.get("indexkey", "IVF32768,SQ8")
        self.metadata_threshold = 1e-3
        self.nt = index_params.get("threads", 1)
        self.type = index_params.get("type", "intersect")
    

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

            self.indices, self.limits, self.clusters, self.cluster_limits, self.id_position_in_cluster = prepare_filter_by_cluster(self.docs_per_word, self.index)
            print('dumping cluster map')
            pickle.dump((self.indices, self.limits, self.clusters, self.cluster_limits, self.id_position_in_cluster), open(self.cluster_sig_name(dataset), "wb"), -1)
            spot_check_filter(self.docs_per_word, self.index, self.indices, self.limits, self.clusters,
                              self.cluster_limits)
    
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
                self.indices, self.limits, self.clusters, self.cluster_limits, self.id_position_in_cluster = pickle.load(open(self.cluster_sig_name(dataset), "rb"))
            else:
                print('cluster file not found')
                self.indices, self.limits, self.clusters, self.cluster_limits, self.id_position_in_cluster = prepare_filter_by_cluster(self.docs_per_word, self.index)
                pickle.dump((self.indices, self.limits, self.clusters, self.cluster_limits, self.id_position_in_cluster), open(self.cluster_sig_name(dataset), "wb"), -1)

            spot_check_filter(self.docs_per_word, self.index, self.indices, self.limits, self.clusters, self.cluster_limits)

        self.ps = faiss.ParameterSpace()
        self.ps.initialize(self.index)

        max_range = find_max_interval(self.limits)
        print('the max range is {}'.format(max_range))

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
            #if False:
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
                if self.type == 'simple':
                    sel = make_id_selector_ivf_two(self.docs_per_word)
                elif self.type == "aware":
                    sel = make_id_selector_cluster_aware(self.indices, self.limits, self.clusters, self.cluster_limits)
                elif self.type == 'intersect':
                    sel = make_id_selector_cluster_aware_intersect(self.indices, self.limits, self.clusters, self.cluster_limits)
                elif self.type == 'direct':
                    sel = make_id_selector_cluster_aware_direct(self.id_position_in_cluster, self.limits, self.clusters,
                                                                   self.cluster_limits)
                else:
                    raise Exception('unknown type ', self.type)
                sel.set_words(int(w1), int(w2))


                if hasattr(self, 'k_factor') and self.k_factor > 0:
                    params = faiss.SearchParametersIVF(sel=sel, nprobe=self.nprobe, k_factor=self.k_factor, max_codes=self.max_codes)
                else:
                    params = faiss.SearchParametersIVF(sel=sel, nprobe=self.nprobe, max_codes=self.max_codes)

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
        if "max_codes" in query_args:
            self.max_codes = query_args['max_codes']
            self.ps.set_index_parameters(self.index, f"max_codes={query_args['max_codes']}")
        else:
            self.max_codes = 0
        if "k_factor" in query_args:
            self.k_factor = query_args['k_factor']
            self.qas = query_args

        if "mt_threshold" in query_args:
            self.metadata_threshold = query_args['mt_threshold']
        else:
            self.metadata_threshold = 1e-3

    def __str__(self):
        return f'Faiss({self.indexkey,self.type, self.qas})'

   