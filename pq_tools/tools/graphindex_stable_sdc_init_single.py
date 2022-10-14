#!/usr/bin/env python
# -*- coding: gbk -*-
import nanopq
import time
import numpy as np
import sys
import os
from log import logger
from scipy.cluster.vq import vq, kmeans2
import itertools
from operator import itemgetter
import multiprocessing as mp
import pickle
import json
from multiprocessing import Process
from multiprocessing import Lock
from multiprocessing import Manager
from multiprocessing import Value
from multiprocessing import Queue

import base64
import json
from tqdm import tqdm


def dump_file_list(file_list):
    with open('.read_locker', 'w') as f:
        json.dump(file_list, f)


def pop_file(lock):
    lock.acquire()
    file_list = []
    with open('.read_locker') as f:
        file_list = json.load(f)
    if (len(file_list) == 0):
        lock.release()
        return None, 0
    file = file_list[0]
    del file_list[0]
    with open('.read_locker', 'w') as f:
        json.dump(file_list, f)
    lock.release()
    return file, len(file_list)


def read_one_file(fp_path, fp, cnt, emb_list, terms):
    with open(os.path.join(fp_path, fp), 'r', encoding="utf-8") as f:
        for i in f:
            try:
                lines = i.strip().split('\t')
                if len(lines) != 2:
                    continue
                term = lines[0]
                emb = lines[1].split(" ")[:63]
                terms.append(term)
                emb_list.append(np.array(emb, dtype='float32'))
            except:
                continue
            cnt += 1
            if cnt % 10000 == 0:
                logger.info("read %d emb" % cnt)
    return cnt


def read_func(tid, fp_path, lock, emb_list_map, terms_map, run_task_num, queue):
    logger.info("begin __read_func, t[%d]", tid)
    cnt = 0
    terms = []
    emb_list = []

    while (1):
        file, remain = pop_file(lock)
        if not file:
            break
 
        logger.info("reading file[%s], remain %s..., t[%d]", file, remain, tid)
        start = time.time()
        run_task_num.value += 1
        cnt = read_one_file(fp_path, file, cnt, emb_list, terms)
        run_task_num.value -= 1
        logger.info("read file[%s] finished, used_time[%lus], running task %d, t[%d]",
                file, time.time()-start, run_task_num.value, tid)

    start = time.time()
    emb_list_map[tid] = emb_list
    terms_map[tid] = terms

    logger.info("end __read_func, t[%d], fill_map_time[%lus]", tid, time.time()-start)

    queue.put(tid)


class PQINDEX(object):

    def __init__(self):
        pass

    @staticmethod
    def read_term_emb_by_fp(fp_path, emb_size):
        terms = []
        emb_list = []
        cnt = 0
        for fp in os.listdir(fp_path):
            logger.info('reading file[' + fp_path + '/' + fp + ']')
            with open(os.path.join(fp_path, fp), 'r', encoding="utf-8") as f:
                for i in f:
                    try:

                        lines = i.strip().split('\t')
                        if len(lines) != 2:
                            continue
                        term = lines[0]
                        emb = lines[1].split(' ')[:emb_size]
                        terms.append(term)
                        emb_list.append(np.array(emb, dtype='float32'))
                    except:
                        continue
                    cnt += 1
                    if cnt % 100000 == 0:
                        logger.info("read %d emb" % cnt)
        logger.info('''read {} pid's embs'''.format(cnt))
        return terms, emb_list
    
    @staticmethod
    def read_term_emb_by_file(fname, emb_size):
        terms = []
        emb_list = []
        cnt = 0
        logger.info('reading file ' + fname)
        with open(fname, 'r', encoding="utf-8") as f:
            for i in f:
                try:

                    lines = i.strip().split('\t')
                    if len(lines) != 2:
                        continue
                    term = lines[0]
                    emb = lines[1].split('|')[:emb_size]
                    terms.append(term)
                    emb_list.append(np.array(emb, dtype='float32'))
                except:
                    continue
                cnt += 1
                if cnt % 100000 == 0:
                    logger.info("read %d emb" % cnt)
        logger.info('''read {} pid's embs'''.format(cnt))
        return terms, emb_list
    
    @staticmethod
    def read_term_emb_by_json(fname, emb_size):
        terms = []
        emb_list = []
        cnt = 0
        logger.info('reading file ' + fname)
        with open(fname, 'r', encoding="utf-8") as f:
            lines=f.readlines()
            for line in tqdm(lines):
                line_json=json.loads(line)
                term=line_json['text_id']
                emb = line_json['emb'].split('|')[:emb_size]
                terms.append(str(term))
                emb_list.append(np.array(emb, dtype='float32'))
                cnt += 1
                if cnt % 10000 == 0:
                    logger.info("read %d emb" % cnt)
        logger.info('''read {} pid's embs'''.format(cnt))
        return terms, emb_list

    @staticmethod
    def read_term_emb_by_fp_parallel(fp_path, threads_num=1):
        terms = []
        emb_list = []
        cnt = 0
        fp_list = os.listdir(fp_path)
        dump_file_list(fp_list)

        process_list = []
        manager = Manager()
        lock = manager.Lock()
        emb_list_map = manager.dict()
        terms_map = manager.dict()
        queue = Queue()
        run_task_num = Value('i', 0)

        for i in range(0, threads_num):
            t = Process(target=read_func,
                    args=(i, fp_path, lock, emb_list_map, terms_map, run_task_num,
                        queue))
            t.start()
            process_list.append(t)
 
        terms = []
        emb_list = []
        joined_thrd_num  = 0
        while joined_thrd_num < threads_num:
            i = queue.get()
            p = process_list[i]
            logger.info('joining tid[%d]' % i)
            p.join()
            start = time.time()
            terms.extend(terms_map[i])
            emb_list.extend(emb_list_map[i])
            logger.info('tid[%d] extend time[%lus] joined_num[%d]' % (i, time.time()-start,
                joined_thrd_num + 1))
            joined_thrd_num = joined_thrd_num + 1

        return terms, emb_list

    @staticmethod
    def _read_pqindex(fp):
        pid2path = {}
        paths = []
        id2pid = []
        with open(fp, 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                pid = line[0]
                id2pid.append(pid) 
                path = line[1].split(' ')
                if len(path) != M:
                    continue
                pid2path[pid] = [int(node) for node in path]
                paths.append([int(node) for node in path])
        return np.array(paths,dtype='int64'), pid2path, id2pid
    

    @staticmethod
    def _read_codeword_emb(fp, M, Ks):
        codewords = []
        with open(fp, 'r') as f:
            for i in f:
                code = i.strip().split('\t')
                code = list(map(float, code))
                codewords.append(code)
        codewords = np.array(codewords)
        codewords = np.reshape(codewords, [M, Ks, -1])
        return codewords
        
    @staticmethod
    def _read_Rmatrix_emb(fp):
        R = []
        with open(fp, 'r') as f:
            for i in f:
                r_emb = i.strip().split(' ')
                R.append(map(float, r_emb))
        R = np.array(R)
        return R
 
    def prepare_opqindex_by_file(self, input_fp, output_fp, codewords_fp, M=8, Ks=20, emb_size=60):
        """
        prepare_opqindex_by_file
        """
        # N, Nt, D = 10000, 2000, 128
        logger.info('begin read file[' + input_fp + ']')
        start = time.time()
        self.id2pid, emb_list = self.read_term_emb_by_json(input_fp, emb_size)
        logger.info('read file time {}'.format(time.time()-start))
        self.emb_list = np.array(emb_list).astype(np.float32)
        logger.info('emb_list.shape:' + str(self.emb_list.shape))
        # Instantiate with M sub-spaces
        #M=6,Ks=256
        pq = nanopq.OPQ(M=M, Ks=Ks)

        # Train codewords
        logger.info('begin fit model')
        start = time.time()
        pq.fit(self.emb_list)
        logger.info('fit model time {}'.format(time.time()-start))

        # Encode to PQ-codes
        logger.info('begin encode')
        start = time.time()
        code = pq.encode(self.emb_list)
        logger.info('encode item time {}'.format(time.time()-start))

        with open(output_fp, 'w', encoding="utf-8") as f:
            for idx, v in enumerate(zip(self.id2pid, code)):
                term = v[0]
                code_i = map(str, v[1])
                f.write('\t'.join([term, ' '.join(code_i)]))
                f.write('\n')
        with open(codewords_fp, 'w', encoding="utf-8") as f:
            for i in range(M):
                for j in range(Ks):
                    codes = pq.codewords[i, j, :]
                    tmp = map(str, codes)
                    f.write('\t'.join(tmp))
                    f.write('\n')
        with open(codewords_fp + '_RMatrix', 'w', encoding="utf-8") as f:
            for i in range(pq.R.shape[0]):
                r_emb = ' '.join([str(i) for i in pq.R[i]])
                f.write(r_emb)
                f.write('\n')
        logger.info('index done cost time {}'.format(time.time()-start))
    
    @staticmethod
    def _get_unique_pqindex_string(pqindex):
        tmp = [''.join(map(str,i)) for i in pqindex]
        tmp = set(tmp)
        return tmp


    def _adjust(self, process_id, queue):
        M, Ks, _ = self.dists.shape
        processed = 0
        catch_time = 0
        processed_res = []
        d = None
        while True:
            if queue.qsize() % 100 == 0 and queue.qsize() != 0:
                logger.info("queue size: {0}, timeout: {1}".format(queue.qsize(), self.timeout))
            t0 = time.time()
            for _ in range(5):
                try:
                    paths, d = queue.get(timeout=self.timeout)
                except:
                    paths = None
                if d == 0:
                    paths = dict(enumerate(self.paths[:,0]))
                if paths is not None and d is not None:
                    break
            t1 = time.time()
            #if paths is not None:
            #    logger.info("layer: {0}, get{1} pids".format(d+1, len(paths)))
            #else:
            if paths is None:
                if processed > 0 and catch_time >= 10:
                    logger.info("Process {0} exits".format(os.getpid()))
                    break
                else:
                    logger.info("Got empty job, pid: {0}, time: {1}".format(
                        os.getpid(), catch_time))
                    catch_time += 1
                    continue
            processed += 1
            catch_time = 0
            tstart = time.time()

            max_num = int(pow(Ks, M-1-d))
            #max_num = N // pow(Ks, d+1) + 1

            assert len(paths) <= max_num * Ks

            balanced_nodes, node_pidlist = set(), {}

            #logger.info("get path from queue")
            #get dist list of each node in current layer
            for id, node in paths.items():
                dist = self.dists[d][node]
                if node in node_pidlist:
                    node_pidlist[node].append([id, dist[node]])
                else:
                    node_pidlist[node] = [[id, dist[node]]]
            t1 = time.time()
            #logger.info("get dist list of each node in layer {0}: {1}".format(d+1, t1-tstart))

            try:
                for _ in range(Ks):
                    node_list_length = [[node, len(node_pidlist[node])] for node in node_pidlist]
                    node_list_length.sort(key=itemgetter(1), reverse=True)
                    top_node, pid_num = node_list_length[0]
                    pid_list = node_pidlist[top_node]
                    diff_num = len(pid_list) - max_num
                    if diff_num <= 0:
                        break

                    #logger.info('layer {}: node {} now num {}, max num {}, exceed num {}'.format(d+1, top_node, pid_num, max_num, diff_num))
                    balanced_nodes.add(top_node)
                    #logger.info("balanced_node {}".format(balanced_nodes))

                    #logger.info("pid list sort, %d" % len(pid_list))
                    #logger.info(pid_list)
                    pid_list.sort(key=lambda x:x[1])
                    #pid_list.sort(key=lambda x:x[0])
                    #logger.info("pidlist sort done")
                    while len(pid_list) > max_num:
                        tmp_id, _ = pid_list.pop()
                        new_node = None
                        for node in self.dists_argsort[d][top_node]:
                            if node not in balanced_nodes:
                                new_node = node
                                break
                        assert new_node is not None
                        assert new_node < Ks
                        if new_node in node_pidlist:
                            node_pidlist[new_node].append([tmp_id, self.dists[d][top_node][new_node]])
                        else:
                            node_pidlist[new_node] = [[tmp_id, self.dists[d][top_node][new_node]]]
                        paths[tmp_id] = new_node
                #logger.info("balance done: {0}, layer {1}, handled {2} pids".format(time.time() - t1, d+1, len(paths)))

                with open(self.fp_output_rebuild + '_' + str(process_id), 'a') as fout:
                    for id, node in paths.items(): 
                        fout.write('\t'.join([str(d), str(id), str(node)]) + '\n')
                    fout.flush()

                for node, pidlist in node_pidlist.items():
                    assert len(pidlist) <= max_num
                    if d+1 < M:
                        node_path = dict([[pid, self.paths[pid][d+1]] for pid, _ in pidlist])
                        queue.put((node_path, d+1))

                self.timeout = int(0.4 * self.timeout + 0.6 * (time.time() - tstart))
                if self.timeout < 5:
                    self.timeout = 5
            except Exception as e:
                logger.info("erro {}".format(e))
                logger.error(traceback.print_exc())

        logger.info("Process {0} process {1} times".format(os.getpid(), len(processed_res)))
    

    def _read_train_file(self, train_path, query_model_logits=None):
        data = {}
        pid2id = {}
        files = [os.path.join(train_path, file) for file in os.listdir(train_path)]
        cnt = 0
        for file in files:
            with open(file, 'r') as fin:
                for line in fin:
                    arr = line.strip().split('\t')
                    if len(arr) != 3:
                        continue
                    query = arr[0]
                    pid = arr[1]
                    cnt += 1
                    if cnt % 1000000 == 0:
                        logger.info("read %d query pid pair" % cnt)
                    if query_model_logits is not None and query not in query_model_logits:
                        continue
                    if query in data:
                        data[query].add(pid)
                    else:
                        data[query] = set([pid])
                    if pid not in pid2id:
                        pid2id[pid] = len(pid2id)
        return pid2id, data

    def _read_query_predict_logits(self, predict_logits_file):
        data = {}
        cnt = 0
        with open(predict_logits_file, 'r') as fin:
            for line in fin:
                arr = line.strip().split('\t')
                if len(arr) != 2:
                    continue
                cnt += 1
                query, logits_b64_npstr = arr
                logits = np.fromstring(base64.b64decode(logits_b64_npstr), dtype=np.float32) #M,Ks
                data[query] = logits
        return data

    def _save_res(self, parallel, N, M):
        out_paths = np.empty([N, M], dtype='int64')
        for process_id in range(parallel):
            if not os.path.exists(self.fp_output_rebuild + '_' + str(process_id)):
                logger.info("check || %s not exists" % (self.fp_output_rebuild + '_' + str(process_id)))
                continue
            logger.info("final || read from %s" % (self.fp_output_rebuild + '_' + str(process_id)))
            with open(self.fp_output_rebuild + '_' + str(process_id), 'r') as f:
                #logger.info("done", d, len(path))
                for line in f:
                    arr = line.strip().split('\t')
                    assert(len(arr)==3)
                    d, id, node = map(int, arr)
                    out_paths[id][d] = node

        with open(self.fp_output_rebuild, 'w') as f:
            for id in range(N):
                pid = self.id2pid[id]
                path = map(str, list(out_paths[id]))
                f.write('\t'.join([pid, ' '.join(path)]) + '\n')
            f.flush()
        logger.info('set of pqindex: ' + str(len(self._get_unique_pqindex_string(out_paths))))

    def compute_dtable(self, emb):
        """
        input: M, Ks, d
        output: M, Ks, Ks
        """
        emb_T = np.transpose(emb, (0, 2, 1))
        a2 = np.sum(np.power(emb, 2), axis=-1, keepdims=True) #M, Ks, 1
        b2 = np.transpose(a2, (0, 2, 1)) #M, 1, Ks
        ab = np.matmul(emb, emb_T)
        return a2 + b2 - 2*ab 

    def jtm_from_file(self, input_fp, output_fp, index_fp, codewords_fp, Rmatrix_fp, M=8, Ks=5, only_save=False):
        start = time.time()
        self.fp_output_rebuild = output_fp
        #logger.info(self.emb_list.shape)
        #N, D = self.emb_list.shape
        #Ds = int(D / M)
        
        ## step1-prepare data: 1)id2path 2)distance matrix 
        logger.info("read codewordemb and Rmatrix") 
        self.paths, pid2path, self.id2pid = self._read_pqindex(index_fp)
        N = len(self.id2pid)
        self.parall = 30
        self.timeout = 5
        if not only_save:
            codeword_embs = self._read_codeword_emb(codewords_fp, M, Ks) #M,ks,32
            #R_matrix = self._read_Rmatrix_emb(Rmatrix_fp)
            #logger.info("rotationing") 
            #self.emb_list = np.matmul(self.emb_list, R_matrix) 
            logger.info("dtable...") 
            self.dists = self.compute_dtable(codeword_embs) #M,Ks,Ks
            logger.info("dtable done") 
            self.dists_argsort = np.argsort(self.dists, axis=2) #M,Ks,Ks
            logger.info("dtable done argsort done") 

            logger.info("dists expample" + str(self.dists[0, :, :]))
            logger.info("path example" + str(self.paths[0]))
            logger.info('set of path' + str(len(self._get_unique_pqindex_string(self.paths))))

            #paths_0 = dict(enumerate(self.paths[:,0]))
            paths_0 = None
            #del self.emb_list
            logger.info("path0 done1") 

            queue = mp.Queue()
            queue.put((paths_0, 0))
            logger.info("put path0 into queue") 
            logger.info("start adjust") 
            processes = []
            for i in range(self.parall):
                p = mp.Process(target=self._adjust, args=(i, queue))
                processes.append(p)
                p.start()

            for p in processes:
                p.join()
            assert(queue.empty())

        self._save_res(self.parall, N, M)
        
        logger.info('adjust complet cost time {}'.format(time.time()-start))

if __name__ == '__main__':
    pqdm = PQINDEX()
    # logger.basicConfig(level=logger.INFO)

    first_init = sys.argv[1] == "1"
    Ks = int(sys.argv[2])
    M = int(sys.argv[3])

    if first_init:
        pid_emb_path = sys.argv[4]
        task_name = sys.argv[5]
        emb_size = int(sys.argv[6])
        output_path=sys.argv[7]
        fp_output = output_path+'pid2path.txt' + '_' + task_name
        fp_output_unique = output_path+'em_pid2path.txt_r0' + '_' + task_name
        fp_codewords = output_path+'codesbook.txt' + '_' + task_name
        fp_Rmatrix = output_path+'codesbook.txt' + '_' + task_name + '_RMatrix'
        pqdm.prepare_opqindex_by_file(pid_emb_path, fp_output, fp_codewords, M, Ks, emb_size)