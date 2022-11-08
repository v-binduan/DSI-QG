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
import traceback
import multiprocessing as mp
import pickle
import json
from multiprocessing import Process
from multiprocessing import Lock
from multiprocessing import Manager
from multiprocessing import Value
from multiprocessing import Queue

import base64


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


def read_one_file(fp_path, fp, cnt, dists_list, dists_argsort_list, terms, M, Ks):
    with open(os.path.join(fp_path, fp), 'r') as fin:
        for line in fin:
            arr = line.strip().split('\t')
            if len(arr) != 3:
                continue
            if cnt % 10000 == 0:
                logger.info("read %d pid's logits" % cnt) 
            pid, dist_str, dist_argsort_str = arr
            dist = np.fromstring(base64.b64decode(dist_str), dtype=np.float32).reshape(M,Ks) #M,Ks
            dist_argsort = np.fromstring(base64.b64decode(dist_argsort_str), dtype=np.int64).reshape(M,Ks) #M,Ks
            terms.append(pid) 
            cnt += 1 
            dists_list.append(dist)
            dists_argsort_list.append(dist_argsort)
    return cnt


def read_func(tid, fp_path, lock, dists_list_map, dists_argsort_list_map, terms_map, M, Ks, run_task_num, queue):
    logger.info("begin __read_func, t[%d]", tid)
    cnt = 0
    terms = []
    dists_list = []
    dists_argsort_list = []

    while (1):
        file, remain = pop_file(lock)
        if not file:
            break
 
        logger.info("reading file[%s], remain %s..., t[%d]", file, remain, tid)
        start = time.time()
        run_task_num.value += 1
        cnt = read_one_file(fp_path, file, cnt, dists_list, dists_argsort_list, terms, M, Ks)
        run_task_num.value -= 1
        logger.info("read file[%s] finished, used_time[%lus], running task %d, t[%d]",
                file, time.time()-start, run_task_num.value, tid)

    start = time.time()
    dists_list_map[tid] = dists_list
    dists_argsort_list_map[tid] = dists_argsort_list
    terms_map[tid] = terms

    logger.info("end __read_func, t[%d], fill_map_time[%lus]", tid, time.time()-start)

    queue.put(tid)


class PQINDEX(object):

    def __init__(self):
        pass

    @staticmethod
    def read_term_emb_by_file_parallel(fp_path, M, Ks, threads_num=128):
        terms = []
        emb_list = []
        cnt = 0
        fp_list = os.listdir(fp_path)
        dump_file_list(fp_list)

        process_list = []
        manager = Manager()
        lock = manager.Lock()
        dists_list_map = manager.dict()
        dists_argsort_list_map = manager.dict()
        terms_map = manager.dict()
        queue = Queue()
        run_task_num = Value('i', 0)

        for i in range(0, threads_num):
            t = Process(target=read_func,
                    args=(i, fp_path, lock, dists_list_map, dists_argsort_list_map, terms_map, M, Ks, run_task_num, queue))
            t.start()
            process_list.append(t)
 
        terms = []
        dists_list = []
        dists_argsort_list = []
        joined_thrd_num  = 0
        while joined_thrd_num < threads_num:
            i = queue.get()
            p = process_list[i]
            logger.info('joining tid[%d]' % i)
            p.join()
            start = time.time()
            terms.extend(terms_map[i])
            dists_list.extend(dists_list_map[i])
            dists_argsort_list.extend(dists_argsort_list_map[i])
            logger.info('tid[%d] extend time[%lus] joined_num[%d]' % (i, time.time()-start,
                joined_thrd_num + 1))
            joined_thrd_num = joined_thrd_num + 1

        return terms, dists_list, dists_argsort_list

    @staticmethod
    def _read_pqindex(fp):
        pid2path = {}
        paths = []
        with open(fp, 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                pid = line[0]
                path = line[1].split(' ')
                if len(path) != M:
                    continue
                pid2path[pid] = [int(node) for node in path]
                paths.append([int(node) for node in path])
        return np.array(paths,dtype='int64'), pid2path
    

    @staticmethod
    def _read_codeword_emb(fp, M, Ks):
        codewords = []
        with open(fp, 'r') as f:
            for i in f:
                code = i.strip().split('\t')
                code = map(float, code)
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
 
    @staticmethod
    def _get_unique_pqindex_string(pqindex):
        tmp = [''.join(map(str,i)) for i in pqindex]
        tmp = set(tmp)
        return tmp


    def _adjust(self, process_id, queue):
        N = len(self.dists) 
        M, Ks = self.dists[0].shape
        processed = 0
        catch_time = 0
        processed_res = []
        d = None
        while True:
            try:
                if queue.qsize() % 100 == 0 and queue.qsize() != 0:
                    logger.info("queue size: {0}, timeout: {1}".format(queue.qsize(), self.timeout))
                t0 = time.time()
                for _ in range(5):
                    try:
                        paths, d = queue.get(timeout=self.timeout)
                    except:
                        paths = None
                    if d == 0:
                        logger.info("get d==0 from queue")
                        paths = dict(enumerate(self.paths[:,0]))
                        logger.info("get paths_0 from self.paths")
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

                #get dist list of each node in current layer
                for id, node in paths.items():
                    dist = self.dists[id][d]
                    if node in node_pidlist:
                        node_pidlist[node].append([id, dist[node]])
                    else:
                        node_pidlist[node] = [[id, dist[node]]]
                t1 = time.time()
                #logger.info("get dist list of each node in layer {0}: {1}".format(d+1, t1-tstart))

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

                    pid_list.sort(key=lambda x:x[1])
                    #pid_list.sort(key=lambda x:x[0])
                    #logger.info("pidlist sort done")
                    while len(pid_list) > max_num:
                        tmp_id, _ = pid_list.pop()
                        new_node = None
                        for node in self.dists_argsort[tmp_id][d]:
                            if node not in balanced_nodes:
                                new_node = node
                                break
                        assert new_node is not None
                        assert new_node < Ks
                        if new_node in node_pidlist:
                            node_pidlist[new_node].append([tmp_id, self.dists[tmp_id][d][new_node]])
                        else:
                            node_pidlist[new_node] = [[tmp_id, self.dists[tmp_id][d][new_node]]]
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
                #processed_res.append([d, paths])
            except BaseException as e:
                logger.info("error and retry {}".format(e))
                logger.error(traceback.print_exc())
                queue.put((paths, d))

        logger.info("Process {0} process {1} times".format(os.getpid(), len(processed_res)))
    

    def _read_train_file(self, train_path, query_model_logits=None):
        data = {}
        pid2id = {}
        files = [os.path.join(train_path, file) for file in os.listdir(train_path)]
        cnt = 0
        for file in files:
            with open(file, 'r') as fin:
                for line in fin:
                    if is_validation_task:
                        arr = line.strip().decode('utf-8').encode('gbk').split("\x01")
                        if len(arr) != 5:
                            continue
                    else:
                        arr = line.strip().split('\t')
                        if len(arr) != 2:
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
                if cnt % 10000 == 0:
                    logger.info("read %d query's logits" % cnt)
                query, logits_b64_npstr = arr
                logits = np.fromstring(base64.b64decode(logits_b64_npstr), dtype=np.float32) #M,Ks
                data[query] = logits
        return data

    def _read_direct_pid_logits(self, predict_logits_path, M, Ks):
        terms, dists_list, dists_argsort_list = self.read_term_emb_by_file_parallel(predict_logits_path, M, Ks)
        id2pid = dict(enumerate(terms))
        return id2pid, dists_list, dists_argsort_list

    def jtm_from_model_logits(self, train_path, predict_logits_file, fp_output_rebuild, M=8, Ks=5):
        start = time.time()

        logger.info('begin read file')
        self.id2pid, self.dists, self.dists_argsort = self._read_direct_pid_logits(predict_logits_file, M, Ks)
        logger.info("read %d pid's predict logits" % len(self.id2pid))

        logger.info(self.id2pid[0])
        self.fp_output_rebuild = fp_output_rebuild

        N = len(self.id2pid)

        self.paths = []
        for id in range(N):
            path = self.dists_argsort[id][:, 0]
            self.paths.append(path.tolist())
        self.paths = np.array(self.paths, dtype='int64')

        logger.info("dists expample" + str(self.dists[0]))
        logger.info("path example" + str(self.paths[0]))
        #logger.info('set of path' + str(len(self._get_unique_pqindex_string(self.paths))))

        #paths_0 = dict(enumerate(self.paths[:,0]))
        paths_0 = None

        self.parall = 30
        self.timeout = 5

        queue = mp.Queue()
        queue.put((paths_0, 0))
        logger.info("put paths_0 into queue")
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
        
    def jtm_from_model_logits_onlysave(self, train_path, predict_logits_file, fp_output_rebuild, M=8, Ks=5):
        start = time.time()

        logger.info('begin read file')
        pid2id, query2pid = self._read_train_file(train_path)
        self.id2pid = dict(zip(pid2id.values(), pid2id.keys()))
        logger.info(self.id2pid[0])
        logger.info(self.id2pid[10])
        self.fp_output_rebuild = fp_output_rebuild

        logger.info("read %d pids, %d querys from %s" % (len(pid2id), len(query2pid), train_path))

        self.parall = 30
        self.timeout = 5
        N = len(pid2id)

        self._save_res(self.parall, N, M)
        logger.info('adjust complet cost time {}'.format(time.time()-start))

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

    def jtm_from_file(self, input_fp, output_fp, index_fp, codewords_fp, Rmatrix_fp, M=8, Ks=5):
        start = time.time()
        self.fp_output_rebuild = output_fp
        logger.info(self.emb_list.shape)
        N, D = self.emb_list.shape
        Ds = int(D / M)
        
        ## step1-prepare data: 1)id2path 2)distance matrix 
        logger.info("read codewordemb and Rmatrix") 
        self.paths, pid2path = self._read_pqindex(index_fp)
        codeword_embs = self._read_codeword_emb(codewords_fp, M, Ks) #M,ks,32
        R_matrix = self._read_Rmatrix_emb(Rmatrix_fp)
        logger.info("rotationing") 
        self.emb_list = np.matmul(self.emb_list, R_matrix) 
        logger.info("rotation done") 
        self.dists = np.empty([N, M, Ks])
        for i in range(M):
            vec_sub = self.emb_list[:, i * Ds : (i + 1) * Ds]
            for j in range(Ks):
                codes = codeword_embs[i, j, :]
                codes = np.expand_dims(codes, axis=0)
                _, dist = vq(vec_sub, codes)
                self.dists[:, i, j] = dist
        self.dists_argsort = np.argsort(self.dists, axis=2)

        logger.info("dists expample" + str(self.dists[0, :, :]))
        logger.info("path example" + str(self.paths[0]))
        logger.info('set of path' + str(len(self._get_unique_pqindex_string(self.paths))))

        paths_0 = dict(enumerate(self.paths[:,0]))
        del self.emb_list

        self.parall = 30
        self.timeout = 5

        queue = mp.Queue()
        queue.put((paths_0, 0))
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
    is_validation_task = False

    if not first_init:
        train_path = sys.argv[4]
        predict_logits_file = sys.argv[5]
        fp_output_rebuild = sys.argv[6]
        logger.info(is_validation_task)
        pqdm.jtm_from_model_logits(train_path, predict_logits_file, fp_output_rebuild, M, Ks)
