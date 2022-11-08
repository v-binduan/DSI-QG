# -*- coding=utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import logging
import numpy as np
import threading
import random
from paddle import fluid
import multiprocessing
from random import choice
import time
try:
    import queue
except ImportError:
    import Queue as queue


class GeneratorEnqueuer(object):
    """
    Builds a queue out of a data generator.
    Args:
        generator: a generator function which endlessly yields data
        use_multiprocessing (bool): use multiprocessing if True,
            otherwise use threading.
        wait_time (float): time to sleep in-between calls to `put()`.
        random_seed (int): Initial seed for workers,
            will be incremented by one for each workers.
    """
    def __init__(self,
                 generator,
                 max_epoch,
                 use_multiprocessing=False,
                 wait_time=0.05,
                 random_seed=None):
        self.wait_time = wait_time
        self._generator = generator
        self._use_multiprocessing = use_multiprocessing
        self._threads = []
        self._stop_event = None
        self.queue = None
        self._manager = None
        self.seed = random_seed
        self.max_epoch = max_epoch

    def start(self, workers=1, max_queue_size=10):
        """
        Start worker threads which add data from the generator into the queue.
        Args:
            workers (int): number of worker threads
            max_queue_size (int): queue size
                (when full, threads could block on `put()`)
        """
        def data_generator_task():
            """
            Data generator task.
            """
            def task():
                if (self.queue is not None and
                        self.queue.qsize() < max_queue_size):
                    generator_output = next(self._generator)
                    self.queue.put((generator_output))
                else:
                    time.sleep(self.wait_time)
            if not self._use_multiprocessing:
                while not self._stop_event.is_set():
                    with self.genlock:
                        try:
                            task()
                        except Exception:
                            self._stop_event.set()
                            break
            else:
                while not self._stop_event.is_set():
                    try:
                        task()
                    except Exception:
                        self._stop_event.set()
                        break
        try:
            if self._use_multiprocessing:
                self._manager = multiprocessing.Manager()
                self.queue = self._manager.Queue(maxsize=max_queue_size)
                self._stop_event = multiprocessing.Event()
            else:
                self.genlock = threading.Lock()
                self.queue = queue.Queue()
                self._stop_event = threading.Event()
            for _ in range(workers):
                if self._use_multiprocessing:
                    # Reset random seed else all children processes
                    # share the same seed
                    np.random.seed(self.seed)
                    thread = multiprocessing.Process(target=data_generator_task)
                    thread.daemon = True
                    if self.seed is not None:
                        self.seed += 1
                else:
                    thread = threading.Thread(target=data_generator_task)
                self._threads.append(thread)
                thread.start()
        except:
            self.stop()
            raise

    def is_running(self):
        """
        Returns:
            bool: Whether the worker theads are running.
        """
        return self._stop_event is not None and not self._stop_event.is_set()

    def stop(self, timeout=None):
        """
        Stops running threads and wait for them to exit, if necessary.
        Should be called by the same thread which called `start()`.
        Args:
            timeout(int|None): maximum time to wait on `thread.join()`.
        """
        if self.is_running():
            self._stop_event.set()
        for thread in self._threads:
            if self._use_multiprocessing:
                if thread.is_alive():
                    thread.join(timeout)
            else:
                thread.join(timeout)
        if self._manager:
            self._manager.shutdown()
        self._threads = []
        self._stop_event = None
        self.queue = None
        
    def get(self):
        """
        Creates a generator to extract data from the queue.
        Skip the data if it is `None`.
        # Yields
            tuple of data in the queue.
        """
        while self.is_running():
            if not self.queue.empty():
                inputs = self.queue.get()
                if inputs is not None:
                    yield inputs
            else:
                time.sleep(self.wait_time)

class DataReader(object):
    def __init__(self, 
        D, 
        K, 
        hidden_dim, 
        data_files_path,
        batch_size,
        epochs,
        pid2path_file,
        pid2emb_file="None",
        random_walk_negsampling_file="None",
        steps_per_epoch=1e10,
        trainer_id=0, 
        num_trainer=1,
        neg_sampling_strategy='nce',
        neg_sampling_exp_base=1, 
        neg_num=1,
        is_sideinfo=False,
        max_queue=128,
        num_workers=1):

        self.D = D
        self.K = K
        self.hidden_dim = hidden_dim
        self.neg_sampling_strategy = neg_sampling_strategy
        self.neg_sampling_exp_base = neg_sampling_exp_base
        self.decay = 0.9883
        self.batch_size = batch_size
        self.epochs = epochs

        self.trainer_id = trainer_id
        self.num_trainer = num_trainer
        self.use_mp = num_workers != 1
        self.max_queue = max_queue
        self.num_workers = num_workers

        dev_count = fluid.core.get_cuda_device_count()
        print("dev_count:", dev_count)
        self.total_iter = self.epochs * steps_per_epoch * dev_count

        self.current_epoch = 0
        self.current_file_index = 0
        self.total_file = 0
        self.current_file = None
        self.data_files = [os.path.join(data_files_path, f) for idx, f in enumerate(os.listdir(data_files_path)) if num_workers==1 or idx % self.num_trainer == self.trainer_id]
        print(self.data_files[:10])

        self.table = []
        self.rtable = list(range(self.D))

        self._initW = None
        self.is_sideinfo = is_sideinfo

        self.neg_num = neg_num

        self.pid2path = {}
        self.path2pid = {}
        self.pid2path_file = pid2path_file
        self._load_pid2path()
        self.pid2emb_file = pid2emb_file
        self.paths = self.path2pid.keys()

        self.random_walk_negsampling_file = random_walk_negsampling_file

        self.not_found = 0
        self.round = 1

        self.pid2emb = {}
        #init path slot fea
        if self.is_sideinfo: 
            self.pid2emb = {}
            with open(self.pid2emb_file, 'r') as fin:
                for line in fin:
                    arr = line.strip().split('\t')
                    if len(arr) != 1+self.hidden_dim:
                        continue
                    pid = arr[0]
                    emb = np.array([float(i) for i in arr[1:]])
                    self.pid2emb[pid] = emb
                    if len(self.pid2emb) % 10000 == 0:
                        print("read %d sideinfo emb" % len(self.pid2emb))
            print("totally read %d sideinfo emb" % len(self.pid2emb))

        self.pid2neg = {}
        if self.neg_sampling_strategy == "random_walk":
            with open(self.random_walk_negsampling_file, 'r') as fin:
                for line in fin:
                    arr = line.strip().split('\t')
                    if len(arr) != 2:
                        pid, neg_pid = arr[0]
                        self.pid2neg[pid] = neg_pid
            print("totally read %d random walk negsamples" % len(self.pid2neg))


    def _load_pid2path(self):
        self.path2pid = {}
        self.pid2path = {}
        with open(self.pid2path_file, 'r') as fin:
            for line in fin:
                arr = line.strip().split('\t')
                pid, path = arr
                self.path2pid[path] = pid
                self.pid2path[pid] = path
        print("load %d pid2path" % len(self.pid2path))   

    def get_progress(self):
        return self.current_epoch, self.current_file_index, self.total_file, self.current_file

    def read_batch_data(self):
        self.total_file = len(self.data_files)
        batch_size = self.batch_size
        assert self.total_file > 0, "[Error] data_files is empty"
        
        def reader():
            for epoch in range(self.epochs + 1):
                self.current_epoch += 1
                if self.neg_sampling_strategy == 'oracle':
                    cur_neg_sampling_exp_base = min(self.neg_sampling_exp_base, 1.0 / pow(self.decay, epoch))
                    self.table = []
                    for i in range(self.D):
                        if i > 0:
                            self.table.extend([i] * int(pow(cur_neg_sampling_exp_base,self.D-1-i)))
                    print("epoch %d, table length:%d" % (epoch, len(self.table)))

                query_data = []
                route_data = []
                neg_route_data = []
                label_data = []
                sideinfo_feas = []

                idx = 0
                random.shuffle(self.data_files)
                for fidx, file_ in enumerate(self.data_files):
                    self.current_file_index = fidx + 1
                    self.current_file = file_
                    with open(file_, 'r') as fin:
                        for line in fin:
                            arr = line.strip().split('\t')
                            if random.random() < 0.5:
                                continue
                            if len(arr) != 3:
                                continue
                            pid = arr[1]
                            if pid not in self.pid2path:
                                self.not_found += 1
                                if self.not_found % 1000 == 0:
                                    print("not found pid: %d" % self.not_found)
                                continue
                            route_str = self.pid2path[pid]
                            route = list(route_str) if len(route_str.split(' ')) == 0 else route_str.split(' ')
                            query_emb = arr[0].split(' ')
                            if len(route) != self.D or len(query_emb) != self.hidden_dim:
                                continue
                            route = np.array(route).astype(np.int64)
                            if self.neg_sampling_strategy == 'oracle': 
                                n = choice(self.table)
                                idxs = set()
                                while len(idxs) < n:
                                    idxs.add(choice(self.rtable))
                                neg_route_str = ""
                                neg_not_in_cnt = 0
                                while neg_route_str not in self.path2pid:
                                    neg_not_in_cnt += 1
                                    if neg_not_in_cnt % 10 == 0:
                                        print("neg_route_str is fuck")
                                    neg_route = np.array(route).astype(np.int64)
                                    for r_i in idxs:
                                        neg_val = random.randint(0, self.K-1)
                                        while neg_val == route[r_i]:
                                            neg_val = random.randint(0, self.K-1)
                                        neg_route[r_i] = neg_val
                                    neg_route_str = ' '.join([str(i) for i in neg_route])
                                neg_pid = self.path2pid[neg_route_str]
                                if neg_pid not in self.pid2emb:
                                    #print("neg_pid: %s not in pid2emb" % neg_pid)
                                    continue
                                neg_sidefea = self.pid2emb[neg_pid] # if neg_pid in self.pid2emb else [0.0]*self.hidden_dim
                            elif self.neg_sampling_strategy == 'random_walk':
                                if pid not in self.pid2neg:
                                    continue
                                neg_pid = self.pid2neg[pid]
                                if neg_pid not in self.pid2emb or neg_pid not in self.pid2path:
                                    continue
                                neg_route_str = self.pid2path[neg_pid]
                                neg_route = np.array(neg_route_str.split(' ')).astype(np.int64)
                                neg_sidefea = self.pid2emb[neg_pid]
                            elif self.neg_sampling_strategy == 'random':
                                neg_route_list = []
                                while len(neg_route_list) < self.neg_num:
                                    neg_route_str = choice(self.paths)
                                    if neg_route_str != route_str:
                                        neg_route_list.append(neg_route_str.split(' '))
                                neg_route = np.array(neg_route_list).astype(np.int64) 
                            query_emb = np.array(query_emb).astype(np.float32)
                            if idx < batch_size:
                                query_data.append(query_emb)
                                route_data.append(route)
                                if self.neg_sampling_strategy != 'None':
                                    neg_route_data.append(neg_route)
                                label_data.append([1])
                                idx += 1
                            else: 
                                if self.neg_sampling_strategy != 'None':
                                    yield np.array(query_data), np.array(route_data), np.array(label_data), np.array(neg_route_data)
                                else:
                                    yield np.array(query_data), np.array(route_data), np.array(label_data)
                                idx = 1
                                query_data = [query_emb]
                                route_data = [route]
                                if self.neg_sampling_strategy != 'None':
                                    neg_route_data = [neg_route]
                                label_data = [[1]]
                                sideinfo_feas = []
        
        def infinite_reader():
            while True:
                for data in reader():
                    yield data

        def mp_reader():
            cnt = 0
            try:
                enqueuer = GeneratorEnqueuer(
                    infinite_reader(), max_epoch=self.epochs, use_multiprocessing=True)
                enqueuer.start(max_queue_size=self.max_queue, workers=self.num_workers)
                generator_out = None
                while True:
                    while enqueuer.is_running():
                        if not enqueuer.queue.empty():
                            generator_out = enqueuer.queue.get()
                            break
                        else:
                            time.sleep(0.02)
                    yield generator_out
                    cnt += 1
                    if cnt >= self.total_iter:
                        enqueuer.stop(5)
                        return
                    generator_out = None
            except Exception as e:
                print("Exception occured in reader: {}".format(str(e)))
            finally:
                if enqueuer:
                    enqueuer.stop()
        
        if self.use_mp:
            return mp_reader
        return reader

class NaiveDataReader(DataReader):

    def read_pid2pqindex(self, fp):
        pid2pqindex = {}
        with open(fp, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split('\t')
                pid = line[0]
                pqindex = line[1].replace(' ', '')
                pid2pqindex[pid] = pqindex
        return pid2pqindex
    
    def read_batch_data_fortest(self, worker_count=1, worker_id=0, infer_device_count=1, infer_device_id=0):
        query_data = []
        query_str = []
        idx = 0
        for fidx, file in enumerate(self.data_files):
            if fidx % worker_count != worker_id or fidx % infer_device_count != infer_device_id:
                    continue
            with open(file, 'r') as fin:
                for line in fin:
                    arr = line.strip().split('\t')
                    if len(arr) != 3:
                        continue
                    query_emb = arr[0].split(' ')
                    query = arr[1]
                    if len(query_emb) != self.hidden_dim:
                        continue
                    query_emb = [float(i) for i in query_emb]
                    query_emb = np.array(query_emb).astype(np.float32)
                    if idx < self.batch_size:
                        query_data.append(query_emb)
                        query_str.append(query)
                        idx += 1
                    else:
                        yield [np.array(query_data), np.array(query_str)]
                        idx = 1
                        query_data = [query_emb]
                        query_str = [query]
        if len(query_data) != 0:
            yield [np.array(query_data), np.array(query_str)]
 

if __name__ == '__main__':
    print('yes')
