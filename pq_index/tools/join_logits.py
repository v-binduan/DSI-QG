import sys
import numpy as np
import base64

def mapper_qad():
    for line in sys.stdin:
        arr = line.strip().split('\t')
        if len(arr) != 2:
            continue
        query, ad = arr
        print '\t'.join([query, ad, 'qad'])

def reducer_join_query_logits():
    pre_key = ''
    pids = []
    logits = None
    def post_data():
        if logits is None or len(pids) == 0:
            return
        res = {}
        for pid_ in pids:
            if pid_ in res:
                res[pid_] += (1-logits)
            else:
                res[pid_] = (1-logits)
        for pid_ in res:
            logits_b64_npstr_ = base64.b64encode(res[pid_])
            print '\t'.join([pid_, logits_b64_npstr_])

    for line in sys.stdin:
        arr = line.strip().split('\t')
        if len(arr) != 2 and len(arr) != 3:
            continue
        query = arr[0]
        if query != pre_key and pre_key != '':
            post_data()
            pids = []
            logits = None
        pre_key = query
        if len(arr) == 2:
            logits_b64_npstr = arr[1]
            try:
                logits = np.fromstring(base64.b64decode(logits_b64_npstr), dtype=np.float32).reshape(M,Ks)
            except:
                continue
        if len(arr) == 3:
            pid = arr[1]
            pids.append(pid)
    if pre_key != '':
        post_data()


def reducer():
    pre_key = ''
    logits = []
    def post_data():
        if len(logits) == 0:
            return
        logits_ = np.sum(logits, axis=0)
        logits_argsort = np.argsort(logits_, axis=1)
        logits_b64_npstr_ = base64.b64encode(logits_)
        logits_argsort_b64_str_ = base64.b64encode(logits_argsort) 
        print '\t'.join([pre_key, logits_b64_npstr_, logits_argsort_b64_str_])

    for line in sys.stdin:
        arr = line.strip().split('\t')
        if len(arr) != 2:
            continue
        pid, logits_b64_npstr = arr
        if pre_key != pid and pre_key != '':
            post_data()
            logits = []
        pre_key = pid 
        logits.append(np.fromstring(base64.b64decode(logits_b64_npstr), dtype=np.float32).reshape(M,Ks))
    if pre_key != '':
        post_data()

if __name__ == '__main__':
    task = sys.argv[1]
    if task == 'map_qad':
        mapper_qad()
    if task == 'red_join_q_logits':
        M, Ks = int(sys.argv[2]), int(sys.argv[3])
        reducer_join_query_logits()
    if task == 'red_joinad':
        M, Ks = int(sys.argv[2]), int(sys.argv[3])
        reducer()
