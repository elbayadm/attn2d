import os.path as osp
import numpy as np
import re


def get_dal(ctxs, src_len):
    # tau = arg min_t {g(t) = src_len}
    prev_gtbis = 0
    dal = 0
    hyp_len = len(ctxs)
    gamma = hyp_len / src_len
    for t, gt in enumerate(ctxs):
        if t:
            gtbis = max(gt, prev_gtbis + 1/gamma)
        else:
            gtbis = gt
        dal += gtbis - t / gamma
        prev_gtbis = gtbis
    return dal / len(ctxs)


def get_al(ctxs, src_len):
    hyp_len = len(ctxs)
    gamma = hyp_len / src_len
    als = []
    tg = []
    for t, c in enumerate(ctxs):
        if c < src_len:
            als.append(c - t / gamma)
            tg.append(t/gamma)
        else:
            als.append(c - t / gamma)
            tg.append(t/gamma)
            break
    return sum(als)/float(len(als))


def get_delays(res, shift=1, delta=1, catchup=1):
    src_lengths = {}
    trg_lengths = {}
    hyp_lengths = {}
    reads = {}
    contexts = {}
    if osp.exists(res):
        with open(res, 'r') as f:
            for line in f:
                if line.startswith('S-'):
                    line = line.split('S-')[-1]
                    line = line.split()
                    sid = line[0]
                    src_lengths[sid] = len(line) - 1
                elif line.startswith('T-'):
                    line = line.split('T-')[-1]
                    line = line.split()
                    sid = line[0]
                    trg_lengths[sid] = len(line) - 1
                elif line.startswith('H-'):
                    line = line.split('H-')[-1]
                    line = line.split()
                    sid = line[0]
                    hyp_lengths[sid] = len(line) - 2  # Id and score
                elif line.startswith('E-'):
                    line = line.split('E-')[-1]
                    line = line.split()
                    sid = line[0]
                    reads[sid] = [int(x) for x in line[1:]]
                elif line.startswith('C-'):
                    line = line.split('C-')[-1]
                    line = line.split()
                    sid = line[0]
                    contexts[sid] = [int(x) for x in line[1:]]
                
                elif 'BLEU' in line:
                    match = re.search(r'BLEU4 = (\S+)', line)
                    if match:
                        bleu = float(match.group(1)[:-1])
                    match = re.search(r'ratio=(\S+)', line)
                    if match:
                        ratio = float(match.group(1)[:-1])

        delays = {}
        lagging = {}
        diff_lagging = {}

        if contexts:
            for k in contexts:
                ctxs = [min(c, src_lengths[k]) for c in contexts[k]]  # error in formatting the contexts at early runs
                # Assert length of contexts is equal to length of hyp:
                if len(ctxs) < hyp_lengths[k]:
                    ctxs = ctxs + [ctxs[-1]] * (hyp_lengths[k] - len(ctxs))
                else:
                    ctxs = ctxs[:hyp_lengths[k]]
                assert len(ctxs) == hyp_lengths[k], 'There should be as many contexts as there are tokens in the hypothesis'
                assert max(ctxs) <= src_lengths[k], 'Contexts should be less or equal than the source lenght!'
                d = sum(ctxs) / len(ctxs) / src_lengths[k]
                delays[k] = d
                lagging[k] = get_al(ctxs, src_lengths[k])
                diff_lagging[k] = get_dal(ctxs, src_lengths[k])

            ap = np.mean(np.array(list(delays.values())))
            sap = np.std(np.array(list(delays.values())))
            al = np.mean(np.array(list(lagging.values())))
            sal = np.std(np.array(list(lagging.values())))
            dal = np.mean(np.array(list(diff_lagging.values())))
            sdal = np.std(np.array(list(diff_lagging.values())))
        else:
            for k, hyp_len  in hyp_lengths.items():
                src_len = src_lengths[k]
                ctx_0 = min(shift, src_len)
                ctxs = [ctx_0]
                for t in range(1, hyp_len):
                    ctx = min(shift + (t // catchup) * delta, src_len)
                    ctxs.append(ctx)
                d = sum(ctxs) / len(ctxs) / src_len
                delays[k] = d
                lagging[k] = get_al(ctxs, src_len)
                diff_lagging[k] = get_dal(ctxs, src_len)

            ap = np.mean(np.array(list(delays.values())))
            sap = np.std(np.array(list(delays.values())))
            al = np.mean(np.array(list(lagging.values())))
            sal = np.std(np.array(list(lagging.values())))
            dal = np.mean(np.array(list(diff_lagging.values())))
            sdal = np.std(np.array(list(diff_lagging.values())))

        return (bleu, ratio, al, sal, ap, sap, dal, sdal)
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--shift', '-w', default=1, type=int)
    parser.add_argument('--delta', '-d', default=1, type=int)
    parser.add_argument('--catchup', '-c', default=1, type=int)

    parser.add_argument('model')
    args = parser.parse_args()
    res = args.model
    results = get_delays(res, args.shift, args.delta, args.catchup)
    if results is not None:
        B, ratio, al, sal, ap, sap, dal, sdal = results
        print('%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' % (B, ratio, ap, sap, al, sal, dal, sdal))

    else:
        print('Missing results')
