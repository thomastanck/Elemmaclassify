import zlib

def shufflehash(x):
    return zlib.crc32(key(x).encode('utf-8')) & 0xffffffff

def shuffle_by_hash(l, key=str):
    return sorted(l, key=lambda x: shufflehash(key(x)))

def split_by_amount(data, cumulative_fractions, sortkey):
    shuf = sorted(data, key=sortkey)
    splits = []
    cur_amt = 0
    for frac in cumulative_fractions:
        amt = len(shuf) * frac
        splits.append(shuf[cur_amt:amt])
        cur_amt = amt
    splits.append(shuf[cur_amt:])
    return splits

def split_by_criteria(data, criteria):
    classes = [[] for i in range(len(criteria) + 1)]
    for p in data:
        for i, c in enumerate(criteria):
            if c(p):
                classes[i].append(p)
                break
        else:
            classes[len(criteria)].append(p)
    return classes

def train_test_split(names, test_split, sortkey):
    return split_by_amount(names, [1-test_split], sortkey)

def pos_neg_split(names, pos_criteria):
    return split_by_criteria(names, [pos_criteria])

Stats = collections.namedtuple('Stats', 'tp tn fp fn')
Stats.__new__.__defaults__ = (0,) * 4

def calc_stats(stats):
    tp, tn, fp, fn = stats
    total = tp + tn + fp + fn
    classified_pos = tp + fp
    actual_pos = tp + fn
    acc = 0 if total == 0 else (tp + tn) / total
    precision = 0 if classified_pos == 0 else tp / classified_pos
    recall = 0 if actual_pos == 0 else tp / actual_pos
    fscore = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return acc, precision, recall, fscore

def update_stats(stats, prediction, actual):
    tp, tn, fp, fn = stats
    if prediction and actual:
        return Stats(tp + 1, tn, fp, fn)
    elif not prediction and actual:
        return Stats(tp, tn, fp, fn + 1)
    elif prediction and not actual:
        return Stats(tp, tn, fp + 1, fn)
    elif not prediction and not actual:
        return Stats(tp, tn + 1, fp, fn)

