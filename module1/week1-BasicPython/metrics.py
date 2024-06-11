import math


def calc_precision(tp: int, fp: int) -> float:
    return tp / (tp + fp)

def calc_recall(tp: int, fn: int) -> float:
    return tp / (tp + fn)

def calc_f1_score(tp: int, fp: int, fn: int) -> float:
    if not (isinstance(tp, int) and isinstance(fp, int) and isinstance(fn, int)):
        raise TypeError("All inputs must be integers")

    if tp <= 0 or fp <= 0 or fn <= 0:
        raise ValueError('All inputs must be greater than zero')

    precision = calc_precision(tp, fp)
    recall = calc_recall(tp, fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score

def calc_metrics(tp, fp, fn):
    if not (isinstance(tp, int) and isinstance(fp, int) and isinstance(fn, int)):
        print('All inputs must be integers')
        return

    if tp <= 0 or fp <= 0 or fn <= 0:
        print('All inputs must be greater than zero')
        return

    precision = calc_precision(tp, fp)
    recall = calc_recall(tp, fn)
    f1_score = calc_f1_score(tp, fp, fn)
    print(f'precision is {precision}')
    print(f'recall is {recall}')
    print(f'f1-score is {f1_score}')


if __name__ == '__main__':
    calc_metrics(tp=2, fp=3, fn=4)
    calc_metrics(tp=2, fp='a', fn=4)
    calc_metrics(tp='a', fp=3, fn=4)
    calc_metrics(tp=2, fp=3, fn=0)
    calc_metrics(tp=2.1, fp=3, fn=0)

    # print(round(calc_f1_score(tp='a', fp=2, fn=4), 2))
    # print(round(calc_f1_score(tp=1.5, fp=5, fn=4), 2))
    print(round(calc_f1_score(tp=-5, fp=5, fn=4), 2))

    assert round(calc_f1_score(tp=2, fp=3, fn=5), 2) == 0.33
    print(round(calc_f1_score(tp=2, fp=4, fn=5), 2))
