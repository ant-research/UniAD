import torch
import json
from bert_score import BERTScorer  # from https://github.com/Tiiiger/bert_score
bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True, baseline_path='put roberta-large.tsv path here')


def recall_within_neighbours(sentences_gt, sentences_gen, topk=(1,5), N=5):
    """compute R@k/N as described in AutoAD-II (https://www.robots.ox.ac.uk/~vgg/publications/2023/Han23a/han23a.pdf)
    This metric compares a (long) list of sentences with another list of sentences.
    It uses BERTScore (https://github.com/Tiiiger/bert_score) to compute sentence-sentence similarity,
    but uses the relative BERTScore values to get a recall, for robustness.
    """
    # get sentence-sentence BertScore
    ss_score = []
    for sent in sentences_gen:
        ss_score.append(bert_scorer.score(sentences_gt, [sent] * len(sentences_gt))[-1])
    ss_score = torch.stack(ss_score, dim=0)

    window = N
    topk_output = []
    for i in range(0, ss_score.shape[0]-window+1, window//2):
        topk_output.append(calc_topk_accuracy(ss_score[i:i+window,i:i+window], torch.arange(window).to(ss_score.device), topk=topk))
    
    topk_avg = torch.stack(topk_output, 0).mean(0).tolist()
    for k, res in zip(topk, topk_avg):
        print(f"Recall@{k}/{N}: {res:.3f}")
    return topk_avg


def calc_topk_accuracy(output, target, topk=(1,)):
    """
    Modified from: https://gist.github.com/agermanidis/275b23ad7a10ee89adccf021536bb97e
    Given predicted and ground truth labels, calculate top-k accuracies.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))
    return torch.stack(res)


if __name__ == '__main__':
    with open('') as f: # put gt AD json path here
        gt_ads = json.load(f)

    with open('') as f: # put predicted AD json path here
        generate_ads = json.load(f)

    results_all = []
    for k, v in gt_ads.items():
        result = recall_within_neighbours(v, generate_ads[k], topk=(5,), N=16)
        results_all.append(result)

    print('Recall value: ', sum(results_all) / len(results_all))