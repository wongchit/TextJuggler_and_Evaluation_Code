import collections
import csv
import re
import string

import pandas as pd


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC"""

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def read_corpus(path, clean=True, MR=True, encoding='utf8', shuffle=False, lower=False):
    data = []
    labels = []
    with open(path, encoding=encoding) as fin:
        for line in fin:
            if MR:
                label, sep, text = line.partition(' ')
                label = int(label)
            else:
                label, sep, text = line.partition(',')
                label = int(label) - 1
            if clean:
                text = clean_str(text.strip()) if clean else text.strip()
            if lower:
                text = text.lower()
            labels.append(label)
            data.append(text)

    return data, labels


def read_data(filepath, data_size=1000, lowercase=False, ignore_punctuation=False, stopwords=[]):
    """
    Read the premises, hypotheses and labels from some NLI dataset's
    file and return them in a dictionary. The file should be in the same
    form as SNLI's .txt files.

    Args:
        filepath: The path to a file containing some premises, hypotheses
            and labels that must be read. The file should be formatted in
            the same way as the SNLI (and MultiNLI) dataset.

    Returns:
        A dictionary containing three lists, one for the premises, one for
        the hypotheses, and one for the labels in the input data.
    """
    labeldict = {"contradiction": 0,
                 "entailment": 1,
                 "neutral": 2}
    with open(filepath, 'r', encoding='utf8') as input_data:
        texts = []
        labels = []

        # Translation tables to remove punctuation from strings.
        punct_table = str.maketrans({key: ' '
                                     for key in string.punctuation})

        for idx, line in enumerate(input_data):
            if idx >= data_size:
                break

            line = line.strip().split('\t')

            # Ignore sentences that have no gold label.
            if line[0] == '-':
                continue

            premise = line[1]
            hypothesis = line[2]

            if lowercase:
                premise = premise.lower()
                hypothesis = hypothesis.lower()

            if ignore_punctuation:
                premise = premise.translate(punct_table)
                hypothesis = hypothesis.translate(punct_table)

            # Each premise and hypothesis is split into a list of words.
            # data.append((collections.OrderedDict({'premise': premise, 'hypothesis': hypothesis}), labeldict[line[0]]))
            texts.append((collections.OrderedDict({'premise': premise, 'hypothesis': hypothesis})))
            labels.append(labeldict[line[0]])
        return texts, labels


def load_data_from_csv(dataset_path):
    texts, labels = read_data(dataset_path)
    data = zip(texts, labels)
    adv_text = []
    adv_label = []
    for text, label in data:
        if 150 >= len(text['hypothesis']) >= 20:
            adv_text.append(text)
            adv_label.append(label)
    return adv_text, adv_label


def save_to_csv(filename, texts, labels):
    row = {'text': texts, 'labels': labels}
    df = pd.DataFrame(row)
    df.to_csv(filename, quoting=csv.QUOTE_NONNUMERIC, index=False, mode='a', header=True)


texts, labels = load_data_from_csv('../xxxx')
save_to_csv(filename='nli.csv', texts=texts, labels=labels)

