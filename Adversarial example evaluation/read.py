import re
import time
import datetime
import csv
import numpy

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


def words_len_from_text(s, words_to_ignore=[]):
    homos = {
        "-": "˗",
        "9": "৭",
        "8": "Ȣ",
        "7": "𝟕",
        "6": "б",
        "5": "Ƽ",
        "4": "Ꮞ",
        "3": "Ʒ",
        "2": "ᒿ",
        "1": "l",
        "0": "O",
        "'": "`",
        "a": "ɑ",
        "b": "Ь",
        "c": "ϲ",
        "d": "ԁ",
        "e": "е",
        "f": "𝚏",
        "g": "ɡ",
        "h": "հ",
        "i": "і",
        "j": "ϳ",
        "k": "𝒌",
        "l": "ⅼ",
        "m": "ｍ",
        "n": "ո",
        "o": "о",
        "p": "р",
        "q": "ԛ",
        "r": "ⲅ",
        "s": "ѕ",
        "t": "𝚝",
        "u": "ս",
        "v": "ѵ",
        "w": "ԝ",
        "x": "×",
        "y": "у",
        "z": "ᴢ",
    }
    """Lowercases a string, removes all non-alphanumeric characters, and splits
    into words."""
    # TODO implement w regex
    words = []
    word = ""
    for c in " ".join(s.split()):
        if c.isalnum() or c in homos.values():
            word += c
        elif c in "'-_*@" and len(word) > 0:
            # Allow apostrophes, hyphens, underscores, asterisks and at signs as long as they don't begin the
            # word.
            word += c
        elif word:
            if word not in words_to_ignore:
                words.append(word)
            word = ""
    if len(word) and (word not in words_to_ignore):
        words.append(word)
    return len(words)


def read_corpus(path, clean=True, MR=True, encoding='utf8', lower=False):
    data = []
    labels = []
    with open(path, encoding=encoding) as fin:
        for line in fin:
            if MR:
                label, sep, text = line.partition(' ')
                if '1' in label:
                    label = '0'
                elif '2' in label:
                    label = '1'
            if clean:
                text = clean_str(text.strip()) if clean else text.strip()
            if lower:
                text = text.lower()
            labels.append(label)
            data.append(text)

    return data, labels


# texts, labels = read_corpus('../amazon.txt',clean=False)
# with open('amazon', 'w', encoding='utf-8') as f:
#     for text, label in zip(texts, labels):
#         if  words_len_from_text(text) <= 15:
#             f.write(label+' '+text)


# import pandas as pd, numpy as np
# test_data = pd.read_csv('../Twitter_test.csv')
# length_avg=0
# data=[]
# d={'positive':1,'negative':0,'neutral':-1}
# number_=0
# for text in test_data.iterrows():
#     text = text[1].to_dict()
#
#     cl_text=clean_str(text['text'])
#     length = words_len_from_text(cl_text.replace('\n', '.'))
#     if 50>=length>=10:
#         length_avg += length
#         data.append((cl_text, d[text['sentiment']]))
#         number_ +=1
#         if number_>500:
#             break


def load_dbpedia_dataset(path):
    print("reading path: %s" % path)
    label_list = []
    clean_text_list = []
    data_len = []
    with open(path, "r", encoding="utf-8") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=",")
        count = 0
        for row in csv_reader:
            count += 1
            text = " . ".join(row[1:]).lower()
            tokens = text_to_tokens(text)
            if 20 < len(tokens) < 70:
                label_list.append(int(row[0]) - 1)
                data_len.append(len(tokens))
                clean_text_list.append(" ".join(tokens))
    print('len %s' %numpy.mean(data_len))

    return clean_text_list, label_list


def text_to_tokens(text):
    """
    Clean the raw text.
    """
    spliter = re.split(
        r"([\'\#\ \!\"\$\%\&\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~\t\n])",
        text,
    )
    tokens = [token for token in filter(lambda x: (x != "" and x != " "), spliter)]
    return tokens


clean_text_list, label_list = load_dbpedia_dataset('../test.csv')

# with open('db', 'w', encoding='utf-8') as f:
#     for text,label in zip(clean_text_list,label_list):
#         f.write(str(label)+' '+text+'\n')