import string
import collections
import textattack
import torch
import pandas as pd
from textattack.commands.attack.attack_args import TEXTATTACK_DATASET_BY_MODEL, HUGGINGFACE_DATASET_BY_MODEL
from textattack.commands.attack.attack_args_helpers import ARGS_SPLIT_TOKEN
from transformers import BertTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import csv
import re
from collections import defaultdict
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hash_func(inp_vector, projections):

    bools = (np.dot(inp_vector, projections.T) > 0).astype('int')
    return ''.join(bools.astype('str'))


class Table:

    def __init__(self, hash_size, dim):
        self.table = defaultdict(list)
        self.hash_size = hash_size
        self.projections = np.random.randn(self.hash_size, dim)

    def add(self, vecs, label):
        h = hash_func(vecs, self.projections)
        self.table[h].append(label)


class LSH:

    def __init__(self, dim):
        self.num_tables = 5
        self.hash_size = 3
        self.tables = list()
        for i in range(self.num_tables):
            self.tables.append(Table(self.hash_size, dim))

    def add(self, vecs, label):
        for table in self.tables:
            table.add(vecs, label)

    def describe(self):
        for table in self.tables:
            print(len(table.table))
            print(table.table)

    def get_result(self):
        len_tables = []
        indices_to_query = []
        final_set_indices = []
        max_value = -1
        for table in self.tables:
            if len(table.table) > max_value:
                max_value = len(table.table)
                final_table = table
        return final_table


def words_from_text(s, words_to_ignore=[]):
    """Lowercases a string, removes all non-alphanumeric characters, and splits
    into words."""
    # TODO implement w regex
    words = []
    word = ""
    for c in " ".join(s.split()):
        if c.isalnum():
            word += c
        elif c in "'-" and len(word) > 0:
            # Allow apostrophes and hyphens as long as they don't begin the
            # word.
            word += c
        elif word:
            if word not in words_to_ignore:
                words.append(word)
            word = ""
    if len(word) and (word not in words_to_ignore):
        words.append(word)
    return words


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


def create_model_form_path(model_name):
        if model_name in HUGGINGFACE_DATASET_BY_MODEL:
            # Support loading models automatically from the HuggingFace model hub.
            import transformers

            model_name = (
                HUGGINGFACE_DATASET_BY_MODEL[model_name][0]
            )
            if ARGS_SPLIT_TOKEN in model_name:
                model_class, model_name = model_name
                model_class = eval(f"transformers.{model_class}")
            else:
                model_class, model_name = (
                    transformers.AutoModelForSequenceClassification,
                    model_name,
                )
            colored_model_name = textattack.shared.utils.color_text(
                model_name, color="blue", method="ansi"
            )
            textattack.shared.logger.info(
                f"Loading pre-trained model from HuggingFace model repository: {colored_model_name}"
            )
            model = model_class.from_pretrained(model_name)
            tokenizer = textattack.models.tokenizers.AutoTokenizer(model_name)
            model = textattack.models.wrappers.HuggingFaceModelWrapper(
                model, tokenizer
            )
        elif model_name in TEXTATTACK_DATASET_BY_MODEL:
            # Support loading TextAttack pre-trained models via just a keyword.
            model_path, _ = TEXTATTACK_DATASET_BY_MODEL[model_name]
            model = textattack.shared.utils.load_textattack_model_from_path(
                model_name, model_path
            )
            # Choose the approprate model wrapper (based on whether or not this is
            # a HuggingFace model).
            if isinstance(model, textattack.models.helpers.T5ForTextToText):
                model = textattack.models.wrappers.HuggingFaceModelWrapper(
                    model, model.tokenizer
                )
            else:
                model = textattack.models.wrappers.PyTorchModelWrapper(
                    model, model.tokenizer
                )
        else:
            raise ValueError(f"Error: unsupported TextAttack model {model_name}")

        return model


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
        data = []

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
            data.append((collections.OrderedDict({'premise': premise, 'hypothesis': hypothesis}), labeldict[line[0]]))
        print()
        return data


def load_data_from_csv(dataset_path):
    texts, labels = read_corpus(dataset_path)
    data = list(zip(texts, labels))
    return data


def  read_dataset(path, data):

        df = pd.read_csv(path)
        dataset = []
        
        if data == '500samples_imdb.csv':
            s_list = df["sentence"]
            s_label = df["polarity"]
        elif data == '500samples_mnli.csv':
            s_list = df['premise']
            s_label = df["label"]
        else:
            s_list = df["text"]
            s_label = df["label"]
        i = 0
        tok = BertTokenizer.from_pretrained('bert-base-uncased')
        for index, label in enumerate(s_label):
            sent = s_list[index]
            sent = sent.replace("<br />", " ")
            
            # limit the number of tokens to 510 for bert (since only bert is used for language modeling, see lm_sampling.py)
            sent = words_from_text(sent)
            # print(sent)
            # print(len(sent))
            sent = ' '.join(sent)
            tokens = tok.tokenize(sent)
            tokens = tokens[:510]
            text = ' '.join(tokens).replace(' ##', '').replace(' - ', '-').replace(" ' ", "'")

            target = int(label)
            i = i+1
            dataset.append((text, target))
                
        return dataset


def write_to_csv(olm_file_name, res_relevances, input_instances, labels_true, labels_pred):
    f = open("olm-files/" + olm_file_name, "w")
    writer = csv.writer(f)  

    writer.writerow(['tokenized_text', 'relevances', 'true label', 'predicted label'])
    
    for i in range(len(res_relevances)):
        list_relevances = []
        for k in range(len(res_relevances[i])):
            list_relevances.append(res_relevances[i][('sent', k)])

        writer.writerow([input_instances[i].sent.tokens, list_relevances, labels_true[i], labels_pred[i]])           
    
    f.close()


def check_if_subword(token, model_type, starting=False):
    """Check if ``token`` is a subword token that is not a standalone word.  检查 ``token`` 是否是不是独立单词的子词标记。

    Args:
        token (str): token to check.
        model_type (str): type of model (options: "bert", "roberta", "xlnet").
        starting (bool): Should be set ``True`` if this token is the starting token of the overall text.
            This matters because models like RoBERTa does not add "Ġ" to beginning token.
    Returns:
        (bool): ``True`` if ``token`` is a subword token.
    """
    avail_models = [
        "bert",
        "gpt",
        "gpt2",
        "roberta",
        "bart",
        "electra",
        "longformer",
        "xlnet",
    ]
    if model_type not in avail_models:
        raise ValueError(
            f"Model type {model_type} is not available. Options are {avail_models}."
        )
    if model_type in ["bert", "electra"]:
        return True if "##" in token else False
    elif model_type in ["gpt", "gpt2", "roberta", "bart", "longformer"]:
        if starting:
            return False
        else:
            return False if token[0] == "Ġ" else True
    elif model_type == "xlnet":
        return False if token[0] == "_" else True
    else:
        return False


def strip_BPE_artifacts(token, model_type):
    """Strip characters such as "Ġ" that are left over from BPE tokenization.

    Args:
        token (str)
        model_type (str): type of model (options: "bert", "roberta", "xlnet")
    """
    avail_models = [
        "bert",
        "gpt",
        "gpt2",
        "roberta",
        "bart",
        "electra",
        "longformer",
        "xlnet",
        "distilroberta",
    ]
    if model_type not in avail_models:
        raise ValueError(
            f"Model type {model_type} is not available. Options are {avail_models}."
        )
    if model_type in ["bert", "electra"]:
        return token.replace("##", "")
    elif model_type in ["gpt", "gpt2", "roberta", "bart", "distilroberta" "longformer"]:
        return token.replace("Ġ", "")
    elif model_type == "xlnet":
        if len(token) > 1 and token[0] == "_":
            return token[1:]
        else:
            return token
    else:
        return token


def check_if_punctuations(word):
    """Returns ``True`` if ``word`` is just a sequence of punctuations."""
    for c in word:
        if c not in string.punctuation:
            return False
    return True


def is_one_word(word):
    return len(words_from_text(word)) == 1

#
# ppl_model = GPT2LMHeadModel.from_pretrained("gpt2")
# ppl_model.to(textattack.shared.utils.device)
# ppl_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# ppl_model.eval()
# max_length = ppl_model.config.n_positions
# stride = 512


# def calc_ppl(texts):
#     with torch.no_grad():
#         text = " ".join(texts)
#         eval_loss = []
#         input_ids = torch.tensor(
#             ppl_tokenizer.encode(text, add_special_tokens=True)
#         ).unsqueeze(0)
#         # Strided perplexity calculation from huggingface.co/transformers/perplexity.html
#         for i in range(0, input_ids.size(1), stride):
#             begin_loc = max(i + stride - max_length, 0)
#             end_loc = min(i + stride, input_ids.size(1))
#             trg_len = end_loc - i
#             input_ids_t = input_ids[:, begin_loc:end_loc].to(
#                 textattack.shared.utils.device
#             )
#             target_ids = input_ids_t.clone()
#             target_ids[:, :-trg_len] = -100
#
#             outputs = ppl_model(input_ids_t, labels=target_ids)
#             log_likelihood = outputs[0] * trg_len
#
#             eval_loss.append(log_likelihood)
#
#     return torch.exp(torch.stack(eval_loss).sum() / end_loc).item()
#

def process_string(string):
    string = re.sub("( )(\'[(m)(d)(t)(ll)(re)(ve)(s)])", r"\2", string)
    string = re.sub("(\d+)( )([,\.])( )(\d+)", r"\1\3\5", string)
    # U . S . -> U.S.
    string = re.sub("(\w)( )(\.)( )(\w)( )(\.)", r"\1\3\5\7", string)
    # reduce left space
    string = re.sub("( )([,\.!?:;)])", r"\2", string)
    # reduce right space
    string = re.sub("([(])( )", r"\1", string)
    string = re.sub("s '", "s'", string)
    # reduce both space
    string = re.sub("(')( )(\S+)( )(')", r"\1\3\5", string)
    string = re.sub("(\")( )(\S+)( )(\")", r"\1\3\5", string)
    string = re.sub("(\w+) (-+) (\w+)", r"\1\2\3", string)
    string = re.sub("(\w+) (/+) (\w+)", r"\1\2\3", string)
    # string = re.sub(" ' ", "'", string)
    return string