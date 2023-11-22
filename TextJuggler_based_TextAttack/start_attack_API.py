import collections
import string
import os

import tensorflow
# import torch

from Constraint import Constraint
from textattack.commands.attack.attack_args_helpers import ARGS_SPLIT_TOKEN
import textattack
import re
import sys
from textattack.goal_functions.classification.untargeted_classification import UntargetedClassification
from textattack.datasets_text_attack import HuggingFaceDataset

import func_timeout

# from outside_classifiction import OutsideClassification

sys.setrecursionlimit(8000)
HUGGINGFACE_DATASET_BY_MODEL = {
    #
    # bert-base-uncased
    #
    "bert-base-uncased-ag-news": (
        "textattack/bert-base-uncased-ag-news",
        ("ag_news", None, "test"),
    ),
    "bert-base-uncased-cola": (
        "textattack/bert-base-uncased-CoLA",
        ("glue", "cola", "validation"),
    ),
    "bert-base-uncased-imdb": (
        "textattack/bert-base-uncased-imdb",
        ("imdb", None, "test"),
    ),
    "bert-base-uncased-mnli": (
        "textattack/bert-base-uncased-MNLI",
        ("glue", "mnli", "validation_matched", [1, 2, 0]),
    ),
    "bert-base-uncased-mrpc": (
        "textattack/bert-base-uncased-MRPC",
        ("glue", "mrpc", "validation"),
    ),
    "bert-base-uncased-qnli": (
        "textattack/bert-base-uncased-QNLI",
        ("glue", "qnli", "validation"),
    ),
    "bert-base-uncased-qqp": (
        "textattack/bert-base-uncased-QQP",
        ("glue", "qqp", "validation"),
    ),
    "bert-base-uncased-rte": (
        "textattack/bert-base-uncased-RTE",
        ("glue", "rte", "validation"),
    ),
    "bert-base-uncased-sst2": (
        "textattack/bert-base-uncased-SST-2",
        ("glue", "sst2", "validation"),
    ),
    "bert-base-uncased-stsb": (
        "textattack/bert-base-uncased-STS-B",
        ("glue", "stsb", "validation", None, 5.0),
    ),
    "bert-base-uncased-wnli": (
        "textattack/bert-base-uncased-WNLI",
        ("glue", "wnli", "validation"),
    ),
    "bert-base-uncased-mr": (
        "textattack/bert-base-uncased-rotten-tomatoes",
        ("rotten_tomatoes", None, "test"),
    ),
    "bert-base-uncased-snli": (
        "textattack/bert-base-uncased-snli",
        ("snli", None, "test", [1, 2, 0]),
    ),
    "bert-base-uncased-yelp": (
        "textattack/bert-base-uncased-yelp-polarity",
        ("yelp_polarity", None, "test"),
    ),
    #
    # distilbert-base-cased
    #
    "distilbert-base-cased-cola": (
        "textattack/distilbert-base-cased-CoLA",
        ("glue", "cola", "validation"),
    ),
    "distilbert-base-cased-mrpc": (
        "textattack/distilbert-base-cased-MRPC",
        ("glue", "mrpc", "validation"),
    ),
    "distilbert-base-cased-qqp": (
        "textattack/distilbert-base-cased-QQP",
        ("glue", "qqp", "validation"),
    ),
    "distilbert-base-cased-snli": (
        "textattack/distilbert-base-cased-snli",
        ("snli", None, "test"),
    ),
    "distilbert-base-cased-sst2": (
        "textattack/distilbert-base-cased-SST-2",
        ("glue", "sst2", "validation"),
    ),
    "distilbert-base-cased-stsb": (
        "textattack/distilbert-base-cased-STS-B",
        ("glue", "stsb", "validation", None, 5.0),
    ),
    #
    # distilbert-base-uncased
    #
    "distilbert-base-uncased-ag-news": (
        "textattack/distilbert-base-uncased-ag-news",
        ("ag_news", None, "test"),
    ),
    "distilbert-base-uncased-cola": (
        "textattack/distilbert-base-cased-CoLA",
        ("glue", "cola", "validation"),
    ),
    "distilbert-base-uncased-imdb": (
        "textattack/distilbert-base-uncased-imdb",
        ("imdb", None, "test"),
    ),
    "distilbert-base-uncased-mnli": (
        "textattack/distilbert-base-uncased-MNLI",
        ("glue", "mnli", "validation_matched", [1, 2, 0]),
    ),
    "distilbert-base-uncased-mr": (
        "textattack/distilbert-base-uncased-rotten-tomatoes",
        ("rotten_tomatoes", None, "test"),
    ),
    "distilbert-base-uncased-mrpc": (
        "textattack/distilbert-base-uncased-MRPC",
        ("glue", "mrpc", "validation"),
    ),
    "distilbert-base-uncased-qnli": (
        "textattack/distilbert-base-uncased-QNLI",
        ("glue", "qnli", "validation"),
    ),
    "distilbert-base-uncased-rte": (
        "textattack/distilbert-base-uncased-RTE",
        ("glue", "rte", "validation"),
    ),
    "distilbert-base-uncased-wnli": (
        "textattack/distilbert-base-uncased-WNLI",
        ("glue", "wnli", "validation"),
    ),
    #
    # roberta-base (RoBERTa is cased by default)
    #
    "roberta-base-ag-news": (
        "textattack/roberta-base-ag-news",
        ("ag_news", None, "test"),
    ),
    "roberta-base-cola": (
        "textattack/roberta-base-CoLA",
        ("glue", "cola", "validation"),
    ),
    "roberta-base-imdb": (
        "textattack/roberta-base-imdb",
        ("imdb", None, "test"),
    ),
    "roberta-base-mr": (
        "textattack/roberta-base-rotten-tomatoes",
        ("rotten_tomatoes", None, "test"),
    ),
    "roberta-base-mrpc": (
        "textattack/roberta-base-MRPC",
        ("glue", "mrpc", "validation"),
    ),
    "roberta-base-qnli": (
        "textattack/roberta-base-QNLI",
        ("glue", "qnli", "validation"),
    ),
    "roberta-base-rte": ("textattack/roberta-base-RTE", ("glue", "rte", "validation")),
    "roberta-base-sst2": (
        "textattack/roberta-base-SST-2",
        ("glue", "sst2", "validation"),
    ),
    "roberta-base-stsb": (
        "textattack/roberta-base-STS-B",
        ("glue", "stsb", "validation", None, 5.0),
    ),
    "roberta-base-wnli": (
        "textattack/roberta-base-WNLI",
        ("glue", "wnli", "validation"),
    ),
    #
    # albert-base-v2 (ALBERT is cased by default)
    #
    "albert-base-v2-ag-news": (
        "textattack/albert-base-v2-ag-news",
        ("ag_news", None, "test"),
    ),
    "albert-base-v2-cola": (
        "textattack/albert-base-v2-CoLA",
        ("glue", "cola", "validation"),
    ),
    "albert-base-v2-imdb": (
        "textattack/albert-base-v2-imdb",
        ("imdb", None, "test"),
    ),
    "albert-base-v2-mr": (
        "textattack/albert-base-v2-rotten-tomatoes",
        ("rotten_tomatoes", None, "test"),
    ),
    "albert-base-v2-rte": (
        "textattack/albert-base-v2-RTE",
        ("glue", "rte", "validation"),
    ),
    "albert-base-v2-qqp": (
        "textattack/albert-base-v2-QQP",
        ("glue", "qqp", "validation"),
    ),
    "albert-base-v2-snli": (
        "textattack/albert-base-v2-snli",
        ("snli", None, "test"),
    ),
    "albert-base-v2-sst2": (
        "textattack/albert-base-v2-SST-2",
        ("glue", "sst2", "validation"),
    ),
    "albert-base-v2-stsb": (
        "textattack/albert-base-v2-STS-B",
        ("glue", "stsb", "validation", None, 5.0),
    ),
    "albert-base-v2-wnli": (
        "textattack/albert-base-v2-WNLI",
        ("glue", "wnli", "validation"),
    ),
    "albert-base-v2-yelp": (
        "textattack/albert-base-v2-yelp-polarity",
        ("yelp_polarity", None, "test"),
    ),
    #
    # xlnet-base-cased
    #
    "xlnet-base-cased-cola": (
        "textattack/xlnet-base-cased-CoLA",
        ("glue", "cola", "validation"),
    ),
    "xlnet-base-cased-imdb": (
        "textattack/xlnet-base-cased-imdb",
        ("imdb", None, "test"),
    ),
    "xlnet-base-cased-mr": (
        "textattack/xlnet-base-cased-rotten-tomatoes",
        ("rotten_tomatoes", None, "test"),
    ),
    "xlnet-base-cased-mrpc": (
        "textattack/xlnet-base-cased-MRPC",
        ("glue", "mrpc", "validation"),
    ),
    "xlnet-base-cased-rte": (
        "textattack/xlnet-base-cased-RTE",
        ("glue", "rte", "validation"),
    ),
    "xlnet-base-cased-stsb": (
        "textattack/xlnet-base-cased-STS-B",
        ("glue", "stsb", "validation", None, 5.0),
    ),
    "xlnet-base-cased-wnli": (
        "textattack/xlnet-base-cased-WNLI",
        ("glue", "wnli", "validation"),
    ),
}
#
# Models hosted by textattack.
#
TEXTATTACK_DATASET_BY_MODEL = {
    #
    # LSTMs
    #
    "lstm-ag-news": (
        "models/classification/lstm/ag-news",
        ("ag_news", None, "test"),
    ),
    "lstm-imdb": ("models/classification/lstm/imdb", ("imdb", None, "test")),
    "lstm-mr": (
        "models/classification/lstm/mr",
        ("rotten_tomatoes", None, "test"),
    ),
    "lstm-sst2": ("models/classification/lstm/sst2", ("glue", "sst2", "validation")),
    "lstm-yelp": (
        "models/classification/lstm/yelp",
        ("yelp_polarity", None, "test"),
    ),
    #
    # CNNs
    #
    "cnn-ag-news": (
        "models/classification/cnn/ag-news",
        ("ag_news", None, "test"),
    ),
    "cnn-imdb": ("models/classification/cnn/imdb", ("imdb", None, "test")),
    "cnn-mr": (
        "models/classification/cnn/rotten-tomatoes",
        ("rotten_tomatoes", None, "test"),
    ),
    "cnn-sst2": ("models/classification/cnn/sst", ("glue", "sst2", "validation")),
    "cnn-yelp": (
        "models/classification/cnn/yelp",
        ("yelp_polarity", None, "test"),
    ),
    #
    # T5 for translation
    #
    "t5-en-de": (
        "english_to_german",
        ("textattack.datasets.translation.TedMultiTranslationDataset", "en", "de"),
    ),
    "t5-en-fr": (
        "english_to_french",
        ("textattack.datasets.translation.TedMultiTranslationDataset", "en", "fr"),
    ),
    "t5-en-ro": (
        "english_to_romanian",
        ("textattack.datasets.translation.TedMultiTranslationDataset", "en", "de"),
    ),
    #
    # T5 for summarization
    #
    "t5-summarization": ("summarization", ("gigaword", None, "test")),
}


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
    if 'snli' in filepath:
        labeldict = {"contradiction": 0,
                     "entailment": 1,
                     "neutral": 2}
    elif 'mnli' in filepath:
        labeldict = {'0': 1, '1': 2, '2': 0}
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
        return data


def load_data_from_csv(dataset_path):
    texts, labels = read_corpus(dataset_path)
    data = list(zip(texts, labels))
    print(str(len(data)))
    return data


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


def attack(model, dataset, dataset_len, recipe, offset=0, goalfunction=None, len_number=-1, number_append=5,
           saved_log=True, constraint=True, failed_list=[]):
    # Only use one GPU, if we have one.
    # TODO: Running Universal Sentence Encoder uses multiple GPUs
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Disable tensorflow logs, except in the case of an error.
    if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    model_name = model + '-' + dataset
    if saved_log:
        STDOUT = open(model_name + '_' + recipe + '.txt', 'a', encoding='utf-8')
        sys.stdout = STDOUT
        sys.stderr = STDOUT

    if goalfunction[0] == 'untargeted':
        model_info = create_model_form_path(model_name)
        model_wrapper = UntargetedClassification(model_info, query_budget=goal_function[1])
    # elif goalfunction[0] == 'outside':
    #     model_wrapper = OutsideClassification(platform=model_name.split('-')[0], query_budget=goal_function[1])
    else:
        raise Exception('wrong goal_function!')

    if dataset == 'sst2':
        dataset = HuggingFaceDataset('glue', 'sst2', 'train')
        dataset = dataset[:dataset_len]
    elif dataset == 'snli' or dataset == 'mnli':
        dataset = read_data('./data/' + dataset)
    else:
        dataset = load_data_from_csv('./data/' + dataset)


    if recipe == 'textfooler':
        attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)
    elif recipe == 'textbugger':
        attack = textattack.attack_recipes.TextBuggerLi2018.build(model_wrapper)
    elif recipe == 'pwws':
        attack = textattack.attack_recipes.PWWSRen2019.build(model_wrapper)
    elif recipe == 'pso':
        attack = textattack.attack_recipes.PSOZang2020.build(model_wrapper)
    elif recipe == 'TextJuggler':
        attack = textattack.attack_recipes.TextJuggler.build(model_wrapper)
    elif recipe == 'olm':
        attack = textattack.attack_recipes.OLMAttack.build(model_wrapper)
    elif recipe == 'bae':
        attack = textattack.attack_recipes.BAEGarg2019.build(model_wrapper)
    elif recipe == 'bertattack':
        attack = textattack.attack_recipes.BERTAttackLi2020.build(model_wrapper)

    else:
        print('wrong recipe!')
        sys.exit()
    while True:
        # break
        try:
            # attack_dataset1(offset,attack,dataset,worklist,model_name,recipe,number_append)

            if constraint:
                if len_number == -1:
                    worklist = collections.deque(range(offset, len(dataset)))
                else:
                    worklist = collections.deque(range(offset, offset + len_number))
            else:
                worklist = collections.deque(failed_list)
                Constraint.constraint = False

            i = 0
            header = True
            filename = os.path.join('./output/tweet', model_name + '_' + recipe + '.csv')

            # filename = now_time + '-' + model_name + '_' + attackmethod + '.csv'
            CSVLogger = textattack.loggers.CSVLogger(filename=filename)
            # CSVLogger = textattack.loggers.CSVLogger(filename=model_name + '_' + recipe + '.csv')

            for result in attack.attack_dataset(dataset, indices=worklist):
                i = i + 1
                offset = offset + 1
                if 'Successful' == result.__class__.__name__.replace("AttackResult", ""):
                    print('index: ' + str(offset - 1))
                elif 'Failed' == result.__class__.__name__.replace("AttackResult", ""):
                    print('index: ' + str(offset - 1) + 'Failed!')
                CSVLogger.log_attack_result(result)
                if i >= number_append:
                    if header:
                        header = False
                        CSVLogger.flush_append(header=True)
                    else:
                        CSVLogger.flush_append()
                    i = 0
                sys.stdout.flush()

            if offset == dataset_len:
                break
        except:
            print(str(offset) + "error,execute next")
            offset = offset + 1
            continue

    if saved_log:
        STDOUT.close()


def attack_dataset1(offset, attack, dataset, worklist, model_name, recipe, number_append):
    i = 0
    header = True
    CSVLogger = textattack.loggers.CSVLogger(filename=model_name + '_' + recipe + '.csv')
    for result in attack.attack_dataset(dataset, indices=worklist):
        i = i + 1
        offset = offset + 1
        if 'Successful' == result.__class__.__name__.replace("AttackResult", ""):
            print('index: ' + str(offset - 1))
        elif 'Failed' == result.__class__.__name__.replace("AttackResult", ""):
            print('index: ' + str(offset - 1) + 'Failed!')
        CSVLogger.log_attack_result(result)
        if i >= number_append:
            if header:
                header = False
                CSVLogger.flush_append(header=True)
            else:
                CSVLogger.flush_append()
            i = 0

        sys.stdout.flush()


if __name__ == '__main__':
    failed_list = []
    # device = cpu
    # if len(goal_function_result.attacked_text.words) == 1:
    #     print('go')
    #     yield FailedAttackResult(goal_function_result)
    # outside/untargeted 1000/float("inf")
    goal_function = ('untargeted', float("inf"))
    # attack(model='distilbert-base-uncased', dataset='mnli', dataset_len=600, recipe='bertattack', offset=490,
    #       goalfunction=goal_function, len_number=600, number_append=1,
    #      saved_log=False, constraint=True, failed_list=failed_list)
    attack(model='cnn', dataset='imdb', dataset_len=300, recipe='textjuggler', offset=0,
           goalfunction=goal_function, len_number=10, number_append=1,
           saved_log=False, constraint=True, failed_list=failed_list)
#albert-base-v2-mr_bertattack
#cnn  ag-news

