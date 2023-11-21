import csv
import torch
import language_tool_python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy
import os
from universal_sentence_encoder import UniversalSentenceEncoder
import re


def process_string(string):
    string = re.sub(r" \'s", "\'s", string)
    string = re.sub(r" \'ve", "\'ve", string)
    string = re.sub(r" n\'t", "n\'t", string)
    string = re.sub(r" \'re", "\'re", string)
    string = re.sub(r" \'d", "\'d", string)
    string = re.sub(r" \'ll", "\'ll", string)
    string = re.sub(r" , ", ",", string)
    string = re.sub(r" ! ", "!", string)
    string = re.sub(r"\(", "(", string)
    string = re.sub(r"\)", ")", string)
    string = re.sub(r"\?", "?", string)
    return string


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


class EvaluateText:
    # Only use one GPU, if we have one.
    # TODO: Running Universal Sentence Encoder uses multiple GPUs
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Disable tensorflow logs, except in the case of an error.
    if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    try:
        # Fix TensorFlow GPU memory growth
        import tensorflow as tf

        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
    except ModuleNotFoundError:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpt_path = os.path.join('./gpt-2')
    tokenizer = GPT2Tokenizer.from_pretrained(gpt_path)
    model = GPT2LMHeadModel.from_pretrained(gpt_path).to(device)
    lang_tool = language_tool_python.LanguageTool("en-US")
    use = UniversalSentenceEncoder()
    sim_metric = torch.nn.CosineSimilarity(dim=1)

    def __init__(self, pre_training_model, recipe, limit_perturb_len, text_len=-1, path='./results'):
        self.csv_path = os.path.join(path, pre_training_model + '_' + recipe + '.csv')
        print(self.csv_path)
        self.pre_model = pre_training_model
        self.recipe = recipe
        self.max_length = self.model.config.n_positions
        self.stride = 512
        self.ini_texts = []
        self.adv_texts = []
        self.queries_num = []
        self.sentences_len = []
        self.sentences_change_len = []
        self.atk_Accuracy = 0
        self.pert_rate = []
        self.ground_truth = []
        self.limit_perturb_len = limit_perturb_len
        if self.limit_perturb_len < 1.:
            self.limit_rate = True
        else:
            self.limit_rate = False
        self.status = []
        self.wash_texts(start_position=0, number=text_len)

    def wash_texts(self, start_position=0, number=-1):
        original_texts = []
        perturbed_texts = []
        result_type = []
        query = []
        ground_truth = []
        fail = 0
        i = 0
        with open(self.csv_path, encoding='UTF-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                original_texts.append(row['original_text'])
                perturbed_texts.append(row['perturbed_text'])
                result_type.append(row['result_type'])
                query.append(float(row['num_queries'].strip()))
                ground_truth.append(int(float(row['ground_truth_output'])))
                i = i + 1
                # print(str(i))
        if number == -1:
            number = len(original_texts)
            self.status = [0] * number
        else:
            self.status = [0] * number
        original_texts = original_texts[start_position:number]
        perturbed_texts = perturbed_texts[start_position:number]
        if 'nli' in self.csv_path:
            original_texts = self.nli_wash(original_texts)
            perturbed_texts = self.nli_wash(perturbed_texts)
        result_type = result_type[start_position:number]
        query = query[start_position:number]
        words_len = []
        for i in range(len(original_texts)):
            if result_type[i] == 'Successful':
                change_len = perturbed_texts[i].count('[[')
                perturbed_texts[i] = perturbed_texts[i].replace('[[', '')
                perturbed_texts[i] = perturbed_texts[i].replace(']]', '')
                original_texts[i] = original_texts[i].replace('[[', '')
                original_texts[i] = original_texts[i].replace(']]', '')
                sentence_len = words_len_from_text(perturbed_texts[i])
                words_len.append(sentence_len)
                pert = change_len / sentence_len
                if (self.limit_rate and self.limit_perturb_len >= pert) or (
                        not self.limit_rate and self.limit_perturb_len >= change_len):
                    self.queries_num.append(query[i])
                    self.sentences_change_len.append(change_len)
                    self.sentences_len.append(sentence_len)
                    self.adv_texts.append(process_string(perturbed_texts[i]))
                    self.ini_texts.append(process_string(original_texts[i]))
                    self.pert_rate.append(pert)
                    self.ground_truth.append(ground_truth[i])
                else:
                    fail += 1
            elif result_type[i] == 'Failed':
                fail += 1
                self.status[i] = 1

        self.atk_Accuracy = len(self.adv_texts) / (len(self.adv_texts) + fail)


    def grammar_error_rate(self):
        gram_num_probs = []
        for i in range(len(self.ini_texts)):
            gram_num_probs.append(
                (len(self.lang_tool.check(self.adv_texts[i])) - len(self.lang_tool.check(self.ini_texts[i]))) /
                self.sentences_len[i])
        return numpy.mean(gram_num_probs)

    def ppl_compute(self, texts):
        # for path in glob.glob('hla_result/*.txt'):
        # lines = filter(lambda x: len(x.strip()) > 0, open(path).readlines())
        # ppls, total_ppl, num_tokens = [], 0, 0
        self.model.eval()
        eval_loss = []
        with torch.no_grad():
            for line in texts:
                input_seq = torch.tensor(self.tokenizer.encode(line)).unsqueeze(0)[:, :1024]
                # if len(input_seq) < 2:
                #     continue
                input_seq = input_seq.cuda()
                output = self.model(input_seq, labels=input_seq)
                lm_loss = output[0]
                eval_loss.append(lm_loss.mean().item())

        # eval_loss = eval_loss/nb_eval_steps
        # perplexity = torch.exp(torch.tensor(eval_loss))
        # return perplexity.item()
        return eval_loss

    def use_sim_metric(self):
        sim = []
        for i in range(len(self.ini_texts)):
            text2 = self.use.encode([self.adv_texts[i]])
            text1 = self.use.encode([self.ini_texts[i]])
            if not isinstance(text1, torch.Tensor):
                text1 = torch.tensor(text1)
            if not isinstance(text2, torch.Tensor):
                text2 = torch.tensor(text2)
            # starting_embedding = torch.unsqueeze(text1, dim=0)
            # transformed_embedding = torch.unsqueeze(text2, dim=0)
            sim.append(self.sim_metric(text1, text2))
        return numpy.mean(sim)

    def nli_wash(self, texts):
        washed_text = []
        split1 = '>>>>[[[[Hypothesis]]]]: '
        for text in texts:
            washed_text.append(text.split(split1)[1])
        return washed_text

    def calc_ppl(self, texts):
        with torch.no_grad():
            text = " ".join(texts)
            eval_loss = []
            input_ids = torch.tensor(
                self.tokenizer.encode(text, add_special_tokens=True)
            ).unsqueeze(0)
            # Strided perplexity calculation from huggingface.co/transformers/perplexity.html
            for i in range(0, input_ids.size(1), self.stride):
                begin_loc = max(i + self.stride - self.max_length, 0)
                end_loc = min(i + self.stride, input_ids.size(1))
                trg_len = end_loc - i
                input_ids_t = input_ids[:, begin_loc:end_loc].to(
                    'cuda'
                )
                target_ids = input_ids_t.clone()
                target_ids[:, :-trg_len] = -100

                outputs = self.model(input_ids_t, labels=target_ids)
                log_likelihood = outputs[0] * trg_len

                eval_loss.append(log_likelihood)

        return torch.exp(torch.stack(eval_loss).sum() / end_loc).item()

    ##################################################1 代表测某指标，0代表不测 ###############################################################
    def eval(self, atk_acc=1, pert_rate=1, gram=1, use=1, ppl_gap=1, query=1):
        print('len:' + str(len(self.ini_texts)))
        if atk_acc:
            print('atk_acc: %.4f' % (self.atk_Accuracy * 100))
        if pert_rate:
            print('pert_rate: %.4f' % (numpy.mean(self.pert_rate) * 100))
        if use:
            print('sim score: %.4f' % self.use_sim_metric())

        if ppl_gap:
            ori = self.calc_ppl(self.ini_texts)
            adv = self.calc_ppl(self.adv_texts)
            print('ori_ppl: %.4f' % ori + '  adv_ppl: %.4f' % adv)
            print('pplgap: %.4f' % (adv - ori))
        if query:
            print('quries: %.4f' % numpy.mean(self.queries_num))

    def pert_word_number(self):
        self.sentences_change_len.sort()
        detail = []
        for i in range(1, 50):
            detail.append(self.sentences_change_len.count(i))
        print(detail)

    def save_pert_texts(self):
        name = self.pre_model + '-' + self.recipe
        with open(name, 'w', encoding='utf-8') as f:
            for text, label in zip(self.adv_texts, self.ground_truth):
                f.write(str(label) + ' ' + text + '\n')


if __name__ == '__main__':
    evaluate = EvaluateText(pre_training_model='bert-base-uncased-ag-news',
                            recipe='textfooler',
                            limit_perturb_len=9999999,
                            text_len=20000, path='./result')
    evaluate.eval()
