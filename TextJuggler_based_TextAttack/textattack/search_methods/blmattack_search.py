from collections import defaultdict
from typing import List, Dict
import numpy as np
import math
import torch
import torch.nn.functional as F
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
import random
from textattack.util import LSH
from textattack.shared import AttackedText
from transformers import BertForMaskedLM, BertTokenizer
from textattack.BLMinstances import OccludedInstance, InputInstance
from textattack.constraints.semantics.sentence_encoders.universal_sentence_encoder import UniversalSentenceEncoder

use = UniversalSentenceEncoder()
np.random.seed(1111)
random.seed(1111)


class BLMWordOrderWIR(SearchMethod):
    """An attack that greedily chooses from a list of possible perturbations in
    order of index, after ranking indices by importance.
    Args:
        wir_method: method for ranking most important words
        model_wrapper: model wrapper used for gradient-based ranking
    """
    index = 0

    def __init__(self, wir_method,
                 bert_model: str = "bert-base-uncased",
                 cuda_device: int = 0,
                 n_samples: int = 30,
                 verbose: bool = False,
                 scoring_method=lambda x: x,
                 ):
        self.wir_method = wir_method
        bert = BertForMaskedLM.from_pretrained(bert_model)
        bert.eval()
        self.bert = bert.to(cuda_device)
        self.bert = bert
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.cuda_device = cuda_device
        self.n_samples = n_samples
        self.scoring_method = scoring_method
        self.verbose = verbose
        self.vocab_size = len(self.tokenizer.vocab)

    def average_relevance_scoring(self, p_original, p_replaced, n_samples, method):
        # takes a relevance scoring method and applies it to original and samples probabilities
        # output is the difference of original and average of the replaced values
        # return method(p_original) - sum([method(probability)*weight for probability, weight in p_replaced]) / n_samples
        return method(p_original) - sum([method(probability) * weight for probability, weight in p_replaced]) / sum(
            weight for _, weight in p_replaced)

    def relevances(self, candidate_instances, candidate_results) -> Dict[int, Dict[str, float]]:
        """"计算每个单词的相关性分数，通过采样得到的候选例子，以及每一个候选例子的置信度"""
        positional_probabilities = defaultdict(list)
        relevances = defaultdict(float)
        for instance, p_instance in zip(candidate_instances, candidate_results):
            positional_probabilities[instance.occluded_indices].append((p_instance, instance.weight))

        for position, probabilities_weights_tuple_list in positional_probabilities.items():
            # 跳过原始输入语句的相关性分数计算
            if position is None:
                continue

            assert len(positional_probabilities[None]) == 1
            p_original = positional_probabilities[None][0][0]
            relevance = self.average_relevance_scoring(p_original,
                                                       probabilities_weights_tuple_list,
                                                       self.n_samples,
                                                       self.scoring_method)
            relevances[position] = relevance
        return relevances

    def prob_for_index(self,
                       tokens: List[str],
                       index: int,
                       mask_index: bool) -> np.array:

        if mask_index:
            tokens = list(tokens)
            tokens[index] = self.tokenizer.mask_token

        input_token_ids = self.tokenizer.encode(text=tokens,
                                                add_special_tokens=True,
                                                # max_length=512,
                                                truncation=True,
                                                return_tensors="pt").to(self.cuda_device)
        # return_tensors = "pt")

        if self.verbose:
            print(f"Input tokens: {tokens}")
            print(f"Input token ids: {input_token_ids}")

        with torch.no_grad():
            logits = self.bert(input_token_ids)[0]

        probabilities = F.softmax(logits[0, 1:-1], dim=1)[index].detach().cpu().numpy()

        return probabilities

    def get_candidate_instances(self, input_instance: InputInstance) -> List[OccludedInstance]:

        occluded_instances = []
        for field_name, token_field in input_instance.token_fields.items():
            # print(len(token_field.tokens), token_field.tokens)
            for token_idx in range(len(token_field.tokens)):
                use_texts = []
                tokens_with_mask = list(token_field.tokens)
                tokens_with_mask[token_idx] = self.tokenizer.mask_token
                subword_tokens = self.tokenizer.tokenize(" ".join(tokens_with_mask))
                masked_index = subword_tokens.index(self.tokenizer.mask_token)
                # print(f"Subword tokens: {subword_tokens}")
                # print(f"Masked index: {masked_index}")
                if self.verbose:
                    print(f"Subword tokens: {subword_tokens}")
                    print(f"Masked index: {masked_index}")

                p_with_masked_token = self.prob_for_index(subword_tokens,
                                                          index=masked_index,
                                                          mask_index=True)

                # TODO: also make sure that filter subword replacement (seq len original != seq len candidate)
                # sample n_samples times from pt
                # predict and get probability of class c at token
                sampled_token_indices = np.random.choice(self.vocab_size, size=self.n_samples, p=p_with_masked_token)
                # print(sampled_token_indices)
                unique_token_indices, unique_tokens_counts = np.unique(sampled_token_indices, return_counts=True)

                occluded_insts = []
                for unique_token_index, unique_token_count in zip(unique_token_indices, unique_tokens_counts):
                    sampled_token = self.tokenizer.convert_ids_to_tokens([unique_token_index])[0]
                    # assert not sampled_token.startswith("##")

                    tokens_with_replacement = list(token_field.tokens)
                    tokens_with_replacement[token_idx] = sampled_token
                    occluded_inst = OccludedInstance.from_input_instance(
                        input_instance,
                        occlude_token=sampled_token,
                        occlude_field_index=(field_name, token_idx),
                        weight=float(unique_token_count))
                    occluded_insts.append(occluded_inst)
                    # print(occluded_inst)

                    use_texts.append(' '.join(word for word in
                                              self.text_window_around_index(index=masked_index, window_size=10,
                                                                            text=tokens_with_replacement)))
                    idx_to_occluded = {}
                    k = 0
                    for occluded in occluded_insts:
                        idx_to_occluded[str(k)] = occluded
                        k += 1

                # Encode all the generates input texts using sentence encoder
                embeddings = use.encode(use_texts)
                lsh = LSH(512)  # dimension size of USE embeddings is 512

                for t in range(len(embeddings)):
                    lsh.add(embeddings[t], str(t))
                table = lsh.get_result()
                occluded_insts = []
                # Get a random candidate from each bucket
                for key, value in table.table.items():
                    val = random.choice(value)
                    occluded_insts.append(idx_to_occluded[val])
                    # print(occluded_insts)
                occluded_instances.extend(occluded_insts)
        # add original sentence to candidates
        occluded_instances.insert(0, OccludedInstance.from_input_instance(input_instance))

        if self.verbose:
            print(occluded_instances)

        return occluded_instances

    def text_window_around_index(self, index, window_size, text):
        """The text window of ``window_size`` words centered around index."""

        length = len(text)
        half_size = (window_size - 1) / 2.0
        if index - half_size < 0:
            start = 0
            end = min(window_size - 1, length - 1)
        elif index + half_size >= length:
            start = max(0, length - window_size)
            end = length - 1
        else:
            start = index - math.ceil(half_size)
            end = index + math.floor(half_size)
        # text_idx_start = self.text_index_of_word_index(start)
        # text_idx_end = self.text_index_of_word_index(end) + len(self.words[end])
        return text[start:end]

    def _get_index_order(self, initial_text):
        """Returns word indices of ``initial_text`` in descending order of importance."""
        initial_text_words = initial_text.words
        instance = InputInstance(sent=initial_text_words)
        candidate_instances = self.get_candidate_instances(instance)
        attacked_texts = []
        candidate_results_probs = []
        for candidate_instance in candidate_instances:
            token_field = candidate_instance.token_fields['sent']
            candidate_text = (" ".join(list(token_field.tokens)))
            attacked_text = AttackedText(candidate_text)
            attacked_texts.append(attacked_text)
        print(len(attacked_texts))
        candidate_results, search_over = self.get_goal_results(attacked_texts)
        with torch.no_grad():
            for result in candidate_results:
                # prediction = result.raw_output.cpu().numpy().tolist()
                prediction = 1 - result.score
                candidate_results_probs.append(prediction)
                # candidate_results_probs.append(prediction[0])
        res_relevance = self.relevances(candidate_instances, candidate_results_probs)
        list_relevances = []
        for k in range(len(res_relevance)):
            list_relevances.append(res_relevance[('sent', k)])
        index_scores = np.array(list_relevances)
        search_over = False

        if self.wir_method != "random":
            index_order = (-index_scores).argsort()

        return index_order, search_over

    def _perform_search(self, initial_result):

        attacked_text = initial_result.attacked_text
        # Sort words by order of importance
        index_order, search_over = self._get_index_order(attacked_text)
        # print(index_order)
        i = 0
        cur_result = initial_result
        results = None
        while i < len(index_order) and not search_over:
            transformed_text_candidates = self.get_transformations(
                cur_result.attacked_text,
                original_text=initial_result.attacked_text,
                indices_to_modify=[index_order[i]],
            )
            i += 1
            if len(transformed_text_candidates) == 0:
                continue
            results, search_over = self.get_goal_results(transformed_text_candidates)
            results = sorted(results, key=lambda x: -x.score)
            # Skip swaps which don't improve the score
            if results[0].score > cur_result.score:
                cur_result = results[0]
            else:
                continue
            # If we succeeded, return the index with best similarity.
            if cur_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                best_result = cur_result
                # @TODO: Use vectorwise operations
                max_similarity = -float("inf")
                for result in results:
                    if result.goal_status != GoalFunctionResultStatus.SUCCEEDED:
                        break
                    candidate = result.attacked_text
                    try:
                        similarity_score = candidate.attack_attrs["similarity_score"]
                    except KeyError:
                        # If the attack was run without any similarity metrics,
                        # candidates won't have a similarity score. In this
                        # case, break and return the candidate that changed
                        # the original score the most.
                        break
                    if similarity_score > max_similarity:
                        max_similarity = similarity_score
                        best_result = result
                return best_result

        return cur_result


    @property
    def is_black_box(self):
        if self.wir_method == "gradient":
            return False
        else:
            return True

    def extra_repr_keys(self):
        return ["wir_method"]
