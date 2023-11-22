from .word_swap import WordSwap
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from textattack.shared import utils

class WordSwapSampling(WordSwap):
    '''
    利用MH采样的方式来向初始对抗样本移动
    '''
    def __init__(
        self,
        method="bae",
        masked_language_model="bert-base-uncased",
        tokenizer=None,
        max_length=512,
        max_candidates=50,
        min_confidence=5e-4,
        batch_size=16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.method = method
        self.max_length = max_length
        self.max_candidates = max_candidates
        self.min_confidence = min_confidence
        self.batch_size = batch_size

        if isinstance(masked_language_model, str):
            self._language_model = AutoModelForMaskedLM.from_pretrained(
                masked_language_model
            )
            self._lm_tokenizer = AutoTokenizer.from_pretrained(
                masked_language_model, use_fast=True
            )
        else:
            self._language_model = masked_language_model
            if tokenizer is None:
                raise ValueError(
                    "`tokenizer` argument must be provided when passing an actual model as `masked_language_model`."
                )
            self._lm_tokenizer = tokenizer
        self._language_model.to(utils.device)
        self._language_model.eval()
        self.masked_lm_name = self._language_model.__class__.__name__


    def _encode_text(self, text):
        """Encodes ``text`` using an ``AutoTokenizer``, ``self._lm_tokenizer``.

        Returns a ``dict`` where keys are strings (like 'input_ids') and
        values are ``torch.Tensor``s. Moves tensors to the same device
        as the language model.
        """
        encoding = self._lm_tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {k: v.to(utils.device) for k, v in encoding.items()}

    def _bae_replacement_words(self, current_text, indices_to_modify):
        """Get replacement words for the word we want to replace using BAE
        method.

        Args:
            current_text (AttackedText): Text we want to get replacements for.
            index (int): index of word we want to replace
        """
        masked_texts = []
        for index in indices_to_modify:
            masked_texts.append(
                current_text.replace_word_at_index(
                    index, self._lm_tokenizer.mask_token
                ).text
            )

        i = 0
        # 2-D list where for each index to modify we have a list of replacement words
        replacement_words = []
        replacement_words_probs =[]
        while i < len(masked_texts):
            inputs = self._encode_text(masked_texts[i : i + self.batch_size])
            ids = inputs["input_ids"].tolist()
            with torch.no_grad():
                preds = self._language_model(**inputs)[0]

            for j in range(len(ids)):
                try:
                    # Need try-except b/c mask-token located past max_length might be truncated by tokenizer
                    masked_index = ids[j].index(self._lm_tokenizer.mask_token_id)
                except ValueError:
                    replacement_words.append([])
                    continue

                mask_token_logits = preds[j, masked_index]
                mask_token_probs = torch.softmax(mask_token_logits, dim=0)
                ranked_indices = torch.argsort(mask_token_probs, descending=True)
                top_words = []
                top_words_probs=[]
                for _id in ranked_indices:
                    _id = _id.item()
                    token = self._lm_tokenizer.convert_ids_to_tokens(_id)
                    if utils.check_if_subword(
                        token,
                        self._language_model.config.model_type,
                        (masked_index == 1),
                    ):
                        word = utils.strip_BPE_artifacts(
                            token, self._language_model.config.model_type
                        )
                        if (
                            mask_token_probs[_id] >= self.min_confidence
                            and utils.is_one_word(word)
                            and not utils.check_if_punctuations(word)
                        ):
                            top_words.append(word)
                            top_words_probs.append(mask_token_probs[_id])

                    if (
                        len(top_words) >= self.max_candidates
                        or mask_token_probs[_id] < self.min_confidence
                    ):
                        break

                replacement_words.append(top_words)
                replacement_words_probs.append(top_words_probs)


            i += self.batch_size

        return replacement_words,replacement_words_probs

    def _get_transformations(self, current_text, indices_to_modify):
        indices_to_modify = list(indices_to_modify)
        if self.method == "bae":
            replacement_words = self._bae_replacement_words(
                current_text, indices_to_modify
            )
            transformed_texts = []
            for i in range(len(replacement_words)):
                index_to_modify = indices_to_modify[i]
                word_at_index = current_text.words[index_to_modify]
                for word in replacement_words[i]:
                    if word != word_at_index and len(utils.words_from_text(word)) == 1:
                        transformed_texts.append(
                            current_text.replace_word_at_index(index_to_modify, word)
                        )
            return transformed_texts
        else:
            raise ValueError(f"Unrecognized value {self.method} for `self.method`.")







