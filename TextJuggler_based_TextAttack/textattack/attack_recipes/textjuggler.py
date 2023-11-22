import transformers
from textattack.transformations import WordSwapEmbedding, CompositeTransformation
from textattack.goal_functions import UntargetedClassification
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification, InputColumnModification
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.shared import Attack
from .attack_recipe import AttackRecipe
from textattack.search_methods.blmattack_search import BLMWordOrderWIR
from textattack.transformations.word_swaps.word_swap_masked_lm import WordSwapMaskedLM
from textattack.transformations.word_insertions.word_insertion_masked_lm import WordInsertionMaskedLM


class TextJuggler(AttackRecipe):

    @staticmethod
    def build(model_wrapper):
        shared_masked_lm = transformers.AutoModelForCausalLM.from_pretrained(
            "distilroberta-base", is_decoder=True
        )
        shared_tokenizer = transformers.AutoTokenizer.from_pretrained(
            "distilroberta-base"
        )
        transformation = CompositeTransformation(
            [
                WordSwapMaskedLM(
                    method="bae",
                    masked_language_model=shared_masked_lm,
                    tokenizer=shared_tokenizer,
                    max_candidates=48,
                    min_confidence=5e-4,#min_confidence是一个阈值，如果替换后的词的概率小于这个阈值，就不会被替换
                ),
                WordInsertionMaskedLM(
                    masked_language_model=shared_masked_lm,
                    tokenizer=shared_tokenizer,
                    max_candidates=50,
                    min_confidence=0.0,
                ),
            ]
        )


        # transformation = WordSwapEmbedding(max_candidates=50)
        constraints = []
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )
        constraints.append(input_column_modification)
        # Minimum word embedding cosine similarity of 0.5.
        # (The paper claims 0.7, but analysis of the released code and some empirical
        # results show that it's 0.5.)
        constraints = [RepeatModification(), StopwordModification()]

        # constraints.append(WordEmbeddingDistance(min_cos_sim=0.5))
        #
        # Only replace words with the same part of speech (or nouns with verbs)
        #
        # constraints.append(PartOfSpeech(allow_verb_noun_swap=True))

        # use_constraint = UniversalSentenceEncoder(
        #     threshold=0.7,
        #     metric="cosine",
        #     compare_against_original=True,
        #     window_size=15,
        #     skip_text_shorter_than_window=True,
        # )
        use_constraint = UniversalSentenceEncoder(
            threshold=0.840845057,
            metric="angular",
            compare_against_original=False,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        constraints.append(use_constraint)

        # Goal is untargeted classification
        goal_function = UntargetedClassification(model_wrapper)

        search_method = BLMWordOrderWIR("blm")
        # search_method = GreedyWordSwapWIR(wir_method="random")

        return Attack(goal_function, constraints, transformation, search_method)