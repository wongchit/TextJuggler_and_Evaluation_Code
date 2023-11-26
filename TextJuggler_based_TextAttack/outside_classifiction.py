# import torch
# from textattack.goal_function_results import ClassificationGoalFunctionResult, GoalFunctionResultStatus
# import json
#
# # from google.cloud import language_v1
#
# from allennlp.predictors.predictor import Predictor
# # import allennlp_models.tagging
#
# import fasttext
#
# from aliyunsdkcore.client import AcsClient
# from aliyunsdkcore.acs_exception.exceptions import ClientException
# from aliyunsdkcore.acs_exception.exceptions import ServerException
# from aliyunsdknlp_automl.request.v20191111 import RunPreTrainServiceRequest
#
#
# class OutsideClassification( ):
#     def __init__(self, platform, query_budget):
#         self.num_queries = 0
#         self.platform = platform
#         self.goal_function_type = ClassificationGoalFunctionResult
#         self.search_budget = query_budget
#         if self.platform == 'fasttext':
#             pass
#             self.model = fasttext.load_model('../models/fasttext/yelp_review_polarity.bin')
#             # self.model = fasttext.load_model('../models/fasttext/amazon_review_polarity.bin')
#         elif self.platform == 'allennlp':
#             pass
#             self.model = Predictor.from_path(
#             "https://storage.googleapis.com/allennlp-public-models/basic_stanford_sentiment_treebank-2020.06.09.tar.gz")
#         elif self.platform == 'alibaba':
#             pass
#             # pip install aliyun-python-sdk-core
#             # pip install aliyun-python-sdk-nlp-automl
#             self.client = AcsClient("", "", "cn-hangzhou")
#         elif self.platform == 'google':
#             # self.client = language_v1.LanguageServiceClient()
#             pass
#
#
#
#     def init_attack_example(self, attacked_text, ground_truth_output):
#         """Called before attacking ``attacked_text`` to 'reset' the goal
#         function and set properties for this example."""
#         self.initial_attacked_text = attacked_text
#         self.ground_truth_output = ground_truth_output
#         self.num_queries = 0
#         self.search_over = False
#         if self.platform == 'fasttext':
#             labels, scores = self.fasttext_analyze_sentiment([attacked_text])
#         elif self.platform == 'allennlp':
#             labels, scores = self.allen_nlp_analyze_sentiment([attacked_text])
#         elif self.platform == 'google':
#             labels, scores = self.google_analyze_sentiment([attacked_text])
#         elif self.platform == 'alibaba':
#             labels, scores = self.alibaba_analyze_sentiment([attacked_text])
#
#         if labels[0] != self.ground_truth_output:
#             status = GoalFunctionResultStatus.SKIPPED
#         else:
#             status = GoalFunctionResultStatus.SEARCHING
#         init_result = self.goal_function_type(attacked_text,
#                                               scores[0],  # confidence score
#                                               labels[0],  # label
#                                               status,  # GoalFunctionResultStatus.SUCCEEDED
#                                               scores[0],  # confidence score
#                                               self.num_queries,
#                                               self.ground_truth_output, )
#         return init_result, False
#
#     def get_results(self, attacked_texts):
#         results = []
#         scores = []
#         labels = []
#         self.num_queries += len(attacked_texts)
#         if self.search_budget <= self.num_queries:
#             self.search_over = True
#         if self.platform == 'google':
#             labels, scores = self.google_analyze_sentiment(attacked_texts)
#         elif self.platform == 'fasttext':
#             labels, scores = self.fasttext_analyze_sentiment(attacked_texts)
#         elif self.platform == 'allennlp':
#             labels, scores = self.allen_nlp_analyze_sentiment(attacked_texts)
#         elif self.platform == 'alibaba':
#             labels, scores = self.alibaba_analyze_sentiment(attacked_texts)
#         for i in range(len(scores)):
#             if labels[i] != self.ground_truth_output:
#                 goal_status = GoalFunctionResultStatus.SUCCEEDED
#             else:
#                 goal_status = GoalFunctionResultStatus.SEARCHING
#             results.append(self.goal_function_type(
#                 attacked_texts[i],
#                 scores[i],  # model_output
#                 labels[i],  # label
#                 goal_status,  # GoalFunctionResultStatus.SUCCEEDED
#                 scores[i],  # confidence score
#                 self.num_queries,
#                 self.ground_truth_output, ))
#         return results, self.search_over
#
#     # api
#     # def google_analyze_sentiment(self, attacked_texts):
#     #     """
#     #     Analyzing Sentiment in a String
#     #     Args:
#     #       text_content The text content to analyze
#     #     """
#     #     txts = ''
#     #     documents = []
#     #     for text in attacked_texts:
#     #         if len(txts) + len(text.text) > 980:
#     #             documents.append(txts)
#     #             txts = ''
#     #         else:
#     #             txts = txts + text.text
#     #
#     #     type_ = language_v1.Document.Type.PLAIN_TEXT
#     #
#     #     # Optional. If not specified, the language is automatically detected.
#     #     # For list of supported languages:
#     #     # https://cloud.google.com/natural-language/docs/languages
#     #     language = "en"
#     #     # Available values: NONE, UTF8, UTF16, UTF32
#     #     encoding_type = language_v1.EncodingType.UTF8
#     #     scores = []
#     #     labels = []
#     #     for texts in documents:
#     #         document = {"content": texts, "type_": type_, "language": language}
#     #
#     #         response = self.client.analyze_sentiment(request={'document': document, 'encoding_type': encoding_type})
#     #
#     #         print(u"Document sentiment score: {}".format(response.document_sentiment.score))
#     #         print(
#     #             u"Document sentiment magnitude: {}".format(
#     #                 response.document_sentiment.magnitude
#     #             )
#     #         )
#     #         #
#     #         # Get sentiment for all sentences in the document
#     #         for sentence in response.sentences:
#     #             scores.append(int(sentence.sentiment.score))
#     #             print(u"Sentence text: {}".format(sentence.text.content))
#     #             print(u"Sentence sentiment score: {}".format(sentence.sentiment.score))
#     #             print(u"Sentence sentiment magnitude: {}".format(sentence.sentiment.magnitude))
#     #
#     #         if len(scores) != len(attacked_texts):
#     #             raise Exception('google-api:len(scores)!=len(atk_texts)')
#     #         for attacked_text, score in zip(attacked_texts, scores):
#     #             if score > 0.:
#     #                 labels.append(1)  # 积极
#     #             elif score < 0.:
#     #                 labels.append(0)   # 消极
#     #             else:
#     #                 labels.append(-1)  # 中性
#     #     return scores, labels
#
#     def alibaba_analyze_sentiment(self, attacked_texts):
#         scores, labels = [], []
#         # 正，中，负三种标签
#         messages = [attacked_text.text for attacked_text in attacked_texts]
#         content = {"messages": messages}
#         # Initialize a request and set parameters
#         request = RunPreTrainServiceRequest.RunPreTrainServiceRequest()
#         request.set_ServiceName('NLP-En-Sentiment-Analysis')
#         request.set_PredictContent(json.dumps(content))
#
#         response = self.client.do_action_with_exception(request)
#         resp_obj = json.loads(response)
#         predict_result = json.loads(resp_obj['PredictResult'])
#         print(predict_result['predictions'])
#         for i in range(len(predict_result['predictions'])):
#             scores.append(0.)
#             if predict_result['predictions'][0] == 'negative':
#                 labels.append(0)
#             elif predict_result['predictions'][0] == 'neutral':
#                 labels.append(-1)
#             elif predict_result['predictions'][0] == 'positive':
#                 labels.append(1)
#
#         return scores, labels
#     #
#     # def huawei_analyze_sentiment(self, attacked_texts):
#     #     pass
#
#     # only linux macos
#     def allen_nlp_analyze_sentiment(self, attacked_texts):
#         scores = []
#         labels = []
#         for atk_text in attacked_texts:
#             result = self.model.predict(atk_text.text)
#             # result=[pso,neg]   1pos 0neg
#             labels.append(int(result['label']))
#             result['probs'].reverse()
#             scores.append(self.get_scores(result['probs']))
#         if len(scores) != len(attacked_texts):
#             return [], self.search_over
#         return labels, scores
#     #
#     # # only linux macos
#     def fasttext_analyze_sentiment(self, attacked_texts):
#         scores = []
#         labels = []
#         for atk_text in attacked_texts:
#             result = self.model.predict(atk_text.text)
#             # result_label2-1pos 1-0neg
#             if result[1][0] > 1:
#                 result[1][0] = 1.-1e-5
#             if '1' in result[0][0]:
#                 labels.append(0)
#                 scores.append(self.get_scores((result[1][0],1-result[1][0])))
#             elif '2' in result[0][0]:
#                 labels.append(1)
#                 scores.append(self.get_scores((1-result[1][0], result[1][0])))
#         if len(scores) != len(attacked_texts):
#             return [], self.search_over
#         return labels, scores
#
#     def get_scores(self, outputs):
#         # outputs(0neg scores, 1positive scores)
#         return 1 - outputs[self.ground_truth_output]
#
#
# if __name__ == '__main__':
#     print('invoke_api')
#     texts = []
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print('1')
#     # texts = start_attack.load_data_from_csv('../data/yelp')
#     # texts = texts[:10]
#     # atk_texts = []
#     # for i in range(len(texts)):
#     #     atk_texts.append(AttackedText(texts[i][0]))
#     #     print(str(texts[i][1]))
#     # APIResult(platform='allennlp', search_budget=10000).allen_nlp_analyze_sentiment(atk_texts)
