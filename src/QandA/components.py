# import itertools
# import operator
# import re
# import concurrent.futures
#
# import requests
# import wikipedia
# from gensim.summarization.bm25 import BM25   #pip install gensim==3.8.3
# from transformers import AutoTokenizer, AutoModelForQuestionAnswering, QuestionAnsweringPipeline
#
#
# class QueryProcessor:
#
#     def __init__(self, nlp, keep=None):
#         self.nlp = nlp
#         self.keep = keep or {'PROPN', 'NUM', 'VERB', 'NOUN', 'ADJ'}
#
#     def generate_query(self, text):
#         doc = self.nlp(text)
#         query = ' '.join(token.text for token in doc if token.pos_ in self.keep)
#         return query
#
#
# class DocumentRetrieval:
#
#     def __init__(self, url='https://en.wikipedia.org/w/api.php'):
#         self.url = url
#
#     def search_pages(self, query):
#         params = {
#             'action': 'query',
#             'list': 'search',
#             'srsearch': query,
#             'format': 'json'
#         }
#         res = requests.get(self.url, params=params)
#         return res.json()
#
#     def search(self, query):
#         pages = self.search_pages(query)
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             process_list = [executor.submit(self.search_page, page['pageid']) for page in pages['query']['search']]
#             docs = [self.post_process(p.result()) for p in process_list]
#         return docs
#
#     def search_page(self, page_id):
#         res = wikipedia.page(pageid=page_id)
#         return res.content
#
#     def post_process(self, doc):
#         pattern = '|'.join([
#             '== References ==',
#             '== Further reading ==',
#             '== External links',
#             '== See also ==',
#             '== Sources ==',
#             '== Notes ==',
#             '== Further references ==',
#             '== Footnotes ==',
#             '=== Notes ===',
#             '=== Sources ===',
#             '=== Citations ===',
#         ])
#         p = re.compile(pattern)
#         indices = [m.start() for m in p.finditer(doc)]
#         if len(indices) is not 0 :
#             min_idx = min(*indices, len(doc))
#             return doc[:min_idx]
#         else:
#             return " "
#
# class PassageRetrieval:
#
#     def __init__(self, nlp):
#         self.tokenize = lambda text: [token.lemma_ for token in nlp(text)]
#         self.bm25 = None
#         self.passages = None
#
#     def preprocess(self, doc):
#         passages = [p for p in doc.split('\n') if p and not p.startswith('=')]
#         return passages
#
#     def fit(self, docs):
#         passages = list(itertools.chain(*map(self.preprocess, docs)))
#         corpus = [self.tokenize(p) for p in passages]
#         # print(corpus)
#         self.bm25 = BM25(corpus)
#         self.passages = passages
#
#     def most_similar(self, question, topn=3):
#         tokens = self.tokenize(question)
#         scores = self.bm25.get_scores(tokens)
#         pairs = [(s, i) for i, s in enumerate(scores)]
#         pairs.sort(reverse=True)
#         passages = [self.passages[i] for _, i in pairs[:topn]]
#         return passages
#
#
# class AnswerExtractor:
#
#     def __init__(self, tokenizer, model):
#         tokenizer = AutoTokenizer.from_pretrained(tokenizer)
#         model = AutoModelForQuestionAnswering.from_pretrained(model)
#         self.nlp = QuestionAnsweringPipeline(model=model, tokenizer=tokenizer)
#
#     def extract(self, question, passages):
#         answers = []
#         for passage in passages:
#             try:
#                 answer = self.nlp(question=question, context=passage)
#                 answer['text'] = passage
#                 answers.append(answer)
#             except KeyError:
#                 pass
#         answers.sort(key=operator.itemgetter('score'), reverse=True)
#         return answers



import torch
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


class DocumentReader:
    def __init__(self, pretrained_model_name_or_path='bert-large-uncased'):
        self.READER_PATH = pretrained_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.READER_PATH)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.READER_PATH)
        self.max_len = self.model.config.max_position_embeddings
        self.chunked = False

    def tokenize(self, question, text):
        self.inputs = self.tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")
        self.input_ids = self.inputs["input_ids"].tolist()[0]

        if len(self.input_ids) > self.max_len:
            self.inputs = self.chunkify()
            self.chunked = True

    def chunkify(self):
        """
        Break up a long article into chunks that fit within the max token
        requirement for that Transformer model.

        Calls to BERT / RoBERTa / ALBERT require the following format:
        [CLS] question tokens [SEP] context tokens [SEP].
        """

        # create question mask based on token_type_ids
        # value is 0 for question tokens, 1 for context tokens
        qmask = self.inputs['token_type_ids'].lt(1)
        qt = torch.masked_select(self.inputs['input_ids'], qmask)
        chunk_size = self.max_len - qt.size()[0] - 1  # the "-1" accounts for
        # having to add an ending [SEP] token to the end

        # create a dict of dicts; each sub-dict mimics the structure of pre-chunked model input
        chunked_input = OrderedDict()
        for k, v in self.inputs.items():
            q = torch.masked_select(v, qmask)
            c = torch.masked_select(v, ~qmask)
            chunks = torch.split(c, chunk_size)

            for i, chunk in enumerate(chunks):
                if i not in chunked_input:
                    chunked_input[i] = {}

                thing = torch.cat((q, chunk))
                if i != len(chunks) - 1:
                    if k == 'input_ids':
                        thing = torch.cat((thing, torch.tensor([102])))
                    else:
                        thing = torch.cat((thing, torch.tensor([1])))

                chunked_input[i][k] = torch.unsqueeze(thing, dim=0)
        return chunked_input

    def get_answer(self):
        if self.chunked:
            answer = ''
            for k, chunk in self.inputs.items():
                answer_start_scores, answer_end_scores = self.model(**chunk)
                answer_start = torch.argmax(answer_start_scores)
                answer_end = torch.argmax(answer_end_scores) + 1

                ans = self.convert_ids_to_string(chunk['input_ids'][0][answer_start:answer_end])
                if ans != '[CLS]':
                    answer += ans + " / "
            return answer
        else:
            answer_start_scores, answer_end_scores = self.model(**self.inputs)

            answer_start = torch.argmax(answer_start_scores)  # get the most likely beginning of answer with the argmax of the score
            answer_end = torch.argmax(answer_end_scores) + 1  # get the most likely end of answer with the argmax of the score

            return self.convert_ids_to_string(self.inputs['input_ids'][0][
                                              answer_start:answer_end])

    def convert_ids_to_string(self, input_ids):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids))