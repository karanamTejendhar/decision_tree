# import os
# import spacy
# from transformers import AutoModelForQuestionAnswering, AutoTokenizer
# model_name = "distilbert-base-cased-distilled-squad"
#
# QA_model = model_name
# tokenizer = AutoTokenizer.from_pretrained(model_name)
#
# from components import QueryProcessor, DocumentRetrieval, PassageRetrieval, AnswerExtractor
# nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser', 'textcat'])
#
#
#
# query_processor = QueryProcessor(nlp)
# document_retriever = DocumentRetrieval()
# passage_retriever = PassageRetrieval(nlp)
# answer_extractor = AnswerExtractor(QA_model, QA_model)
#
#
# # question = "describe table and it features?"
# print("Enter the question")
# question = input()
#
# query = query_processor.generate_query(question)
# docs = document_retriever.search(query)
# passage_retriever.fit(docs)
# passages = passage_retriever.most_similar(question)
# answers = answer_extractor.extract(question, passages)
#
# i=0
# for ans in answers:
#     # print(ans)
#     if i!=3:
#         score = ans['score']
#         start = ans['start']
#         end = ans['end']
#         text = ans['text']
#         print(text)
#         # print(ans['answer'])
#         print("------------------------------")
#         i =i+1

import wikipedia as wiki

from components import DocumentReader
reader = DocumentReader("deepset/bert-base-cased-squad2")


questions = {
    'How many wheels in a bicycle?',
}

for question in questions:
    print(f"Question: {question}")
    results = wiki.search(question)

    # for i in range(len(results)):
    #     print(results[i])
    page = wiki.page(results[0],auto_suggest=False)
    print(f"Top wiki result: {page}")

    text = page.content

    reader.tokenize(question, text)
    print(f"Answer: {reader.get_answer()}")
    print()