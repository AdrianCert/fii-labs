import pandas as pd
import numpy as np
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from pathlib import Path
import requests


def download_data():
    coqa = pd.read_json(
        "http://downloads.cs.stanford.edu/nlp/data/coqa/coqa-train-v1.0.json"
    )
    coqa.head()
    del coqa["version"]

    # required columns in our dataframe
    cols = ["text", "question", "answer"]  # list of lists to create our dataframe
    comp_list = []
    for index, row in coqa.iterrows():
        for i in range(len(row["data"]["questions"])):
            temp_list = []
            temp_list.append(row["data"]["story"])
            temp_list.append(row["data"]["questions"][i]["input_text"])
            temp_list.append(row["data"]["answers"][i]["input_text"])
            comp_list.append(temp_list)
    new_df = pd.DataFrame(
        comp_list, columns=cols
    )  # saving the dataframe to csv file for further loading
    new_df.to_csv("CoQA_data.csv", index=False)


def get_data():
    dataset_file = Path(__file__, "../CoQA_data.csv")
    if not dataset_file.exists():
        download_data()
    data = pd.read_csv()
    return data


model = BertForQuestionAnswering.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)
tokenizer = BertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)


def translate_to_romanian(text):
    url = "https://api.mymemory.translated.net/get"
    params = {"q": text, "langpair": "en|ro"}

    try:
        response = requests.get(url, params=params)
        response_data = response.json()
        if (
            "responseData" in response_data
            and "translatedText" in response_data["responseData"]
        ):
            return response_data["responseData"]["translatedText"]
        else:
            return None
    except Exception as e:
        print(f"Error during translation: {e}")
        return None


def question_answer(question, text):
    # tokenize question and text as a pair
    input_ids = tokenizer.encode(question, text)

    # string version of tokenized ids
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # segment IDs
    # first occurence of [SEP] token
    sep_idx = input_ids.index(
        tokenizer.sep_token_id
    )  # number of tokens in segment A (question)
    num_seg_a = sep_idx + 1  # number of tokens in segment B (text)
    num_seg_b = len(input_ids) - num_seg_a

    # list of 0s and 1s for segment embeddings
    segment_ids = [0] * num_seg_a + [1] * num_seg_b
    assert len(segment_ids) == len(input_ids)

    # model output using input_ids and segment_ids
    output = model(
        torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids])
    )

    # reconstructing the answer
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)
    if answer_end >= answer_start:
        answer = tokens[answer_start]
        for i in range(answer_start + 1, answer_end + 1):
            if tokens[i][0:2] == "##":
                answer += tokens[i][2:]
            else:
                answer += " " + tokens[i]

    if answer.startswith("[CLS]"):
        answer = "Unable to find the answer to your question."

    print("\nPredicted answer:\n{}".format(answer.capitalize()))

    # Translate answer to Romanian using LibreTranslate API
    translated_answer = translate_to_romanian(answer)
    if translated_answer:
        print("\nTranslated answer in Romanian:\n{}".format(translated_answer))


# get_data()

# text = input("Please enter your text: \n")
text = Path(__file__, "../context.txt").read_text()
question = input("\nPlease enter your question: \n")
while True:
    question_answer(question, text)

    flag = True
    flag_N = False

    while flag:
        response = input(
            "\nDo you want to ask another question based on this text (Y/N)? "
        )
        if response[0] == "Y":
            question = input("\nPlease enter your question: \n")
            flag = False
        elif response[0] == "N":
            print("\nBye!")
            flag = False
            flag_N = True

    if flag_N == True:
        break
