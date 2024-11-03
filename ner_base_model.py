import logging
import os
import sys
import copy
from dataclasses import dataclass
from dataclasses import field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn import model_selection
import pandas as pd
from datasets import Dataset, DatasetDict, ClassLabel, Sequence, concatenate_datasets
from sklearn.model_selection import KFold

import transformers
from datasets import load_dataset
from seqeval.metrics import accuracy_score
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import recall_score
from seqeval.metrics import classification_report
from transformers.training_args import TrainingArguments
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    pipeline,
    set_seed
)

# from google.colab import output
from IPython.display import Markdown
from IPython.display import display

def process_sentence(sentence):
    tokens = []
    ner_tags = []
    for line in sentence.split('\n'):
        if line.strip():  # Skip empty lines
            try:
                token, tag = line.split()
                tokens.append(token)
                ner_tags.append(label2id[tag])
            except:
                print("======================== Skipping numbers ==============")
                print(line)

    return ' '.join(tokens), tokens, ner_tags


def create_df_from_file(file_path):
    # Open the file and read its contents
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()

    # Process the data
    sentences = data.strip().split('\n\n')
    processed_data = [process_sentence(sentence) for sentence in sentences]

    # Create a DataFrame
    df = pd.DataFrame(processed_data, columns=['text', 'tokens', 'ner_tags'])

    return df

def evaluate_scandi_model(model_path, test_data_path):
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    # Create NER pipeline
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="max")
    # Load and preprocess test data
    test_df = create_df_from_file(test_data_path)
    # Lists to store predictions and true labels
    all_true_labels = []
    all_pred_labels = []
    for _, row in test_df.iterrows():
        text = " ".join(row['tokens'])
        true_labels = row['ner_tags']
        # Get predictions from the model
        predictions = nlp(text)
        # Convert predictions to BIO format
        pred_labels = ['O'] * len(row['tokens'])
        for pred in predictions:
            if pred['entity_group'] in ['PER', 'LOC', 'ORG']:
                # Get word index from the start and end character positions
                start_token_idx = None
                end_token_idx = None
                curr_length = 0

                for idx, token in enumerate(row['tokens']):
                    token_length = len(token) + 1  # +1 for space
                    if curr_length <= pred['start'] < curr_length + token_length:
                        start_token_idx = idx
                    if curr_length <= pred['end'] <= curr_length + token_length:
                        end_token_idx = idx
                    curr_length += token_length

                if start_token_idx is not None and end_token_idx is not None:
                    # Set the labels
                    pred_labels[start_token_idx] = f'B-{pred["entity_group"]}'
                    for i in range(start_token_idx + 1, end_token_idx + 1):
                        pred_labels[i] = f'I-{pred["entity_group"]}'
        # for pred in predictions:
        #     if pred['entity_group'] in ['PER', 'LOC', 'ORG']:  # Changed from 'entity' to 'entity_group'
        #         # Get word index from the start and end character positions
        #         start_token_idx = 0
        #         curr_length = 0
        #         for idx, token in enumerate(row['tokens']):
        #             if curr_length >= pred['start']:
        #                 start_token_idx = idx
        #                 break
        #             curr_length += len(token) + 1  # +1 for space
        #         end_token_idx = start_token_idx
        #         curr_length = sum(len(t) + 1 for t in row['tokens'][:start_token_idx])
        #         for idx in range(start_token_idx, len(row['tokens'])):
        #             curr_length += len(row['tokens'][idx]) + 1
        #             if curr_length > pred['end']:
        #                 end_token_idx = idx
        #                 break
        #             end_token_idx = idx
        #         # Set the labels
        #         pred_labels[start_token_idx] = f'B-{pred["entity_group"]}'
        #         for i in range(start_token_idx + 1, end_token_idx + 1):
        #             pred_labels[i] = f'I-{pred["entity_group"]}'
        # Filter true labels to only include PER, LOC, ORG
        filtered_true = [label_list[label] if label_list[label][2:] in ['PER', 'LOC', 'ORG'] else 'O'
                        for label in true_labels]
        all_true_labels.append(filtered_true)
        all_pred_labels.append(pred_labels)
    # Generate classification report
    report = classification_report(all_true_labels, all_pred_labels, digits=4)
    report_dict = classification_report(all_true_labels, all_pred_labels, digits=4, output_dict=True)
    # Convert to DataFrame for easier analysis
    df_report = pd.DataFrame(report_dict).transpose()
    return report, df_report
# Usage

outf = "annotations/BIO/BIO_synthetic_spacebased_3_classes.txt"  
test_data_path = 'annotations/BIO/BIO_synthetic_spacebased.txt'
label_list = [
    "O",
    "B-PER", "I-PER",
    "B-LOC", "I-LOC",
    "B-ORG", "I-ORG",

]

with open(test_data_path, "r") as infile:  
    with open(outf, "w") as outfile:  
        for fline in infile:  
            if len(fline.strip()) != 0 :
                ec = fline.split(" ")[-1].strip() 
                
                if ec not in label_list:
                    fline = fline.replace(ec, "O")

            outfile.write(fline)
            
model_path = "bert_data/nbailab-base-ner-scandi"
label2id = {label: i for i, label in enumerate(label_list)}
report, df_report = evaluate_scandi_model(model_path, outf)
print("Classification Report:")
print(report)
# Save results
df_report.to_csv("scandi_baseline_results.csv")
# Per-entity analysis
print("\nPer-entity analysis:")
for entity in ['PER', 'LOC', 'ORG']:
    entity_type = f"{entity}"  # Get the full label name
    if entity_type in df_report.index:
        print(f"\nMetrics for {entity}:")
        print(f"Precision: {df_report.loc[entity_type]['precision']:.4f}")
        print(f"Recall: {df_report.loc[entity_type]['recall']:.4f}")
        print(f"F1-score: {df_report.loc[entity_type]['f1-score']:.4f}")
        print(f"Support: {df_report.loc[entity_type]['support']}")