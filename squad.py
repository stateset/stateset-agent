import json


with open('data/train-v2.0.json', 'r') as f:
    train_data = json.load(f)

train_data = [item for topic in train_data['data'] for item in topic['paragraphs'] ]