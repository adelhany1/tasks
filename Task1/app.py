from flask import Flask, request, jsonify
from transformers import BertModel, BertTokenizer
import torch
from scipy.spatial.distance import cosine

app = Flask(__name__)

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


#encode text to get embeddings
def encode(text):
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=32)
    with torch.no_grad():
        output = model(**encoded_input)
    embeddings = output.last_hidden_state[:, 0, :]
    return embeddings

#calculate cosine similarity between two names
def name_similarity(name1, name2):
    embedding1 = encode(name1)
    embedding2 = encode(name2)
    embedding1 = embedding1.squeeze().numpy()
    embedding2 = embedding2.squeeze().numpy()
    similarity = 1 - cosine(embedding1, embedding2)
    return similarity

@app.route('/similarity', methods=['POST'],endpoint='calculate_similarity')
def get_similarity():
    data = request.get_json()
    name1 = data['name1']
    name2 = data['name2']
    similarity = name_similarity(name1, name2)
    return jsonify({'similarity': similarity})

if __name__ == '__main__':
    app.run(port=9090, debug=True)
    
    