from llm2vec_copy.llm2vec import LLM2Vec

from datasets import load_dataset

import torch

import numpy as np

print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

dataset = load_dataset("copenlu/spiced", split='train')
# .select(range(160))
test_dataset = load_dataset("copenlu/spiced", split='test')
# .select(range(20))

sentences_train_1 = dataset["News Finding"]
sentences_train_2 = dataset["Paper Finding"]

sentences_test_1 = test_dataset["News Finding"]
sentences_test_2 = test_dataset["Paper Finding"]

min_score, max_score = 1.0, 5.0
normalize = lambda x: (x - min_score) / (max_score - min_score)
y_train = list(map(normalize, dataset["final_score"]))
y_test = list(map(normalize, test_dataset["final_score"]))

print("Loading model...")
access_token = "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"
model = LLM2Vec.from_pretrained(
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
#     "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp",
    peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
#     peft_model_name_or_path="McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-supervised",
    device_map="cuda:0" if torch.cuda.is_available() else "cpu",
    enable_bidirectional=True,
    torch_dtype=torch.bfloat16,
    token=access_token,
)

def append_instruction(instruction, sentences):
    new_sentences = []
    for s in sentences:
        new_sentences.append([instruction, s, 0])
    return new_sentences

batch_size = 8

instruction = "Retrieve semantically similar text: "

print(f"Encoding training sentences...")
sentences_train_1 = append_instruction(instruction, sentences_train_1)
X_train_1 = np.asarray(model.encode(sentences_train_1, batch_size=batch_size))

sentences_train_2 = append_instruction(instruction, sentences_train_2)
X_train_2 = np.asarray(model.encode(sentences_train_2, batch_size=batch_size))

X_train = np.concatenate((X_train_1, X_train_2), axis=1)

print(f"Encoding test sentences...")
sentences_test_1 = append_instruction(instruction, sentences_test_1)
X_test_1 = np.asarray(model.encode(sentences_test_1, batch_size=batch_size))

sentences_test_2 = append_instruction(instruction, sentences_test_2)
X_test_2 = np.asarray(model.encode(sentences_test_2, batch_size=batch_size))

X_test = np.concatenate((X_test_1, X_test_2), axis=1)

# --------------------------------------------------------------------------------

# from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, r2_score

# rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
gb_regressor = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)

gb_regressor.fit(X_train, y_train)

# Predict on test set
y_pred = gb_regressor.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')

import joblib

# Save the trained model to a file
joblib.dump(gb_regressor, 'gb_regressor_model.pkl')

# --------------------------------------------------------------------------------
import pandas as pd
import ast
from tqdm import tqdm
tqdm.pandas()

df = pd.read_csv("df_low_ims_single.tsv", sep='\t',index_col=0,)
df['result_sentences'] = df['result_sentences'].progress_apply(ast.literal_eval)

single_strings = df['context'].values
list_of_strings = df['result_sentences'].values

torch.cuda.empty_cache()

similarity_list = []

for sentences1, sentences2 in zip(single_strings, list_of_strings):
    
    sentences1 = append_instruction(instruction, [sentences1])
    embeddings1 = np.asarray(model.encode(sentences1, batch_size=1))
    
    # print(f"Encoding {len(list_of_string)} sentences1...")
    sentences2 = append_instruction(instruction, sentences2)
    embeddings2 = np.asarray(model.encode(sentences2, batch_size=batch_size))
  
    # Calculate cosine similarities
    # Since q_reps is 1x2048 and d_reps is nx2048, we can use broadcasting
    embeddings1 = np.broadcast_to(embeddings1, (embeddings2.shape[0], embeddings2.shape[1]))
    
    X_input = np.concatenate((embeddings1, embeddings2), axis=1)
    y_pred = gb_regressor.predict(X_input)

    sim_score = y_pred * 4 + 1
    
    similarity_list.append(sim_score.tolist())

df['new_similarity_list'] = similarity_list
df['new_max_similarity'] = df['new_similarity_list'].progress_apply(lambda x: max(x))
df['new_max_index'] = df['new_similarity_list'].progress_apply(lambda x: x.index(max(x)))
df['new_the_sentence'] = df.progress_apply(lambda x: x['result_sentences'][x['new_max_index']], axis=1)

df.to_csv("df_low_ims_single_with_sim_aug_12.tsv",sep="\t")


