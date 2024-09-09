from llm2vec_copy.llm2vec import LLM2Vec

from transformers import TrainingArguments, Trainer
from datasets import load_dataset

import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoConfig, AutoModel, AutoTokenizer, DataCollatorWithPadding

class ModelForSentenceSimilarity(PreTrainedModel):
    def __init__(self, config, model):
        super().__init__(config)
        self.model = model
        
        classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        # Assume hidden states from both sentences are pooled (e.g., [CLS] token output)
#         self.regression = nn.Linear(config.hidden_size * 2, 1)  # Combining two sentence embeddings
        self.similarity = nn.CosineSimilarity(dim=1)  # or another appropriate similarity measure


    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, labels=None):
        
        sentence_feature_1 = {
            'input_ids': input_ids_1,
            'attention_mask': attention_mask_1
        }
        outputs_1 = self.model(sentence_feature=sentence_feature_1)

        sentence_feature_2 = {
            'input_ids': input_ids_2,
            'attention_mask': attention_mask_2
        }
        outputs_2 = self.model(sentence_feature=sentence_feature_2)
        
#         outputs_1 = self.model(input_ids=input_ids_1, attention_mask=attention_mask_1)
#         outputs_2 = self.model(input_ids=input_ids_2, attention_mask=attention_mask_2)
        
#         print(outputs_1.shape)  # Expected to be something like [batch_size, features]
#         print(outputs_2.shape)  # Expected to be something like [batch_size, features]


#         pooled_output_1 = outputs_1[1]  # Assuming that outputs[1] is the pooled output
#         pooled_output_2 = outputs_2[1]

#         print(pooled_output_1.shape)  # Expected to be something like [batch_size, features]
#         print(pooled_output_2.shape)  # Expected to be something like [batch_size, features]

                # Assuming the first element of outputs is the last hidden state
#         pooled_output_1 = outputs_1.last_hidden_state[:, 0]  # usually the [CLS] token
#         pooled_output_2 = outputs_2.last_hidden_state[:, 0]
        
        pooled_output_1 = self.dropout(outputs_1)
        pooled_output_2 = self.dropout(outputs_2)
        
        similarity_scores = self.similarity(pooled_output_1, pooled_output_2)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss(reduction='mean')  # or another appropriate regression loss
            loss = loss_fct(similarity_scores, labels.view(-1))

#         print("Similarity scores:", similarity_scores.shape)
#         print("Labels:", labels.shape)
#         print("Loss value calculated:", loss.item())  # should not error out if loss is scalar
        
        return (loss, similarity_scores) if labels is not None else similarity_scores
#         return (similarity_scores, loss) if labels is not None else similarity_scores


training_args = TrainingArguments(
    output_dir='./results_trained',  # Directory where the model predictions and checkpoints will be written.
    num_train_epochs=10,      # Total number of training epochs to perform.
    per_device_train_batch_size=16,  # Batch size per device during training
    per_device_eval_batch_size=64,   # Batch size for evaluation
    warmup_steps=500,                # Number of steps to perform learning rate warmup.
    weight_decay=0.01,               # Strength of weight decay
    learning_rate=5e-4,
    logging_dir='./logs',            # Directory for storing logs
    logging_steps=50,                # Log every X updates steps.
    do_train=True,                   # Whether to run training.
    do_eval=True,                    # Whether to run eval on the dev set.
    evaluation_strategy="steps",     # Evaluation is done (and saved) every X steps
    save_strategy="steps",           # Checkpoints are saved every X steps
    save_steps=500,                  # Save checkpoint every 500 steps
    eval_steps=500,                  # Evaluate model every 500 steps
    load_best_model_at_end=True,     # Whether to load the best model found at each evaluation.
    remove_unused_columns=False
)
    
model_name_or_path = "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp"
peft_addr          = "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-supervised"

access_token = "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"
l2v = LLM2Vec.from_pretrained(
    model_name_or_path,
    peft_model_name_or_path=peft_addr,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    enable_bidirectional=True,
    torch_dtype=torch.bfloat16,
    token=access_token,
)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# dataset = load_dataset("copenlu/spiced", split='train')
# dataset = load_dataset("copenlu/spiced")

dataset = load_dataset("copenlu/spiced", split='train').select(range(50))
eval_dataset = load_dataset("copenlu/spiced", split='validation').select(range(20))

def preprocess_function(examples):
    # Tokenize the pairs of sentences to get the tensors
#     encodings = tokenizer(examples['News Finding'], examples['Paper Finding'], truncation=True, padding=True, return_tensors="pt")

    # Tokenize the pairs of sentences separately
    tokenized_inputs_1 = tokenizer(examples['News Finding'], truncation=True, padding='max_length', max_length=256, return_tensors="pt")
    tokenized_inputs_2 = tokenizer(examples['Paper Finding'], truncation=True, padding='max_length', max_length=256, return_tensors="pt")
    
    
    min_score, max_score = 1.0, 5.0
    normalize = lambda x: (x - min_score) / (max_score - min_score)
    labels = list(map(normalize, examples["final_score"]))
    labels = torch.tensor(labels)
    labels = labels.to(torch.bfloat16)
#     encodings['labels'] = torch.tensor(scores)

#     return {
#         'input_ids': encodings['input_ids'],
#         'attention_mask': encodings['attention_mask'],
#         'labels': torch.tensor(scores)
#     }

    return {
        'input_ids_1': tokenized_inputs_1['input_ids'],
        'attention_mask_1': tokenized_inputs_1['attention_mask'],
        'input_ids_2': tokenized_inputs_2['input_ids'],
        'attention_mask_2': tokenized_inputs_2['attention_mask'],
        'labels': labels
    }
    

#     return encodings

# tokenized_datasets = dataset.map(preprocess_function, batched=True)
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
eval_tokenized_datasets = eval_dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

class CustomDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        # First, separate the inputs for the two sentence embeddings
        first_sentences = [{key.replace("_1", ""): value for key, value in feature.items() if "_1" in key} for feature in features]
        second_sentences = [{key.replace("_2", ""): value for key, value in feature.items() if "_2" in key} for feature in features]

        # Use the parent class method to pad each list separately
        batch1 = super().__call__(first_sentences)
#         batch1 = {k: v.to(torch.bfloat16) for k, v in batch1.items() if isinstance(v, torch.Tensor)}

        batch2 = super().__call__(second_sentences)
#         batch2 = {k: v.to(torch.bfloat16) for k, v in batch2.items() if isinstance(v, torch.Tensor)}

        # Reassemble the inputs into the expected format for the model
        batch = {f"{key}_1": batch1[key] for key in batch1}
        batch.update({f"{key}_2": batch2[key] for key in batch2})

        # Handle labels if they exist
        if 'labels' in features[0]:
            batch['labels'] = torch.tensor([f['labels'] for f in features], dtype=torch.float32)

        return batch

data_collator = CustomDataCollator(tokenizer=tokenizer)

config = AutoConfig.from_pretrained(model_name_or_path, num_labels=1)

model = ModelForSentenceSimilarity(config, model=l2v)
# Set the model to use BFloat16 if it's supported and activated in your model configuration
model = model.to(torch.bfloat16)

class CustomTrainer(Trainer):
    def __init__(self, *args, data_collator=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure the custom data collator is used if not provided
        self.data_collator = CustomDataCollator(tokenizer=self.tokenizer)

    def training_step(self, model, inputs):
        # Ensure inputs are converted to BFloat16 if required
        inputs = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v for k, v in inputs.items()}
        
        # Print input data types for debugging
#         print("Input data types:", {k: v.dtype for k, v in inputs.items()})
        
        # Forward pass: Get model outputs and ensure outputs are in BFloat16
        outputs = model(**inputs)
#         print("Model output type:", outputs[0].dtype if isinstance(outputs, tuple) else outputs.dtype)
        
        # Check labels dtype and ensure consistency
        labels = inputs.get('labels')
        if labels is not None:
            labels = labels.to(torch.bfloat16)
#             print("Labels type:", labels.dtype)
        
        # Calculate loss and ensure it's in BFloat16
        loss = outputs[0] if isinstance(outputs, tuple) else outputs
#         print("Loss type before backward:", loss.dtype)
#         print("Loss value before backward:", loss.item())
        
        # Backward pass
        self.accelerator.backward(loss)

        return loss
    
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=eval_tokenized_datasets,  # Add this line
#     data_collator=CustomDataCollator(tokenizer=tokenizer),
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=lambda p: {'mse': ((p.predictions - p.label_ids) ** 2).mean().item()}  # Simple MSE for regression
)
for param in model.parameters():
    param.requires_grad = True
for name, param in model.named_parameters():
    if not param.requires_grad:
        print(f"Parameter {name} with shape {param.shape} does not require gradients.")
    assert param.requires_grad, f"All parameters should have requires_grad=True, but {name} does not."

# Train the model
trainer.train()

trainer.save_model("./results_trained")
