import json
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

# Step 1: Load your JSON dataset
with open('D:\\asmnt\\scraped_data.json', 'r') as f:
    data = json.load(f)

# Step 2: Extract text from nested structure
texts = []
for entry in data:
    if isinstance(entry, dict) and "content" in entry:
        # If 'p' is present, join the list of paragraphs into a single text
        if "p" in entry["content"]:
            text = " ".join(entry["content"]["p"]).strip()
            if text:  # Ensure the text is not empty
                texts.append({"text": text})

# Ensure data is not empty
if not texts:
    raise ValueError("No valid text found in the JSON data.")

# Debugging: Print the first few processed entries
print("Processed texts:", texts[:5])

# Step 3: Create a Hugging Face dataset
dataset = Dataset.from_list(texts)

# Step 4: Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Add a padding token (using EOS token as a padding token)
tokenizer.pad_token = tokenizer.eos_token  # or tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Debugging: Print the first entry's text before tokenization
print("Text to be tokenized:", texts[0]["text"])

# Step 5: Define the tokenization function with labels
def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()  # Set labels same as input_ids
    return tokenized_inputs

# Step 6: Apply the tokenization
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Debugging: Check if the columns were created successfully
print("Columns in the tokenized dataset:", tokenized_dataset.column_names)

# Step 7: Convert dataset to PyTorch tensors
tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# Step 8: Load the GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Step 9: Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # Changed to the new eval_strategy
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Step 10: Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # You can split for training/eval sets separately
)

# Step 11: Train the model
trainer.train()

# Save the model
trainer.save_model("./fine_tuned_gpt2")

# Optional: Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Step 12: Inference - Testing the fine-tuned model
# Example input text for generation
input_text = "best hospital for cancer"

# Tokenize the input text
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate text using the fine-tuned model
output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

# Decode and print the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Generated text: {generated_text}")
