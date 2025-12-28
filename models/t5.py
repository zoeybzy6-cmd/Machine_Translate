import os
import torch
import json
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
from nltk.translate.bleu_score import corpus_bleu

# Use "google/mt5-small" for multilingual tasks (Chinese -> English)
MODEL_NAME = "google/mt5-small"

def train_t5(args):
    """
    Fine-tune mT5 using Hugging Face Trainer.
    Optimized for larger datasets (100k samples).
    """
    print(f"Loading mT5 model: {MODEL_NAME}...")
    
    # Use legacy=False to avoid warnings; AutoTokenizer automatically selects the best class
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, legacy=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # 1. Load Data
    # Mapping files to dataset splits
    data_files = {
        "train": os.path.join(args.data_dir, f'train_{args.dataset_size}.jsonl'),
        "validation": os.path.join(args.data_dir, 'valid.jsonl'),
    }
    # Load dataset from JSONL files
    dataset = load_dataset("json", data_files=data_files)

    # 2. Preprocess Data
    # Although mT5 is multilingual, adding a prefix often helps convergence
    prefix = "translate Chinese to English: "
    max_input_length = 200
    max_target_length = 200

    def preprocess_function(examples):
        """
        Tokenize inputs and targets, and handle padding for loss calculation.
        """
        # Add prefix to source text
        inputs = [prefix + zh for zh in examples["zh"]]
        targets = examples["en"]
        
        # Tokenize inputs
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")
        
        # Tokenize targets
        labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length")

        # [CRITICAL STEP for T5/mT5]
        # Replace the padding token ID in labels with -100.
        # PyTorch CrossEntropyLoss ignores indices set to -100, preventing the model 
        # from learning to predict padding tokens (which ruins performance).
        labels_with_ignore_index = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] 
            for label in labels["input_ids"]
        ]

        model_inputs["labels"] = labels_with_ignore_index
        return model_inputs

    print("Preprocessing data...")
    # Apply preprocessing to the entire dataset
    # 'remove_columns' deletes the original text columns ('zh', 'en') to leave only tensors
    tokenized_datasets = dataset.map(
        preprocess_function, 
        batched=True, 
        remove_columns=dataset["train"].column_names 
    )
    print("pad_token_id:", tokenizer.pad_token_id)
    print("sample labels head:", tokenized_datasets["train"][0]["labels"][:30])
    print("valid label count:", sum([1 for x in tokenized_datasets["train"][0]["labels"] if x != -100]))

    # 3. Define Metrics
    def compute_metrics(eval_preds):
        """
        Calculate BLEU score during validation.
        """
        preds, labels = eval_preds
        
        # If preds is a tuple (loss, logits), take the logits
        if isinstance(preds, tuple):
            preds = preds[0]
        
        # Decode predictions
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        # Replace -100 in labels with pad_token_id to decode correctly
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Simple post-processing: tokenize by splitting on whitespace
        decoded_preds = [pred.strip().split() for pred in decoded_preds]
        decoded_labels = [[label.strip().split()] for label in decoded_labels]

        result = corpus_bleu(decoded_labels, decoded_preds)
        return {"bleu": result * 100}

    # 4. Setup Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(args.ckpt_dir, 't5_results'),
        evaluation_strategy="epoch",    # Evaluate at the end of each epoch
        save_strategy="epoch",          # Save checkpoint at the end of each epoch
        learning_rate=args.lr,          # T5 usually works well with 1e-4 or 3e-4
        per_device_train_batch_size=16, # Adjust based on GPU VRAM (16 or 32 for small models)
        per_device_eval_batch_size=32,
        save_total_limit=2,             # Only keep the last 2 checkpoints to save disk space
        num_train_epochs=args.epochs,
        predict_with_generate=True,     # Enable generation for metric calculation during eval
        generation_max_length=200,
        generation_num_beams=1, # Enable Mixed Precision (FP16) for speed and memory efficiency
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,    # Load the best model (by validation loss) after training
    )

    # 5. Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=compute_metrics,
    )

    print("Starting T5 Training...")
    trainer.train()
    
    # Save the best model and tokenizer
    save_path = os.path.join(args.ckpt_dir, 't5_best')
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path) # Important: save tokenizer to ensure consistency
    print(f"T5 Model saved to {save_path}")


def test_t5(args):
    """
    Test T5 model using Batch Inference.
    Batch inference is significantly faster than single-sample inference loop.
    """
    # Determine model path: use fine-tuned checkpoint if available, otherwise base model
    ft_path = os.path.join(args.ckpt_dir, 't5_best')
    model_path = ft_path if os.path.exists(ft_path) else MODEL_NAME
    print(f"Loading T5 for testing from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    # Load Test Data directly using 'datasets' library
    test_file = os.path.join(args.data_dir, 'test.jsonl')
    dataset = load_dataset("json", data_files={"test": test_file})["test"]

    print(f"Evaluating T5 on {len(dataset)} test sentences...")

    # Batch parameters
    BATCH_SIZE = 32
    hypotheses = []
    references = []

    # Iterate over the dataset in batches
    for i in range(0, len(dataset), BATCH_SIZE):
        # Slice the dataset to get a batch
        batch = dataset[i : i + BATCH_SIZE]
        
        # Add prefix
        inputs = [f"translate Chinese to English: {zh}" for zh in batch['zh']]
        targets = batch['en']

        # Batch Tokenization: Pad to the longest sequence in the batch
        input_ids = tokenizer(
            inputs, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=200
        ).input_ids.to(model.device)

        # Batch Generation
        with torch.no_grad():
            outputs = model.generate(
                input_ids, 
                max_length=200, 
                num_beams=4,        # Beam search for better quality
                early_stopping=True
            )
        
        # Batch Decode
        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Store results for BLEU calculation
        for pred, ref in zip(decoded_preds, targets):
            hypotheses.append(pred.strip().split())
            references.append([ref.strip().split()])

        # Logging progress
        if (i // BATCH_SIZE + 1) % 10 == 0:
            print(f"Processed {min(i + BATCH_SIZE, len(dataset))} sentences...")

    # Calculate final BLEU score
    score = corpus_bleu(references, hypotheses)
    print(f"T5 BLEU Score: {score*100:.2f}")