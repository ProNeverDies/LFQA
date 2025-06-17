import os
import json
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from transformers import (
    BartTokenizerFast,
    BartForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    pipeline,
)


PREPROCESSED_DIR = "./preprocessed_data"
CLEAN_CSV_PATH = os.path.join(PREPROCESSED_DIR, "cleaned_qa_data.csv")
MODEL_DIR = "./bart_lfqa_model_fixed"
FEEDBACK_JSON = os.path.join(MODEL_DIR, "feedback.json")
TRAIN_SPLIT_PATH = os.path.join(PREPROCESSED_DIR, "train_split.csv")
TEST_SPLIT_PATH = os.path.join(PREPROCESSED_DIR, "test_split.csv")

PRETRAINED_MODEL = "facebook/bart-base"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 512
BATCH_SIZE = 2           
NUM_TRAIN_EPOCHS = 3     
RETRAIN_EPOCHS = 2       

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(MODEL_DIR, exist_ok=True)


def load_and_split_dataset(test_size=0.1, random_seed=42):
    """
    Load the cleaned CSV and split into train/test CSVs if not already done.
    """
    if os.path.exists(TRAIN_SPLIT_PATH) and os.path.exists(TEST_SPLIT_PATH):
        print(f"Loading existing train/test splits from {PREPROCESSED_DIR}")
        train_df = pd.read_csv(TRAIN_SPLIT_PATH)
        test_df = pd.read_csv(TEST_SPLIT_PATH)
    else:
        print("Loading cleaned dataset and creating train/test split...")
        df = pd.read_csv(CLEAN_CSV_PATH)
        
        if "question_clean" not in df.columns or "answer_clean" not in df.columns:
            raise ValueError("CSV must contain 'question_clean' and 'answer_clean' columns.")
        train_df, test_df = train_test_split(
            df[["question_clean", "answer_clean"]],
            test_size=test_size,
            random_state=random_seed,
            shuffle=True
        )
        train_df.to_csv(TRAIN_SPLIT_PATH, index=False)
        test_df.to_csv(TEST_SPLIT_PATH, index=False)
        print(f"Created train split ({len(train_df)} samples) and test split ({len(test_df)} samples).")
    return train_df, test_df

class QADataset(Dataset):
    """
    Torch Dataset for QA pairs (question -> answer) to feed into BART.
    """
    def __init__(self, df: pd.DataFrame, tokenizer: BartTokenizerFast,
                 max_input_length: int, max_target_length: int):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        question = str(self.df.loc[idx, "question_clean"])
        answer = str(self.df.loc[idx, "answer_clean"])

        model_inputs = self.tokenizer(
            question,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                answer,
                max_length=self.max_target_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        input_ids = model_inputs["input_ids"].squeeze()
        attention_mask = model_inputs["attention_mask"].squeeze()
        labels_ids = labels["input_ids"].squeeze()
        labels_ids[labels_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels_ids,
        }

def get_model_and_tokenizer():
    """
    Load (or instantiate) the BART model and tokenizer.
    """
    tokenizer = BartTokenizerFast.from_pretrained(PRETRAINED_MODEL)
    model = BartForConditionalGeneration.from_pretrained(PRETRAINED_MODEL)
    model.to(DEVICE)
    return model, tokenizer

def train_bart_model(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str,
                     epochs: int = NUM_TRAIN_EPOCHS):
    """
    Fine-tune BART on the provided train_df. Validation will be done manually
    after training (since older TrainingArguments lack built-in eval support).
    Saves the model & tokenizer to output_dir.
    """
    print("Initializing tokenizer and model...")
    model, tokenizer = get_model_and_tokenizer()

    print("Preparing datasets...")
    train_dataset = QADataset(train_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    test_dataset  = QADataset(test_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        do_train=True,
        # do_eval=True,        
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=epochs,
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=test_dataset,  
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()
    print("Training completed. Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model & tokenizer saved to {output_dir}")

def load_feedback():
    """
    Load existing feedback JSON (list of {"question":..., "improvement":...}).
    If not present, return an empty list.
    """
    if os.path.exists(FEEDBACK_JSON):
        with open(FEEDBACK_JSON, "r", encoding="utf-8") as f:
            feedback_list = json.load(f)
        print(f"Loaded {len(feedback_list)} feedback items.")
    else:
        feedback_list = []
    return feedback_list

def save_feedback(feedback_list):
    """
    Save feedback_list to FEEDBACK_JSON.
    """
    with open(FEEDBACK_JSON, "w", encoding="utf-8") as f:
        json.dump(feedback_list, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(feedback_list)} feedback items to {FEEDBACK_JSON}.")

def retrain_on_feedback(base_model_dir: str, feedback_list: list, epochs: int = RETRAIN_EPOCHS):
    """
    Fine-tune the existing model on feedback (incorrect QA → improved answer).
    Append these pairs to a temporary DataFrame, fine-tune for a few epochs,
    and save the updated model back to base_model_dir.
    """
    if not feedback_list:
        print("No feedback found. Nothing to retrain.")
        return

    # Convert feedback list into a DataFrame
    fb_df = pd.DataFrame([
        {"question_clean": entry["question"], "answer_clean": entry["improvement"]}
        for entry in feedback_list
    ])

    train_df, test_df = load_and_split_dataset()

    combined_train_df = pd.concat([train_df, fb_df], ignore_index=True)
    print(f"Retraining on {len(fb_df)} feedback examples + {len(train_df)} original examples = {len(combined_train_df)} total.")

    print("Loading existing model for retraining...")
    model = BartForConditionalGeneration.from_pretrained(base_model_dir).to(DEVICE)
    tokenizer = BartTokenizerFast.from_pretrained(base_model_dir)

    # Create datasets
    train_dataset = QADataset(combined_train_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    test_dataset = QADataset(test_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=base_model_dir,
        overwrite_output_dir=False,           
        do_train=True,
        do_eval=True,
        evaluate_during_training=True,        
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=epochs,
        logging_steps=50,
        save_steps=200,
        eval_steps=200,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        learning_rate=2e-5,                  
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Starting feedback-based fine-tuning...")
    trainer.train()
    print("Fine-tuning completed. Saving updated model...")
    trainer.save_model(base_model_dir)
    tokenizer.save_pretrained(base_model_dir)
    print(f"Retrained model saved back to {base_model_dir}")


def interactive_qa_session(model_dir: str):
    """
    Run an interactive session:
      - Ask user for a question
      - Model generates an answer
      - User marks 'correct' or 'incorrect'
      - If incorrect, user provides an improvement
      - Save feedback to FEEDBACK_JSON
      - At end of session, prompt to retrain on feedback
    """
    
    print("Loading model for interactive session...")
    model = BartForConditionalGeneration.from_pretrained(model_dir).to(DEVICE)
    tokenizer = BartTokenizerFast.from_pretrained(model_dir)
    gen_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if DEVICE == "cuda" else -1
    )

    feedback_list = load_feedback()
    session_feedback = []  

    print("\n==> INTERACTIVE QA SESSION <==")
    print("Type 'quit' to return to menu.\n")

    while True:
        question = input("Enter your question: ").strip()
        if question.lower() in ["quit", "exit"]:
            break
        if not question:
            print("Please type a non-empty question.")
            continue

        print("\n[Model is thinking...]")
        gen_output = gen_pipeline(
            question,
            max_length=MAX_TARGET_LENGTH,
            num_beams=4,
            early_stopping=True
        )
        answer = gen_output[0]["generated_text"].strip()
        print(f"\nModel Answer:\n{answer}\n")

        correctness = input("Is the answer correct? (yes/no): ").strip().lower()
        if correctness not in ["yes", "no"]:
            print("Please respond with 'yes' or 'no'. Assuming 'no'.")
            correctness = "no"

        if correctness == "no":
            improvement = input("Please provide a correct/improved answer:\n").strip()
            if improvement:
                fb_entry = {
                    "question": question,
                    "generated_answer": answer,
                    "improvement": improvement,
                    "timestamp": pd.Timestamp.now().isoformat()
                }
                session_feedback.append(fb_entry)
                print("Feedback saved for retraining.")
            else:
                print("No improvement provided; skipping.")

        else:
            print("Great! Answer marked correct.")

        print("\n--- Ask another question or type 'quit' to finish session ---\n")

    if session_feedback:
        all_feedback = feedback_list + session_feedback
        save_feedback(all_feedback)
    else:
        print("No new feedback collected this session.")

    if session_feedback:
        retrain_now = input("Retrain model on collected improvements now? (yes/no): ").strip().lower()
        if retrain_now == "yes":
            all_feedback = load_feedback()
            retrain_on_feedback(model_dir, all_feedback)
        else:
            print("Skipping retraining. You can retrain later from the main menu.")

def test_model(model_dir: str, test_df: pd.DataFrame):
    """
    Evaluate the model on the test split with a few random samples.
    """
    print("Loading model for testing...")
    model = BartForConditionalGeneration.from_pretrained(model_dir).to(DEVICE)
    tokenizer = BartTokenizerFast.from_pretrained(model_dir)
    gen_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if DEVICE == "cuda" else -1
    )

    print("\n--> MODEL TESTING <--")
    sample_df = test_df.sample(min(5, len(test_df))).reset_index(drop=True)
    for idx, row in sample_df.iterrows():
        question = row["question_clean"]
        gold_answer = row["answer_clean"]
        print(f"\n[Sample {idx+1}]")
        print(f"Question: {question}")
        gen_output = gen_pipeline(
            question,
            max_length=MAX_TARGET_LENGTH,
            num_beams=4,
            early_stopping=True
        )
        pred_answer = gen_output[0]["generated_text"].strip()
        print(f"Model Answer: {pred_answer}")
        print(f"Gold Answer:  {gold_answer}")
        print("\n")
        print("\n")


def main_menu():
    """
    Display an interactive menu for training, testing, retraining, or exit.
    """
    while True:
        print("\n")
        print("       BART LFQA MODEL MANAGEMENT MENU       ")
        print("\n")
        print("1) Train model from scratch")
        print("2) Test model / sample QA {requires trained model}")
        print("3) Retrain on user improvements {if any feedback exists}")
        print("4) Interactive QA session {collect feedback}")
        print("5) Exit")
        choice = input("Choose an option (1-5): ").strip()

        if choice == "1":
            
            train_df, test_df = load_and_split_dataset()
            train_bart_model(train_df, test_df, MODEL_DIR, epochs=NUM_TRAIN_EPOCHS)

        elif choice == "2":
            
            if not os.path.exists(os.path.join(MODEL_DIR, "config.json")) \
               and not os.path.exists(os.path.join(MODEL_DIR, "pytorch_model.bin")):
                print("No trained model found. Please train first.")
            else:
                _, test_df = load_and_split_dataset()
                test_model(MODEL_DIR, test_df)

        elif choice == "3":
            
            feedback_list = load_feedback()
            if not feedback_list:
                print("No feedback to retrain on. Run an interactive QA session first.")
            else:
                retrain_on_feedback(MODEL_DIR, feedback_list, epochs=RETRAIN_EPOCHS)

        elif choice == "4":
            
            if not os.path.exists(os.path.join(MODEL_DIR, "config.json")) \
               and not os.path.exists(os.path.join(MODEL_DIR, "pytorch_model.bin")):
                print("No trained model found. Please train first.")
            else:
                interactive_qa_session(MODEL_DIR)

        elif choice == "5":
            print("Exiting. Goodbye!")
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 5.")


if __name__ == "__main__":
    
    if not os.path.exists(CLEAN_CSV_PATH):
        raise FileNotFoundError(f"Cannot find {CLEAN_CSV_PATH}. Please run preprocessing first.")
    main_menu()
