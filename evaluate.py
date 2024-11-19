
# evaluate.py
from transformers import BertTokenizer, BertForQuestionAnswering
import torch

def evaluate_model():
    # Load the fine-tuned model and tokenizer
    model = BertForQuestionAnswering.from_pretrained("./blood_donation_bot")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Example question and context
    question = "What is blood donation?"
    context = ""

    # Tokenize input
    inputs = tokenizer(question, context, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    # Get model output
    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    # Find the tokens with the highest start and end scores
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    # Convert tokens to answer
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    print("Answer:", answer)