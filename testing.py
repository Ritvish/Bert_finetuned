from transformers import BertTokenizer, BertForQuestionAnswering
import torch

def evaluate_model(question):
    # Define the fixed context
    context = "None of your business"

    # Load the fine-tuned model and tokenizer
    model = BertForQuestionAnswering.from_pretrained("./blood_donation_bot")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Tokenize the input (question + fixed context)
    inputs = tokenizer(question, context, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    # Get model output (start_logits and end_logits represent the start and end of the answer span)
    outputs = model(**inputs)  # Pass the tokenized input to the model
    answer_start_scores = outputs.start_logits  # Scores for the start of the answer
    answer_end_scores = outputs.end_logits  # Scores for the end of the answer

    # Find the tokens with the highest start and end scores
    answer_start = torch.argmax(answer_start_scores)  # Index of the start of the answer
    answer_end = torch.argmax(answer_end_scores) + 1  # Index of the end of the answer (end is exclusive, hence +1)

    # Convert token indices back to string using the tokenizer
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    print("Answer:", answer)

if __name__ == "__main__":
    # You can test this by passing a question directly here
    question = "What should i do before donating blood?"
    evaluate_model(question)