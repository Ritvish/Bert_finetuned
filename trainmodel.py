from transformers import BertForQuestionAnswering, Trainer, TrainingArguments
from dataprepare import load_data

def train_model():
    # Load the BERT model for question answering
    model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
    train_dataset = load_data()

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
    )

    # Initialize Trainer API with model and data
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    # Start training
    trainer.train()
    
    # Save the trained model
    model.save_pretrained("./blood_donation_bot")