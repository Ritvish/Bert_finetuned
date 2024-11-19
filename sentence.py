from sentence_transformers import SentenceTransformer
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys

# Initialize the BERT QA model and tokenizer
model = BertForQuestionAnswering.from_pretrained("./blood_donation_bot")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load a pre-trained Sentence-BERT model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Your predefined contexts
contexts = [
        "Blood donation is a process where a donor voluntarily has blood drawn for medical needs.",
        "Anyone in good health, usually aged 18-65, who meets health requirements can donate blood.",
        "Healthy individuals can donate whole blood every 56 days, or up to 6 times a year.",
        "Blood donation is safe, and all equipment is sterilized and used only once.",
        "Donating blood helps save lives and can provide a sense of fulfillment.",
        "It's recommended not to donate if you have a cold, as it may affect donation quality.",
        "Stay hydrated, eat iron-rich foods, and get adequate sleep before donating.",
        "A balanced meal with iron-rich foods, like spinach and beans, is beneficial before donating.",
        "Mild weakness may occur but typically fades quickly after donating.",
        "Inform the blood center about any medications; they will assess your eligibility.",
        "There are several types, including whole blood, plasma, platelet, and double red cell donations.",
        "Donors should weigh at least 50 kg (110 pounds) to be eligible to donate blood.",
        "The process usually takes around 8-10 minutes, with preparation taking up to an hour.",
        "Recent travel to certain areas may affect eligibility due to disease risks.",
        "Generally, a minimum of 8 weeks is recommended between whole blood donations.",
        "Plasma donation is the process of donating the liquid portion of the blood.",
        "Blood donations are tested for infectious diseases as a safety measure.",
        "You may donate if you’ve had a tattoo in a licensed facility; wait 3-6 months if unlicensed.",
        "Common side effects are minor, like slight dizziness or bruising.",
        "The minimum age is 18 in most places, with a maximum age limit depending on health.",
        "Pregnant women are typically advised against donating blood.",
        "People with well-controlled diabetes may be eligible to donate blood.",
        "Avoid alcohol, fatty foods, and intense exercise before donating blood.",
        "The pain is minimal, similar to a quick needle prick.",
        "You may need to wait several weeks after surgery, depending on recovery.",
        "Blood is separated into components, stored, and used for various treatments.",
        "Vaccinations may require a temporary wait before you can donate blood.",
        "Being mentally prepared and relaxed can help reduce any anxiety about donating.",
        "Yes, friends or family can often accompany you to the donation center.",
        "Dizziness is rare and usually temporary; staying hydrated helps reduce this risk.",
        "Type O-negative blood is most needed as it can be given to patients of any blood type.",
        "Blood is used for surgeries, emergencies, cancer treatments, and chronic illnesses.",
        "High blood pressure must be controlled and within certain limits to donate blood.",
        "Foods rich in iron and fluids help recovery; consider orange juice and leafy greens.",
        "Fainting is uncommon but can happen; staff are trained to respond to such cases.",
        "People with anemia are generally advised not to donate until they recover.",
        "People with heart conditions may need medical clearance to donate blood.",
        "Double red cell donation allows donors to give twice the amount of red cells.",
        "No, knowing your blood type is not required; it will be determined at the center.",
        "Breastfeeding women should consult with a healthcare provider before donating.",
        "Report any prolonged discomfort to the blood donation center or a healthcare provider.",
        "Yes, blood is sometimes tested for COVID-19 antibodies to help research efforts.",
        "Regular blood donation may provide cardiovascular benefits for some individuals.",
        "People with mild allergies who are otherwise healthy can usually donate.",
        "Bring a photo ID, list of medications, and eat a nutritious meal before donating.",
        "Visit the blood bank website or call to find your nearest blood donation center.",
        "If nervous, practice relaxation techniques or bring a friend for support.",
        "You may want to rest and avoid intense exercise for a day after donating.",
        "Most people can return to normal activities shortly after resting post-donation.",
        "While age limits vary, seniors in good health can often continue donating.",
        "Yes, vegetarians can donate blood without issue as long as they meet health criteria."
    ]

# Function to find the most similar context using Sentence-BERT
def get_best_context(question):
    question_embedding = sentence_model.encode([question])
    context_embeddings = sentence_model.encode(contexts)
    
    # Compute cosine similarity between the question and all contexts
    similarities = cosine_similarity(question_embedding, context_embeddings)
    
    # Get the most similar context (highest cosine similarity)
    best_context_idx = np.argmax(similarities)
    return contexts[best_context_idx]

def evaluate_model(question):
    # Try to get a relevant context using semantic similarity
    context = get_best_context(question)
    
    # Use the BERT QA model to answer the question with the selected context
    inputs = tokenizer(question, context, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer

# Example usage
question = sys.argv[1]
answer = evaluate_model(question)
print(f"Answer: {answer}")

