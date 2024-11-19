
# dataprepare.py
from transformers import BertTokenizerFast
from datasets import Dataset

# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Function to preprocess data
def preprocess_data(examples):
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    # Mapping start and end tokens
    offset_mapping = tokenized_examples.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        sequence_ids = tokenized_examples.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        if offsets[context_start][0] <= start_char and offsets[context_end][1] >= end_char:
            start_positions.append(context_start)
            end_positions.append(context_end)
        else:
            start_positions.append(0)
            end_positions.append(0)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    return tokenized_examples

# Function to load and tokenize data
def load_data():
    # Define example dataset
    data = {
    "question": [
        "What is blood donation?",
        "Who can donate blood?",
        "How often can I donate blood?",
        "Is blood donation safe?",
        "What are the benefits of donating blood?",
        "Can I donate blood if I have a cold?",
        "What should I do before donating blood?",
        "What should I eat before donating blood?",
        "Will donating blood make me weak?",
        "Can I donate blood if I'm on medication?",
        "What types of blood donation are there?",
        "Is there a minimum weight requirement for donating blood?",
        "How long does a blood donation take?",
        "Can I donate blood if I recently traveled?",
        "How long should I wait between blood donations?",
        "What is plasma donation?",
        "Will I be tested for diseases when I donate blood?",
        "Can I donate blood if I have a tattoo?",
        "Are there any side effects of donating blood?",
        "How old do I need to be to donate blood?",
        "Can pregnant women donate blood?",
        "Can I donate blood if I have diabetes?",
        "What should I avoid before donating blood?",
        "Is blood donation painful?",
        "Can I donate blood if I had surgery recently?",
        "What happens to my blood after donation?",
        "Can I donate if I've had a recent vaccination?",
        "How can I prepare myself mentally for blood donation?",
        "Can I bring someone with me for support when donating?",
        "Will I feel dizzy after donating blood?",
        "Is there a blood type that is most needed?",
        "How is blood used in hospitals?",
        "Can I donate blood if I have high blood pressure?",
        "Are there foods that help me recover after donating?",
        "What if I faint while donating blood?",
        "Can I donate blood if I am anemic?",
        "Can I donate if I have a heart condition?",
        "What is double red cell donation?",
        "Do I need to know my blood type to donate?",
        "Can I donate if I’m breastfeeding?",
        "What if I feel unwell after donating?",
        "Is my blood checked for COVID-19 antibodies?",
        "Can blood donation improve my health?",
        "Can people with allergies donate blood?",
        "What should I bring with me to the donation site?",
        "How do I find a blood donation center near me?",
        "What do I do if I am nervous about needles?",
        "Will donating blood affect my exercise routine?",
        "How soon can I go back to work after donating?",
        "Are there any age limits to donating blood?",
        "Can I donate blood if I am a vegetarian?"
    ],
    "context": [
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
    ],
    "answers": [
        {"text": ["Blood donation is a process where a donor voluntarily has blood drawn for medical needs."], "answer_start": [0]},
        {"text": ["Anyone in good health, usually aged 18-65, who meets health requirements can donate blood."], "answer_start": [0]},
        {"text": ["Healthy individuals can donate whole blood every 56 days, or up to 6 times a year."], "answer_start": [0]},
        {"text": ["Blood donation is safe, and all equipment is sterilized and used only once."], "answer_start": [0]},
        {"text": ["Donating blood helps save lives and can provide a sense of fulfillment."], "answer_start": [0]},
        {"text": ["It's recommended not to donate if you have a cold, as it may affect donation quality."], "answer_start": [0]},
        {"text": ["Stay hydrated, eat iron-rich foods, and get adequate sleep before donating."], "answer_start": [0]},
        {"text": ["A balanced meal with iron-rich foods, like spinach and beans, is beneficial before donating."], "answer_start": [0]},
        {"text": ["Mild weakness may occur but typically fades quickly after donating."], "answer_start": [0]},
        {"text": ["Inform the blood center about any medications; they will assess your eligibility."], "answer_start": [0]},
                {"text": ["There are several types, including whole blood, plasma, platelet, and double red cell donations."], "answer_start": [0]},
        {"text": ["Donors should weigh at least 50 kg (110 pounds) to be eligible to donate blood."], "answer_start": [0]},
        {"text": ["The process usually takes around 8-10 minutes, with preparation taking up to an hour."], "answer_start": [0]},
        {"text": ["Recent travel to certain areas may affect eligibility due to disease risks."], "answer_start": [0]},
        {"text": ["Generally, a minimum of 8 weeks is recommended between whole blood donations."], "answer_start": [0]},
        {"text": ["Plasma donation is the process of donating the liquid portion of the blood."], "answer_start": [0]},
        {"text": ["Blood donations are tested for infectious diseases as a safety measure."], "answer_start": [0]},
        {"text": ["You may donate if you’ve had a tattoo in a licensed facility; wait 3-6 months if unlicensed."], "answer_start": [0]},
        {"text": ["Common side effects are minor, like slight dizziness or bruising."], "answer_start": [0]},
        {"text": ["The minimum age is 18 in most places, with a maximum age limit depending on health."], "answer_start": [0]},
        {"text": ["Pregnant women are typically advised against donating blood."], "answer_start": [0]},
        {"text": ["People with well-controlled diabetes may be eligible to donate blood."], "answer_start": [0]},
        {"text": ["Avoid alcohol, fatty foods, and intense exercise before donating blood."], "answer_start": [0]},
        {"text": ["The pain is minimal, similar to a quick needle prick."], "answer_start": [0]},
        {"text": ["You may need to wait several weeks after surgery, depending on recovery."], "answer_start": [0]},
        {"text": ["Blood is separated into components, stored, and used for various treatments."], "answer_start": [0]},
        {"text": ["Vaccinations may require a temporary wait before you can donate blood."], "answer_start": [0]},
        {"text": ["Being mentally prepared and relaxed can help reduce any anxiety about donating."], "answer_start": [0]},
        {"text": ["Yes, friends or family can often accompany you to the donation center."], "answer_start": [0]},
        {"text": ["Dizziness is rare and usually temporary; staying hydrated helps reduce this risk."], "answer_start": [0]},
        {"text": ["Type O-negative blood is most needed as it can be given to patients of any blood type."], "answer_start": [0]},
        {"text": ["Blood is used for surgeries, emergencies, cancer treatments, and chronic illnesses."], "answer_start": [0]},
        {"text": ["High blood pressure must be controlled and within certain limits to donate blood."], "answer_start": [0]},
        {"text": ["Foods rich in iron and fluids help recovery; consider orange juice and leafy greens."], "answer_start": [0]},
        {"text": ["Fainting is uncommon but can happen; staff are trained to respond to such cases."], "answer_start": [0]},
        {"text": ["People with anemia are generally advised not to donate until they recover."], "answer_start": [0]},
        {"text": ["People with heart conditions may need medical clearance to donate blood."], "answer_start": [0]},
        {"text": ["Double red cell donation allows donors to give twice the amount of red cells."], "answer_start": [0]},
        {"text": ["No, knowing your blood type is not required; it will be determined at the center."], "answer_start": [0]},
        {"text": ["Breastfeeding women should consult with a healthcare provider before donating."], "answer_start": [0]},
        {"text": ["Report any prolonged discomfort to the blood donation center or a healthcare provider."], "answer_start": [0]},
        {"text": ["Yes, blood is sometimes tested for COVID-19 antibodies to help research efforts."], "answer_start": [0]},
        {"text": ["Regular blood donation may provide cardiovascular benefits for some individuals."], "answer_start": [0]},
        {"text": ["People with mild allergies who are otherwise healthy can usually donate."], "answer_start": [0]},
        {"text": ["Bring a photo ID, list of medications, and eat a nutritious meal before donating."], "answer_start": [0]},
        {"text": ["Visit the blood bank website or call to find your nearest blood donation center."], "answer_start": [0]},
        {"text": ["If nervous, practice relaxation techniques or bring a friend for support."], "answer_start": [0]},
        {"text": ["You may want to rest and avoid intense exercise for a day after donating."], "answer_start": [0]},
        {"text": ["Most people can return to normal activities shortly after resting post-donation."], "answer_start": [0]},
        {"text": ["While age limits vary, seniors in good health can often continue donating."], "answer_start": [0]},
        {"text": ["Yes, vegetarians can donate blood without issue as long as they meet health criteria."], "answer_start": [0]}
    ]
}

    dataset = Dataset.from_dict(data)
    tokenized_dataset = dataset.map(preprocess_data, batched=True)
    return tokenized_dataset