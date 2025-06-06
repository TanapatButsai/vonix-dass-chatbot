import difflib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

# โหลด stopwords ภาษาอังกฤษ
nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# คำถาม-คำตอบจาก MedQuAD แบบง่าย
qa_pairs = {
    "what is glaucoma": "Glaucoma is a group of diseases that can damage the eye's optic nerve and result in vision loss.",
    "what causes glaucoma": "Nearly 2.7 million people have glaucoma. Although exact causes are unknown, pressure buildup in the eye is a common factor.",
    "what are the symptoms of glaucoma": "Glaucoma may have no symptoms at first. Vision loss can occur gradually.",
    "how to prevent glaucoma": "Currently, there's no known prevention, but early detection via eye exams can help manage it.",
    "who is at risk for glaucoma": "Higher risk groups include African-Americans, older adults, and those with a family history.",
    "what are the treatments for glaucoma": "Open-angle glaucoma cannot be cured but can usually be controlled with medications or surgery.",

    "what is high blood pressure": "High blood pressure is a condition where blood flows through arteries at higher than normal pressure.",
    "what causes high blood pressure": "Causes include genetics, poor diet, lack of exercise, and certain medical conditions.",
    "what are the symptoms of high blood pressure": "Often called the 'silent killer' because it may have no symptoms.",
    "how to prevent high blood pressure": "Adopt healthy habits: reduce salt, exercise, avoid alcohol and smoking.",
    "what are the treatments for high blood pressure": "Treatment involves lifestyle changes and medications such as beta-blockers or ACE inhibitors.",

    "what is uti": "Urinary tract infections (UTIs) are infections in the urinary system, common in older adults.",
    "what causes uti": "UTIs are caused by bacteria entering the urinary tract, usually through the urethra.",
    "what are the symptoms of uti": "Symptoms include pain during urination, cloudy urine, and frequent urge to urinate.",
    "how to prevent uti": "Good hygiene, staying hydrated, and urinating after sex can help prevent UTIs."
}

def clean_input(text):
    tokens = tokenizer.tokenize(text.lower())
    filtered = [w for w in tokens if w not in stop_words]
    return " ".join(filtered)

def get_best_match(user_input, qa_keys):
    cleaned_input = clean_input(user_input)
    matches = difflib.get_close_matches(cleaned_input, qa_keys, n=1, cutoff=0.5)
    return matches[0] if matches else None

def medical_bot():
    print("=== VONIX Smart Medical QA Bot (Draft)  ===")
    print("Type your question (or type 'exit' to quit)\n")

    qa_keys = list(qa_pairs.keys())

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Bot: Take care! 👋")
            break

        best_match = get_best_match(user_input, qa_keys)
        if best_match:
            print(f"Bot: {qa_pairs[best_match]}\n")
        else:
            print("Bot: Sorry, I couldn't understand your question. Try rephrasing it.\n")

# Run the bot
if __name__ == "__main__":
    medical_bot()
