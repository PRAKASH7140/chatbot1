import spacy
import torch
import faiss
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Load NLP Model for Named Entity Recognition
nlp = spacy.load("en_core_web_sm")

# Load a lightweight sentence transformer for intent matching
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Falcon-7B model for better text generation (alternative to GPT-Neo)
model_name = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Sample predefined responses with embeddings
intent_responses = {
    "what objects are in the image": f"I detected {objects}.",
    "is there a person in the image": "Yes, I see a person." if any(obj['label'] in ["person", "man", "woman", "boy", "girl", "runner", "dancer"] for obj in recognized_objects) else "No person detected.",
    "what is the main subject of the image": "The main subject appears to be {main_object} with {confidence}% confidence.",
    "does the image contain text": "{text}",
    "does the image contain any vehicles": "Yes, I see a vehicle." if "{vehicle}" in "{objects}" else "No vehicles detected.",
    "is there a landmark in the image": "I might be able to identify famous objects, but I don't explicitly detect landmarks.",
}

# Convert predefined responses into embeddings
intent_embeddings = torch.stack([embedding_model.encode(key, convert_to_tensor=True) for key in intent_responses.keys()])

# Initialize FAISS Index for fast semantic search
index = faiss.IndexFlatL2(intent_embeddings.shape[1])
faiss.normalize_L2(intent_embeddings.numpy())  # Normalize embeddings
index.add(intent_embeddings.numpy())  # Add to FAISS

def generate_llm_response(prompt):
    """
    Generate a response using Falcon-7B (or your preferred model).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = llm.generate(**inputs, max_length=100, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_response(recognized_objects, detected_faces, extracted_text, user_query):
    """
    Generate a chatbot response based on the image analysis and user query.
    """
    objects = ", ".join([f"{obj['label']} ({obj['confidence']}%)" for obj in recognized_objects]) if recognized_objects else "nothing specific"
    main_object = recognized_objects[0]['label'] if recognized_objects else "unknown"
    confidence = recognized_objects[0]['confidence'] if recognized_objects else "unknown"
    faces = f"{detected_faces} face(s) detected." if isinstance(detected_faces, int) and detected_faces > 0 else "No faces detected."
    text = f"Extracted text: {extracted_text}" if extracted_text else "No text detected."

    # Convert user query to embedding
    query_embedding = embedding_model.encode(user_query, convert_to_tensor=True).unsqueeze(0)

    # Search FAISS for closest intent match
    D, I = index.search(query_embedding.numpy(), 1)  # Find closest match
    similarity_score = 1 - D[0][0]  # Convert FAISS distance to similarity
    best_match = list(intent_responses.keys())[I[0][0]]

    # If similarity is high, return predefined response
    if similarity_score > 0.7:
        return intent_responses[best_match].format(
            objects=objects, person="person" in objects, 
            main_object=main_object, confidence=confidence, 
            text=text, vehicle="car" in objects or "bus" in objects or "truck" in objects
        )

    # Otherwise, generate a response dynamically using LLM
    prompt = f"Analyze this: {objects}, {faces}, {text}. Based on the user's question: {user_query}"
    return generate_llm_response(prompt)
