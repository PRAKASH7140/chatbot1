import spacy
from transformers import pipeline
import torch

nlp = spacy.load("en_core_web_sm")

device = 0 if torch.cuda.is_available() else -1
gpt_pipeline = pipeline("text-generation", model="EleutherAI/gpt-neo-125M", device=device)

def generate_response(recognized_objects, detected_faces, extracted_text, user_query):
    if not recognized_objects:
        recognized_objects = [{"label": "unknown", "confidence": 0}]
    
    objects = ", ".join([f"{obj['label']} ({obj['confidence']}%)" for obj in recognized_objects])
    faces = f"{detected_faces} face(s) detected." if detected_faces > 0 else "No faces detected."
    text = f"Extracted text: {extracted_text}"

    responses = {
        "what objects are in the image": f"I detected {objects}.",
        "is there a person in the image": "Yes, I see a person." if any(obj['label'].lower() == "person" for obj in recognized_objects) else "No person detected.",
        "what is the main subject of the image": f"The main subject appears to be {recognized_objects[0]['label']} with {recognized_objects[0]['confidence']}% confidence.",
        "does the image contain text": text,
        "does the image contain any vehicles": f"Yes, I see a vehicle: {objects}" if any(
            obj['label'] in ["car", "bus", "truck", "motorcycle", "bicycle"] for obj in recognized_objects
        ) else "No vehicles detected.",
        "is there a landmark in the image": "I might be able to identify famous objects, but I don't explicitly detect landmarks.",
    }

    doc = nlp(user_query.lower())

    # Find the best-matching question using similarity
    best_match = max(responses.keys(), key=lambda q: nlp(q).similarity(doc))
    print(f"DEBUG: Best Matched Question: {best_match}")

    # Return the best-matching response
    if best_match:
        return responses[best_match]

    # Fallback: Use GPT if no good match
    gpt_response = gpt_pipeline(
        f"Analyze this: {objects}, {faces}, {text}. User's question: {user_query}",
        max_new_tokens=50,
        num_return_sequences=1,
        temperature=0.8,
        top_p=0.9
    )

    return gpt_response[0]['generated_text']
