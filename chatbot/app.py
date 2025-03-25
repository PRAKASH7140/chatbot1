import os
from flask import Flask, request, jsonify, render_template, send_from_directory, session
from werkzeug.utils import secure_filename
from main import recognize_image
from nlp import generate_response
from ocr import extract_text
from face_detection import detect_faces

app = Flask(__name__)
app.secret_key = "supersecretkey"  # For maintaining session data

UPLOAD_FOLDER = 'media'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/media/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def home():
    session.clear()  # Clear chat history on new visit
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(image_path)

    user_query = request.form.get("user_query", "What do you see in this image?")

    try:
        # Run image recognition
        recognized_objects = recognize_image(image_path)
        detected_faces = detect_faces(image_path) or 0  # Ensure None is replaced with 0
        extracted_text = extract_text(image_path) or "No text detected."

        print(f"DEBUG: Recognized Objects: {recognized_objects}")
        print(f"DEBUG: Detected Faces: {detected_faces}")
        print(f"DEBUG: Extracted Text: {extracted_text}")

        # Maintain chat history in session
        if "chat_history" not in session:
            session["chat_history"] = []

        response = generate_response(recognized_objects, detected_faces, extracted_text, user_query)

        session["chat_history"].append({"user": user_query, "bot": response})

    except Exception as e:
        print(f"ERROR: {str(e)}")
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

    return jsonify({
        "response": response,
        "image_path": f"/media/{filename}",
        "chat_history": session["chat_history"]
    })

if __name__ == '__main__':
    app.run(debug=True)
