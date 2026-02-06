from flask import Flask, render_template, request, jsonify
import os
import secrets
from inference import load_model, predict_image
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create uploads folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model once at startup
print("Initializing model...")
model = load_model("best_model.pth")
print("Model initialized.")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if file and allowed_file(file.filename):
            # Save securely
            filename = secure_filename(file.filename)
            # Add random hex to avoid collisions
            unique_name = f"{secrets.token_hex(8)}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
            file.save(filepath)
            
            # Predict
            label, score = predict_image(model, filepath)
            
            # Clean up - optionally keep them if you want logs
            # os.remove(filepath) 
            
            return jsonify({
                'label': label,
                'score': score,
                'filename': filename
            })
                
        return jsonify({'error': 'File type not allowed'}), 400

    except Exception as e:
        print(f"Error in predict route: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
