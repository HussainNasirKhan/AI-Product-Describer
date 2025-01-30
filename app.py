from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import uuid
import logging
import base64
from groq import Groq
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# API Key
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=groq_api_key)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_unique_filename(original_filename):
    """Generate a unique filename with timestamp and UUID"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = str(uuid.uuid4())[:8]
    ext = original_filename.rsplit('.', 1)[1].lower()
    return f"{timestamp}_{unique_id}.{ext}"

def encode_image_to_base64(image_path):
    """Convert image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_description(image_path):
    """
    Generate product description using Groq's Llama Vision API
    """
    start_time = time.time()
    
    try:
        # Convert image to base64
        base64_image = encode_image_to_base64(image_path)
        
        # Prepare the message for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Write a detailed e-commerce product description for this item. Include key features, materials, and potential uses. Make it appealing for online shoppers."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]

        # Call the Groq API
        chat_completion = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=messages,
            temperature=0.7,
            max_tokens=500,
            top_p=0.9
        )

        # Get the response
        description = chat_completion.choices[0].message.content
        processing_time = f"{time.time() - start_time:.2f}s"
        
        return {
            "description": description,
            "metadata": {
                "processing_time": processing_time,
                "model": "llama-3.2-90b-vision-preview",
                "tokens_used": chat_completion.usage.total_tokens,
                "confidence_score": 0.95  # Placeholder as the model doesn't provide confidence scores
            }
        }
        
    except Exception as e:
        logger.error(f"Error in generate_description: {str(e)}")
        raise

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and generate description"""
    try:
        # Check if image file is present in request
        if 'image' not in request.files:
            return jsonify({
                "error": "No image file provided"
            }), 400
        
        file = request.files['image']
        
        # Check if a file was actually selected
        if file.filename == '':
            return jsonify({
                "error": "No selected file"
            }), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({
                "error": f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400
        
        # Generate secure filename and save file
        filename = secure_filename(generate_unique_filename(file.filename))
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"Successfully saved image: {filename}")
        
        # Check for API key
        if not os.environ.get("GROQ_API_KEY"):
            return jsonify({
                "error": "GROQ_API_KEY not set in environment variables"
            }), 500
        
        # Generate description using the vision model
        try:
            result = generate_description(filepath)
            
            return jsonify({
                "success": True,
                "filename": filename,
                "description": result["description"],
                "metadata": result["metadata"]
            })
            
        except Exception as e:
            logger.error(f"Error generating description: {str(e)}")
            return jsonify({
                "error": "Failed to generate description",
                "details": str(e)
            }), 500
            
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        return jsonify({
            "error": "Failed to process upload",
            "details": str(e)
        }), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file size exceeded error"""
    return jsonify({
        "error": "File size exceeded",
        "max_size": "16MB"
    }), 413

if __name__ == '__main__':
    app.run(debug=True, port=5000)