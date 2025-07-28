"""
RSS Publishing AI - Flask application for generating AI-powered blog posts from RSS feeds
"""
import os
import re
import json
import logging
from flask import Flask, render_template, request, jsonify, send_file
import google.genai
from google.genai.types import HarmCategory, HarmBlockThreshold, GenerateImagesConfig, GenerateContentConfig
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import feedparser
import uuid  # For generating unique filenames for images

# Import our utility modules
from utils.text_utils import generate_post_content, parse_rss_feed
from utils.image_utils import (
    generate_image, cleanup_temp_images, validate_and_fix_image, get_temp_image_dir
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Get the directory for temporary images
TEMP_IMG_DIR = get_temp_image_dir()

# Clean up old images at startup
cleanup_temp_images()

# Configure Gemini AI
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    logger.error("GEMINI_API_KEY not found in environment variables")
    raise ValueError("GEMINI_API_KEY is required")

client = google.genai.Client(api_key=api_key)

# Available Gemini models
IMAGE_GENERATION_MODEL = 'gemini-2.0-flash-preview-image-generation'  # For image generation
TEXT_MODEL = 'gemini-2.5-flash'  # For text generation

class RSSPostGenerator:
    def __init__(self):
        # Safety settings to allow creative content while maintaining safety
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        logger.info("Gemini client initialized for text and image generation")
    
    def parse_rss_feed(self, rss_url):
        """Parse RSS feed and return list of entries"""
        return parse_rss_feed(rss_url)
    
    def generate_post_content(self, title, description):
        """Generate blog post content using Gemini AI"""
        return generate_post_content(client, title, description)
    
    def generate_image(self, image_description):
        """Generate an image using Gemini image generation model based on the description."""
        try:
            logger.info(f"Using Gemini model '{IMAGE_GENERATION_MODEL}' for image generation")
            logger.info(f"Image description: {image_description[:100]}...")

            # Simplify the image description for better generation
            simplified_description = self._simplify_image_description(image_description)
            logger.info(f"Simplified image description: {simplified_description}")

            # Attempt text-to-image generation first
            try:
                logger.info(f"🔄 Attempting text-to-image generation with {IMAGE_GENERATION_MODEL}...")
                response = client.models.generate_content(
                    model=IMAGE_GENERATION_MODEL,
                    contents=[simplified_description],
                    config=GenerateContentConfig(
                        response_modalities=["IMAGE", "TEXT"],
                        temperature=0.3,  # Slightly higher for creative text placement
                        max_output_tokens=2048,
                        candidate_count=1
                    )
                )
                logger.info("✅ Text-to-image generation request sent successfully")

                # Extract image data from the response
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, 'inline_data') and part.inline_data:
                                image_data = part.inline_data.data
                                mime_type = part.inline_data.mime_type
                                extension = mime_type.split('/')[-1]  # Extract file extension from MIME type
                                filename = f"generated_{uuid.uuid4().hex}.{extension}"
                                file_path = os.path.join(TEMP_IMG_DIR, filename)
                                with open(file_path, "wb") as f:
                                    f.write(image_data)

                                logger.info(f"Successfully generated image using Gemini and saved to {file_path}")

                                return {
                                    "image_url": f"/static/temp/{filename}",
                                    "status": "success",
                                    "source": "gemini",
                                    "local_path": file_path,
                                    "message": "Generated image using Gemini AI"
                                }
                        logger.error(f"No inline_data found in the candidate parts: {candidate.content.parts}")
                        raise ValueError("No image data found in the candidate response")
                    else:
                        logger.error(f"Candidate does not contain valid content parts: {candidate}")
                        raise ValueError("No image data found in the candidate response")
                else:
                    logger.error(f"Response does not contain candidates or is malformed: {response}")
                    raise ValueError("No candidates found in the response")

            except Exception as text_to_image_error:
                logger.warning(f"⚠️  Text-to-image model failed: {text_to_image_error}")
                logger.info("🔄 Falling back to image generation model with modified prompt...")

                # Modify the prompt for fallback
                fallback_prompt = f"Professional studio photograph of {simplified_description}.\n\nRequirements:\n- High quality product photography\n- Professional studio lighting with soft shadows\n- Clean, professional look suitable for advertising\n- No text or watermarks on the image"

                # Dynamically select a valid model for image generation
                available_models = client.models.list()
                logger.info("Available models for fallback:")
                for model in available_models:
                    logger.info(f"Model: {model.name}, Supported Actions: {model.supported_actions}")

                # Adjust fallback logic to use models supporting 'generateContent'
                valid_model = next((model.name for model in available_models if "generateContent" in model.supported_actions), IMAGE_GENERATION_MODEL)

                if not valid_model:
                    raise ValueError("No valid image generation model found. Ensure the model supports 'generateContent'.")

                logger.info(f"Using fallback model: {valid_model}")

                # Generate the image using fallback approach
                response = client.models.generate_content(
                    model=valid_model,
                    contents=[fallback_prompt],
                    config=GenerateContentConfig(
                        response_modalities=["IMAGE", "TEXT"],
                        temperature=0.1,  # Low temperature for consistent results
                        max_output_tokens=2048,
                        candidate_count=1
                    )
                )

                # Extract image data from the response
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and candidate.content.parts:
                        for part in candidate.content.parts:
                            logger.debug(f"Inspecting part: {part}")
                            if hasattr(part, 'inline_data') and part.inline_data:
                                logger.info(f"Found inline_data with MIME type: {part.inline_data.mime_type}")
                                image_data = part.inline_data.data
                                mime_type = part.inline_data.mime_type
                                extension = mime_type.split('/')[-1]  # Extract file extension from MIME type
                                filename = f"generated_{uuid.uuid4().hex}.{extension}"
                                file_path = os.path.join(TEMP_IMG_DIR, filename)
                                with open(file_path, "wb") as f:
                                    f.write(image_data)

                                logger.info(f"Successfully generated image using fallback model and saved to {file_path}")

                                return {
                                    "image_url": f"/static/temp/{filename}",
                                    "status": "success",
                                    "source": "gemini",
                                    "local_path": file_path,
                                    "message": "Generated image using fallback model"
                                }
                        logger.error(f"No valid inline_data found in candidate parts: {candidate.content.parts}")
                        raise ValueError("No image data found in the candidate response")
                    else:
                        logger.error(f"Candidate does not contain valid content parts: {candidate}")
                        raise ValueError("No image data found in the candidate response")
                else:
                    logger.error(f"Response does not contain candidates or is malformed: {response}")
                    raise ValueError("No candidates found in the response")

        except google.genai.errors.ClientError as e:
            if "NOT_FOUND" in str(e):
                logger.error("Model not found. Listing available models for debugging.")
                self.list_available_models()
            logger.error(f"Error in image generation: {e}", exc_info=True)
            return {
                "image_url": "https://via.placeholder.com/800x450/ff0000/ffffff?text=Error+Occurred",
                "status": "error",
                "source": "gemini",
                "message": f"Error generating image: {str(e)}"
            }
    
    def _simplify_image_description(self, description):
        """Simplify the image description for better image generation."""
        # Basic simplification: remove extra spaces and limit to 200 characters
        simplified = re.sub(r'\s+', ' ', description).strip()
        return simplified[:200]  # Limit to 200 characters
    
    def list_available_models(self):
        """List available models and log them for debugging purposes."""
        try:
            logger.info("Fetching available models...")
            models = client.models.list()
            for model in models:
                logger.info(f"Model object: {model}")  # Log the entire model object for debugging
        except Exception as e:
            logger.error(f"Error fetching available models: {e}", exc_info=True)
            

# Initialize the generator
generator = RSSPostGenerator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/parse-rss', methods=['POST'])
def parse_rss():
    """Parse RSS feed and return entries"""
    data = request.get_json()
    rss_url = data.get('rss_url')
    
    if not rss_url:
        return jsonify({'error': 'RSS URL is required'}), 400
    
    entries = generator.parse_rss_feed(rss_url)
    return jsonify({'entries': entries})

@app.route('/api/generate-post', methods=['POST'])
def generate_post():
    """Generate blog post and image from title and description"""
    try:
        logger.info("API Request received: /api/generate-post")
        data = request.get_json()
        logger.info(f"Request data: {data}")
        
        title = data.get('title')
        description = data.get('description')
        
        if not title or not description:
            logger.warning("Missing title or description")
            return jsonify({'error': 'Title and description are required'}), 400
        
        logger.info(f"Generating post content for title: {title}")
        # Generate post content and image description
        content_result = generator.generate_post_content(title, description)
        logger.info(f"Content result keys: {list(content_result.keys()) if content_result else 'None'}")
        
        # Check if we're in processing mode (background request still running)
        processing = content_result.get('processing', False)
        
        image_description = content_result.get('image_description', f"An illustration representing: {title}")
        logger.info(f"Image description: {image_description[:50]}...")
        
        # Generate image using available image services
        logger.info("Generating image...")
        try:
            image_result = generator.generate_image(image_description)
            logger.info(f"Image result status: {image_result.get('status')}")
        except Exception as img_error:
            logger.error(f"Error during image generation: {img_error}", exc_info=True)
            # Fall back to placeholder image on error
            image_result = {
                'image_url': f"https://via.placeholder.com/800x450/ff0000/ffffff?text={image_description.replace(' ', '+')}",
                'status': 'error',
                'message': f"Error generating image: {str(img_error)}"
            }
        
        # Prepare comprehensive response with all image metadata
        response_data = {
            'title': title,
            'post_content': content_result.get('post_content', f"Blog post about: {title}\n\n{description}"),
            'image_description': image_description,
            'image_url': image_result.get('image_url', ''),
            'status': image_result.get('status', 'error'),
            'source': image_result.get('source', 'unknown'),
            'processing': processing,  # Include processing flag in response
            'message': image_result.get('message', ''),
            'local_path': image_result.get('local_path', '')
        }
        
        # If still processing, add a special status message
        if processing:
            response_data['status_message'] = "Content generation is still in progress. Partial results shown."
        
        logger.info(f"API Response prepared: {len(response_data['post_content'])} chars of content, image status: {response_data['status']}, processing: {processing}")
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error in generate_post endpoint: {e}", exc_info=True)
        error_html = f"""<div class="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 my-4">
            <p class="font-bold">Error</p>
            <p>{str(e)}</p>
            <p>Please try again in a few moments.</p>
        </div>"""
        
        return jsonify({
            'error': f"An error occurred: {str(e)}",
            'title': title if 'title' in locals() else "Error",
            'post_content': error_html,
            'image_description': "Error occurred",
            'image_url': "https://via.placeholder.com/800x450/ff0000/ffffff?text=Error+Occurred",
            'status': 'error'
        }), 500

@app.route('/api/test', methods=['GET'])
def test_api():
    """Test endpoint to check if API is working"""
    return jsonify({
        'status': 'success',
        'message': 'API is working correctly'
    })

@app.route('/api/check-image-model', methods=['GET'])
def check_image_model():
    """Check if the image generation model is available"""
    try:
        logger.info("Checking image generation model availability...")
        
        # Try a simple request to check if the model is available
        test_prompt = "Test prompt to check model availability"
        test_response = None
        
        try:
            # Try generating content with the image model with a timeout
            import threading
            import time
            
            response_data = [None]
            response_error = [None]
            request_completed = [False]
            
            def make_test_request():
                try:
                    response_data[0] = client.models.generate_content(
                        model=IMAGE_GENERATION_MODEL,
                        contents=[test_prompt],
                        config=GenerateContentConfig(
                            response_modalities=["IMAGE", "TEXT"],
                            temperature=0.1,
                            max_output_tokens=10
                        )
                    )
                    request_completed[0] = True
                except Exception as e:
                    response_error[0] = e
                    request_completed[0] = True
            
            # Start test request in a thread
            test_thread = threading.Thread(target=make_test_request)
            test_thread.start()
            
            # Wait for 5 seconds max
            start_time = time.time()
            while not request_completed[0] and time.time() - start_time < 5:
                time.sleep(0.1)
                
            model_available = request_completed[0] and response_data[0] is not None and not response_error[0]
            error_message = str(response_error[0]) if response_error[0] else None
            
        except Exception as e:
            model_available = False
            error_message = str(e)
        
        # List available models from environment
        available_models = [
            TEXT_MODEL,
            IMAGE_GENERATION_MODEL
        ]
        
        return jsonify({
            'status': 'success',
            'image_model_available': model_available,
            'image_model_name': IMAGE_GENERATION_MODEL,
            'text_model_name': TEXT_MODEL,
            'available_models': available_models,
            'error_message': error_message
        })
    except Exception as e:
        logger.error(f"Error checking image model: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/generated-image/<filename>', methods=['GET'])
def generated_image(filename):
    """Serve a generated image directly with on-the-fly validation and fixing"""
    try:
        # Security check to prevent directory traversal attacks
        if '..' in filename or filename.startswith('/'):
            return "Invalid filename", 400
            
        # Only allow jpg/jpeg/png files
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            return "Invalid file type", 400
            
        file_path = os.path.join(TEMP_IMG_DIR, filename)
        
        # Check if file exists
        if not os.path.isfile(file_path):
            # Create a special "not found" image
            try:
                from PIL import Image, ImageDraw, ImageFont
                
                # Create a simple "not found" image
                img = Image.new('RGB', (800, 450), color=(238, 238, 238))
                draw = ImageDraw.Draw(img)
                
                # Try to load a font
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
                    small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
                except Exception:
                    font = ImageFont.load_default()
                    small_font = ImageFont.load_default()
                
                # Add text
                draw.text((400, 200), "Image Not Found", font=font, fill=(33, 33, 33), anchor="mm")
                draw.text((400, 250), filename, font=small_font, fill=(100, 100, 100), anchor="mm")
                
                # Add border
                draw.rectangle([(0, 0), (799, 449)], outline=(200, 200, 200), width=5)
                
                from io import BytesIO
                img_io = BytesIO()
                img.save(img_io, 'JPEG', quality=95)
                img_io.seek(0)
                
                return send_file(img_io, mimetype='image/jpeg')
            except Exception as nf_err:
                logger.error(f"Error creating not found image: {nf_err}")
                return "File not found", 404
            
        # Attempt to validate and fix the image
        if not validate_and_fix_image(file_path):
            logger.warning(f"Image validation failed for {filename}, attempting to serve anyway")
            
        # Determine content type
        content_type = 'image/jpeg'
        if filename.lower().endswith('.png'):
            content_type = 'image/png'
            
        # Add cache control headers to prevent browser caching
        response = send_file(file_path, mimetype=content_type)
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        
        return response
    except Exception as e:
        logger.error(f"Error serving image file: {e}")
        
        # Try to create a simple error image
        try:
            from PIL import Image, ImageDraw, ImageFont
            from io import BytesIO
            
            # Create a simple error image
            img = Image.new('RGB', (800, 450), color=(255, 240, 240))
            draw = ImageDraw.Draw(img)
            
            # Try to load a font
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
                small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            except Exception:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # Add text
            draw.text((400, 180), "Error Loading Image", font=font, fill=(200, 0, 0), anchor="mm")
            
            error_text = str(e)
            if len(error_text) > 60:
                error_text = error_text[:57] + "..."
                
            draw.text((400, 240), error_text, font=small_font, fill=(100, 0, 0), anchor="mm")
            draw.text((400, 280), filename, font=small_font, fill=(100, 0, 0), anchor="mm")
            
            # Add border
            draw.rectangle([(0, 0), (799, 449)], outline=(255, 0, 0), width=5)
            
            img_io = BytesIO()
            img.save(img_io, 'JPEG', quality=95)
            img_io.seek(0)
            
            return send_file(img_io, mimetype='image/jpeg')
        except:
            return "Internal server error", 500

@app.route('/api/check-image/<filename>', methods=['GET'])
def check_image(filename):
    """Check if an image exists and is valid"""
    try:
        # Security check to prevent directory traversal attacks
        if '..' in filename or filename.startswith('/'):
            return jsonify({'error': 'Invalid filename', 'valid': False}), 400
            
        # Only allow jpg/jpeg/png files
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            return jsonify({'error': 'Invalid file type', 'valid': False}), 400
            
        file_path = os.path.join(TEMP_IMG_DIR, filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            return jsonify({
                'error': 'File not found',
                'valid': False,
                'filename': filename,
                'requested_path': file_path
            }), 404
            
        # Check if file is valid image
        try:
            from PIL import Image
            
            # Get file size
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return jsonify({
                    'error': 'Empty file',
                    'valid': False,
                    'filename': filename,
                    'size': 0
                }), 400
                
            # Open the image to check format and get properties
            img = Image.open(file_path)
            width, height = img.size
            img_format = img.format
            img_mode = img.mode
            
            # Try to fully load image data to verify it's not corrupt
            img.load()
            
            # Check for reasonable dimensions
            if width < 10 or height < 10:
                return jsonify({
                    'error': 'Image too small',
                    'valid': False,
                    'filename': filename,
                    'dimensions': f'{width}x{height}',
                    'format': img_format,
                    'mode': img_mode,
                    'size': file_size
                }), 400
            
            # Get paths for both access methods
            static_path = f'/static/temp/{filename}'
            direct_path = f'/generated-image/{filename}'
            
            # Return successful validation with metadata
            return jsonify({
                'valid': True,
                'filename': filename,
                'path': static_path,
                'direct_path': direct_path,
                'size': file_size,
                'dimensions': f'{width}x{height}',
                'format': img_format,
                'mode': img_mode
            })
        except Exception as img_error:
            logger.error(f"Invalid image file: {filename}, Error: {img_error}")
            
            # Attempt to fix the image
            try:
                # Use the validation and fix function
                if validate_and_fix_image(file_path):
                    # Try opening the image again after fixing
                    img = Image.open(file_path)
                    width, height = img.size
                    img_format = img.format
                    img_mode = img.mode
                    file_size = os.path.getsize(file_path)
                    
                    # Return success with warning about the fix
                    return jsonify({
                        'valid': True,
                        'filename': filename,
                        'path': f'/static/temp/{filename}',
                        'direct_path': f'/generated-image/{filename}',
                        'size': file_size,
                        'dimensions': f'{width}x{height}',
                        'format': img_format,
                        'mode': img_mode,
                        'warning': 'Image was fixed during validation'
                    })
                else:
                    # If fixing failed
                    return jsonify({
                        'error': f'Invalid image file: {str(img_error)}',
                        'valid': False,
                        'filename': filename,
                        'fix_attempted': True,
                        'fix_successful': False
                    }), 400
            except Exception as fix_error:
                # If the fix attempt failed
                return jsonify({
                    'error': f'Invalid image file and fix failed: {str(fix_error)}',
                    'valid': False,
                    'filename': filename,
                    'original_error': str(img_error)
                }), 400
            
    except Exception as e:
        logger.error(f"Error checking image {filename}: {e}")
        return jsonify({'error': str(e), 'valid': False}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
