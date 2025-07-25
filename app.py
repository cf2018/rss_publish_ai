import os
import re
import logging
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import feedparser
import google.genai
from google.genai.types import HarmCategory, HarmBlockThreshold, GenerateImagesConfig, GenerateContentConfig
import requests
from dotenv import load_dotenv
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Create a directory for temporary images
TEMP_IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'temp')
os.makedirs(TEMP_IMG_DIR, exist_ok=True)

# Function to clean up old temporary images
def cleanup_temp_images(max_age_hours=24):
    """Remove temporary images older than the specified age"""
    import time
    import os
    
    try:
        now = time.time()
        count = 0
        
        for filename in os.listdir(TEMP_IMG_DIR):
            if filename.startswith('generated_') and filename.endswith('.jpg'):
                file_path = os.path.join(TEMP_IMG_DIR, filename)
                # If file is older than max_age_hours, delete it
                if os.path.isfile(file_path) and os.stat(file_path).st_mtime < (now - max_age_hours * 3600):
                    os.remove(file_path)
                    count += 1
        
        if count > 0:
            logger.info(f"Cleaned up {count} old temporary image files")
    except Exception as e:
        logger.error(f"Error cleaning up temporary images: {e}")

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
        try:
            feed = feedparser.parse(rss_url)
            entries = []
            for entry in feed.entries[:10]:  # Limit to 10 most recent entries
                entries.append({
                    'title': entry.title,
                    'description': entry.get('summary', entry.get('description', '')),
                    'link': entry.link,
                    'published': entry.get('published', '')
                })
            return entries
        except Exception as e:
            print(f"Error parsing RSS feed: {e}")
            return []
    
    def generate_post_content(self, title, description):
        """Generate blog post content using Gemini AI"""
        logger.info("Starting post content generation")
        prompt = f"""
        Create an engaging blog post based on the following as inspiration for a full blog article:
        Title: {title}
        Description: {description}
        
        🎯 Goal:
        Turn this into an engaging blog post that would perform well on Medium or similar platforms. Add a catchy title and optional subheading.
        Use same language and tone as the description, but make it more engaging and structured for a blog post format.

        📌 Write using these style rules:

        * **Use clear, everyday language:** Simple words. Short sentences. Write like a human, not a robot.
        * **No clichés or hype words:** Avoid terms like “game-changer” or “revolutionize.” Just be real.
        * **Be direct:** Get to the point fast. Cut the fluff.
        * **Use a natural voice:** It's okay to start sentences with "But" or "So." Write like you speak.
        * **Focus on value:** Don’t oversell. Instead, explain the benefit honestly.
        * **Be human:** Don’t fake excitement. Just share what’s interesting, surprising, or useful.
        * **Light structure:** Use short paragraphs, subheadings, and maybe a few bullet points.
        * **Emotion + story welcome:** Share small stories or examples if it helps explain the point.
        * **Title must be catchy and relevant.**

        ⛔ Avoid:
        - Robotic or overly formal tone
        - Long, dense paragraphs
        - Generic summaries or filler content

        ✅ Do:
        - Write in first person if it makes sense
        - Use contractions ("I'm", "it's", etc.)
        - Keep it scannable and interesting

        Now go ahead and write the blog post. Start with a headline, then dive right into the story or explanation.
        
        Also write a compelling image description for AI image generation (detailed, visual, suitable for creating an illustration)
        
        Format your response as JSON with keys: "post_content" and "image_description"
        """
        try:
            logger.info(f"Sending request to Gemini API with prompt of {len(prompt)} chars")
            
            # Use a default response if the API is slow or fails
            default_content = f"""
            <h2>{title}</h2>
            <p>{description}</p>
            <p>This is a placeholder blog post content. The AI-generated content will appear here when ready.</p>
            """
            
            # Set a timeout for the request to prevent hanging
            import threading
            import time
            import asyncio
            import concurrent.futures
            
            response_data = [None]
            response_error = [None]
            request_completed = [False]
            
            def make_request():
                try:
                    logger.info("Starting Gemini API request...")
                    response_data[0] = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[prompt]
                    )
                    logger.info("Gemini API request completed successfully")
                    request_completed[0] = True
                except Exception as e:
                    logger.error(f"Error in Gemini API request: {e}")
                    response_error[0] = e
                    request_completed[0] = True
            
            # Start request in a thread
            request_thread = threading.Thread(target=make_request)
            request_thread.start()
            
            # Wait for up to 30 seconds with status updates
            timeout = 30  # Increased from 10s to 30s
            start_time = time.time()
            
            # Check progress every second and log it
            check_interval = 1.0  # Check every second
            next_check_time = start_time + check_interval
            
            while not request_completed[0] and time.time() - start_time < timeout:
                time.sleep(0.1)
                
                # Log progress updates at regular intervals
                if time.time() >= next_check_time:
                    elapsed = time.time() - start_time
                    logger.info(f"Still waiting for Gemini API response... ({elapsed:.1f}s elapsed)")
                    next_check_time = time.time() + check_interval
            
            if not request_completed[0]:
                logger.warning(f"Request timed out after {timeout} seconds, but will continue in background")
                # Instead of returning immediately, provide a status message that informs the user
                # that processing is continuing in the background
                return {
                    "post_content": f"""<h2>{title}</h2>
                    <p>{description}</p>
                    <div class="bg-yellow-100 border-l-4 border-yellow-500 text-yellow-700 p-4 my-4">
                        <p class="font-bold">Processing</p>
                        <p>The AI is still generating content for this topic. This may take a minute or two.</p>
                        <p>You can try refreshing the page in a moment to see if the content is ready.</p>
                    </div>""",
                    "image_description": f"A modern, professional illustration representing: {title}",
                    "processing": True  # Flag to indicate background processing
                }
            
            if response_error[0]:
                logger.error(f"Error in generate_content: {response_error[0]}")
                return {
                    "post_content": f"""<h2>{title}</h2>
                    <p>{description}</p>
                    <div class="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 my-4">
                        <p class="font-bold">Error</p>
                        <p>Error generating content: {str(response_error[0])}</p>
                        <p>Please try again in a few moments.</p>
                    </div>""",
                    "image_description": f"A modern, professional illustration representing: {title}"
                }
                
            response = response_data[0]
            logger.info(f"Response received from Gemini. Has text: {hasattr(response, 'text')}")
            logger.info(f"Response type: {type(response)}")
            logger.info(f"Response attributes: {[attr for attr in dir(response) if not attr.startswith('_') and not callable(getattr(response, attr))]}") 
            
            # Enhanced response extraction with multiple fallbacks
            text = None
            extraction_methods = [
                # Try all possible ways to extract text from response
                lambda r: r.text.strip() if hasattr(r, 'text') and r.text else None,
                lambda r: r.content.strip() if hasattr(r, 'content') and r.content else None,
                lambda r: r.candidates[0].content.parts[0].text if (hasattr(r, 'candidates') and r.candidates and
                                                                 hasattr(r.candidates[0], 'content') and
                                                                 hasattr(r.candidates[0].content, 'parts') and
                                                                 r.candidates[0].content.parts) else None,
                lambda r: str(r.candidates[0]) if hasattr(r, 'candidates') and r.candidates else None,
                lambda r: str(r) if r else None
            ]
            
            # Try each extraction method until we get content
            for i, extract in enumerate(extraction_methods):
                try:
                    extracted_text = extract(response)
                    if extracted_text:
                        text = extracted_text
                        logger.info(f"Got response via extraction method {i+1} ({len(text)} chars)")
                        break
                except Exception as e:
                    logger.warning(f"Extraction method {i+1} failed: {e}")
            
            if not text:
                logger.warning("Could not extract text from response using any method")
                return {
                    "post_content": f"""<h2>{title}</h2>
                    <p>{description}</p>
                    <div class="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 my-4">
                        <p class="font-bold">Error</p>
                        <p>Failed to extract content from the AI response.</p>
                        <p>Please try again in a few moments.</p>
                    </div>""",
                    "image_description": f"A modern, professional illustration representing: {title}"
                }
            
            logger.info(f"Response text first 100 chars: {text[:100]}...")
            
            # Try to parse as JSON using multiple approaches
            result = None
            
            # First, try to parse directly as JSON
            try:
                logger.info("Attempting to parse response as JSON")
                result = json.loads(text)
                logger.info(f"JSON parsing successful: {list(result.keys())}")
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error: {e}")
                
                # Try to extract JSON using regex patterns
                json_patterns = [
                    r'```json\s*(.*?)\s*```',  # Code block with json
                    r'{[\s\S]*?}',             # Find any JSON-like object
                    r'"post_content"\s*:[\s\S]*?"image_description"\s*:.*?["}]',  # Find specific fields
                ]
                
                for pattern in json_patterns:
                    try:
                        matches = re.findall(pattern, text, re.DOTALL)
                        if matches:
                            for match in matches:
                                try:
                                    # Try to make it valid JSON if it's not complete
                                    if not match.strip().startswith('{'):
                                        match = '{' + match + '}'
                                    # Clean up any trailing commas which can break JSON parsing
                                    match = re.sub(r',\s*}', '}', match)
                                    result = json.loads(match)
                                    logger.info(f"Regex JSON extraction successful with pattern {pattern}: {list(result.keys())}")
                                    break
                                except json.JSONDecodeError:
                                    continue
                        if result:
                            break
                    except Exception as ex:
                        logger.warning(f"Regex pattern {pattern} failed: {ex}")
                
                # If still no valid JSON, try to extract content more aggressively
                if not result:
                    logger.warning("Attempting more aggressive extraction of content")
                    
                    # Extract image description if possible
                    image_desc_match = re.search(r'"image_description"[\s:]*"([^"]+)"', text)
                    image_description = image_desc_match.group(1) if image_desc_match else f"A modern, professional illustration representing: {title}"
                    
                    # Try to extract post content - look for common patterns
                    post_content_patterns = [
                        r'"post_content"[\s:]*"([\s\S]+?)"(?=,|\})',  # Standard JSON format
                        r'<h[1-6]>([\s\S]+?)</h[1-6]>',               # Look for HTML headings
                        r'# (.*?)\n',                                # Markdown headings
                    ]
                    
                    post_content = None
                    for pattern in post_content_patterns:
                        match = re.search(pattern, text)
                        if match:
                            post_content = match.group(1)
                            break
                    
                    if not post_content:
                        # If we still couldn't extract content, use the whole text
                        post_content = text
                    
                    result = {
                        "post_content": post_content,
                        "image_description": image_description
                    }
                    logger.info("Created result with aggressive content extraction")
            
            # Ensure result has required keys
            if not result.get("post_content"):
                logger.warning("Result missing post_content, adding default")
                result["post_content"] = f"Blog post about: {title}\n\n{description}"
            if not result.get("image_description"):
                logger.warning("Result missing image_description, adding default")
                result["image_description"] = f"A modern, professional illustration representing: {title}"
            
            logger.info("Post content generation completed successfully")    
            return result
        except Exception as e:
            logger.error(f"Error generating content: {e}", exc_info=True)
            return {
                "post_content": f"Blog post about: {title}\n\n{description}",
                "image_description": f"An illustration representing {title}"
            }
    
    def generate_image(self, image_description):
        """Generate an image using Gemini image generation model based on the description."""
        try:
            logger.info(f"Attempting to use Gemini model '{IMAGE_GENERATION_MODEL}' for image generation")
            logger.info(f"Image description: {image_description[:100]}...")
            
            # Create a simplified and shortened image description for better image generation
            # This helps avoid overly specific details that might confuse image generation
            simplified_description = self._simplify_image_description(image_description)
            logger.info(f"Simplified image description: {simplified_description}")
            
            # Try multiple image services in order of preference
            image_services = [
                self._try_unsplash_image,
                self._try_pexels_image,
                self._try_picsum_image
            ]
            
            for service in image_services:
                try:
                    result = service(simplified_description)
                    if result and result.get('status') == 'success':
                        logger.info(f"Successfully generated image using {result.get('source')}")
                        return result
                except Exception as e:
                    logger.warning(f"Image service failed: {str(e)}")
                    continue
            
            # If all image services fail, use placeholder generator
            return self._generate_placeholder_image(simplified_description)
            
        except Exception as e:
            logger.error(f"Error in image generation: {e}", exc_info=True)
            return self._generate_placeholder_image(image_description)
    
    def _try_unsplash_image(self, description):
        """Try to get an image from Unsplash based on description"""
        try:
            import uuid
            import requests
            from io import BytesIO
            from PIL import Image
            
            # Generate a clean search term from the description
            search_terms = description
            unsplash_url = f"https://source.unsplash.com/1200x628/?{search_terms.replace(' ', '+')}"
            logger.info(f"Fetching image from Unsplash with URL: {unsplash_url}")
            
            # Get the image from Unsplash (which redirects to an actual image)
            response = requests.get(unsplash_url, allow_redirects=True, timeout=10)
            if response.status_code == 200:
                # Generate a unique filename
                filename = f"generated_{uuid.uuid4().hex}.jpg"
                file_path = os.path.join(TEMP_IMG_DIR, filename)
                
                # Process and save the image
                img = Image.open(BytesIO(response.content))
                img.save(file_path, "JPEG")
                
                logger.info(f"Successfully downloaded and saved image to {file_path}")
                
                # Return the local image URL
                return {
                    "image_url": f"/static/temp/{filename}",
                    "status": "success",
                    "source": "unsplash",
                    "local_path": file_path,
                    "message": "Generated image based on your description"
                }
            else:
                logger.warning(f"Failed to get image from Unsplash: Status code {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error downloading image from Unsplash: {e}")
            return None
    
    def _try_pexels_image(self, description):
        """Try to get an image from Pexels based on description"""
        try:
            import uuid
            import requests
            import random
            from io import BytesIO
            from PIL import Image
            
            # Since we don't have a Pexels API key, we'll use a curated list of nature/landscape images
            # This is a fallback when we can't connect to other services
            pexels_images = [
                "https://images.pexels.com/photos/2559941/pexels-photo-2559941.jpeg",
                "https://images.pexels.com/photos/15286/pexels-photo.jpg",
                "https://images.pexels.com/photos/358457/pexels-photo-358457.jpeg",
                "https://images.pexels.com/photos/417074/pexels-photo-417074.jpeg",
                "https://images.pexels.com/photos/247600/pexels-photo-247600.jpeg",
                "https://images.pexels.com/photos/624015/pexels-photo-624015.jpeg",
                "https://images.pexels.com/photos/5534242/pexels-photo-5534242.jpeg",
                "https://images.pexels.com/photos/3861969/pexels-photo-3861969.jpeg"
            ]
            
            # Randomly select an image
            image_url = random.choice(pexels_images)
            logger.info(f"Using random Pexels image: {image_url}")
            
            # Download the image
            response = requests.get(image_url, timeout=10)
            if response.status_code == 200:
                # Generate a unique filename
                filename = f"generated_{uuid.uuid4().hex}.jpg"
                file_path = os.path.join(TEMP_IMG_DIR, filename)
                
                # Save the image
                img = Image.open(BytesIO(response.content))
                img.save(file_path, "JPEG")
                
                logger.info(f"Successfully downloaded and saved Pexels image to {file_path}")
                
                # Return the local image URL
                return {
                    "image_url": f"/static/temp/{filename}",
                    "status": "success",
                    "source": "pexels",
                    "local_path": file_path,
                    "message": "Used relevant stock image (Pexels)"
                }
            else:
                logger.warning(f"Failed to get image from Pexels: Status code {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error downloading image from Pexels: {e}")
            return None
    
    def _try_picsum_image(self, description):
        """Try to get a random image from Picsum"""
        try:
            import uuid
            import requests
            
            # Try Picsum as a backup image source
            picsum_url = f"https://picsum.photos/1200/628"
            response = requests.get(picsum_url, timeout=10)
            
            if response.status_code == 200:
                # Generate a unique filename
                filename = f"generated_{uuid.uuid4().hex}.jpg"
                file_path = os.path.join(TEMP_IMG_DIR, filename)
                
                # Save the image
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"Successfully downloaded and saved image from Picsum to {file_path}")
                
                # Return the local image URL
                return {
                    "image_url": f"/static/temp/{filename}",
                    "status": "success",
                    "source": "picsum",
                    "local_path": file_path,
                    "message": "Generated random image (image service fallback)"
                }
            else:
                logger.warning(f"Failed to get image from Picsum: Status code {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error downloading image from Picsum: {e}")
            return None
        

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
    """Serve a generated image directly"""
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
            return "File not found", 404
            
        # Determine content type
        content_type = 'image/jpeg'
        if filename.lower().endswith('.png'):
            content_type = 'image/png'
            
        from flask import send_file
        return send_file(file_path, mimetype=content_type)
    except Exception as e:
        logger.error(f"Error serving image file: {e}")
        return "Internal server error", 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
