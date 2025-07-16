import os
import re
import logging
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import feedparser
import google.genai
from google.genai.types import HarmCategory, HarmBlockThreshold, GenerateImagesConfig
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

# Configure Gemini AI
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    logger.error("GEMINI_API_KEY not found in environment variables")
    raise ValueError("GEMINI_API_KEY is required")

client = google.genai.Client(api_key=api_key)

IMAGE_GENERATION_MODEL = 'gemini-1.5-pro-vision'

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
            
            response_data = [None]
            response_error = [None]
            request_completed = [False]
            
            def make_request():
                try:
                    response_data[0] = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[prompt]
                    )
                    request_completed[0] = True
                except Exception as e:
                    response_error[0] = e
                    request_completed[0] = True
            
            # Start request in a thread
            request_thread = threading.Thread(target=make_request)
            request_thread.start()
            
            # Wait for up to 10 seconds
            timeout = 10
            start_time = time.time()
            while not request_completed[0] and time.time() - start_time < timeout:
                time.sleep(0.1)
            
            if not request_completed[0]:
                logger.warning(f"Request timed out after {timeout} seconds")
                return {
                    "post_content": f"<h2>{title}</h2><p>{description}</p><p>The AI is still thinking about this topic. Please try again in a moment.</p>",
                    "image_description": f"A modern, professional illustration representing: {title}"
                }
            
            if response_error[0]:
                logger.error(f"Error in generate_content: {response_error[0]}")
                return {
                    "post_content": f"<h2>{title}</h2><p>{description}</p><p>Error generating content: {str(response_error[0])}</p>",
                    "image_description": f"A modern, professional illustration representing: {title}"
                }
                
            response = response_data[0]
            logger.info(f"Response received from Gemini. Has text: {hasattr(response, 'text')}")
            logger.info(f"Response type: {type(response)}")
            logger.info(f"Response attributes: {[attr for attr in dir(response) if not attr.startswith('_') and not callable(getattr(response, attr))]}") 
            
            # Try multiple ways to get the response content
            if hasattr(response, 'text') and response.text:
                text = response.text.strip()
                logger.info(f"Got response via .text ({len(text)} chars)")
            elif hasattr(response, 'content') and response.content:
                text = response.content.strip()
                logger.info(f"Got response via .content ({len(text)} chars)")
            elif hasattr(response, 'candidates') and response.candidates:
                # If we have candidates, get text from the first one
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    text = candidate.content.parts[0].text
                    logger.info(f"Got response via .candidates[0].content.parts[0].text ({len(text)} chars)")
                else:
                    logger.warning("Candidate structure doesn't match expected format")
                    text = str(candidate)
            else:
                logger.warning("Could not extract text from response")
                return {
                    "post_content": f"Blog post about: {title}\n\n{description}",
                    "image_description": f"A modern, professional illustration representing: {title}"
                }
            
            logger.info(f"Response text first 100 chars: {text[:100]}...")
            
            # Try to parse as JSON
            try:
                logger.info("Attempting to parse response as JSON")
                result = json.loads(text)
                logger.info(f"JSON parsing successful: {list(result.keys())}")
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error: {e}")
                # Try to extract JSON using regex
                match = re.search(r'{.*}', text, re.DOTALL)
                if match:
                    try:
                        result = json.loads(match.group(0))
                        logger.info(f"Regex JSON extraction successful: {list(result.keys())}")
                    except Exception as e:
                        logger.warning(f"Regex JSON parse failed: {e}")
                        result = None
                else:
                    logger.warning("No JSON-like content found in response")
                    result = None
                if not result:
                    # Fallback: extract post and image description heuristically
                    post_content = text
                    image_desc_match = re.search(r'"image_description"\s*:\s*"([^"]+)"', text)
                    image_description = image_desc_match.group(1) if image_desc_match else f"A modern, professional illustration representing: {title}"
                    result = {
                        "post_content": post_content,
                        "image_description": image_description
                    }
                    logger.info("Created result with heuristic extraction")
            
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
        """Fallback to placeholder image since no valid Gemini image model is available."""
        logger.warning("No valid Gemini image model available, using placeholder.")
        return {
            "image_url": "https://via.placeholder.com/800x600?text=" + image_description.replace(" ", "+"),
            "status": "error"
        }

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
        
        image_description = content_result.get('image_description', f"An illustration representing: {title}")
        logger.info(f"Image description: {image_description[:50]}...")
        
        # Generate image using Gemini image model
        logger.info("Generating image...")
        image_result = generator.generate_image(image_description)
        logger.info(f"Image result: {image_result}")
        
        response_data = {
            'title': title,
            'post_content': content_result.get('post_content', f"Blog post about: {title}\n\n{description}"),
            'image_description': image_description,
            'image_url': image_result.get('image_url', ''),
            'status': image_result.get('status', 'error')
        }
        
        logger.info(f"API Response prepared: {len(response_data['post_content'])} chars of content, image status: {response_data['status']}")
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error in generate_post endpoint: {e}", exc_info=True)
        return jsonify({
            'error': f"An error occurred: {str(e)}",
            'title': title if 'title' in locals() else "Error",
            'post_content': "Error generating content. Please try again.",
            'image_description': "Error",
            'image_url': "https://via.placeholder.com/800x600?text=Error",
            'status': "error"
        }), 500

@app.route('/api/test', methods=['GET'])
def test_api():
    """Test endpoint to check if API is working"""
    return jsonify({
        'status': 'success',
        'message': 'API is working correctly'
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)
