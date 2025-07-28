"""
Utility functions for text generation with AI models.
"""
import os
import re
import json
import logging
import threading
import time

# Google Gemini imports
import google.genai
from google.genai.types import HarmCategory, HarmBlockThreshold, GenerateContentConfig

# Configure logging
logger = logging.getLogger(__name__)

# Available Gemini models
TEXT_MODEL = 'gemini-2.5-flash'  # For text generation

def simplify_image_description(description):
    """Simplify an image description for better image generation results.
    
    This function takes a detailed image description and creates a simplified version
    that works better for image generation services.
    
    Args:
        description (str): The original image description
        
    Returns:
        str: A simplified image description
    """
    # Check if the description is too long
    if len(description) > 150:
        # Extract the most important part (first 150 chars)
        simplified = description[:150].strip()
        
        # Try to end at a sentence boundary
        if '.' in simplified:
            simplified = simplified.rsplit('.', 1)[0] + '.'
        
        # Clean up any trailing commas or other punctuation
        simplified = simplified.rstrip(',;:-')
        
        # Remove any markdown-like formatting
        simplified = re.sub(r'[*_#]', '', simplified)
        
        return simplified
    
    return description

def generate_post_content(client, title, description):
    """Generate blog post content using Gemini AI
    
    Args:
        client: The Google Gemini AI client
        title (str): The title of the blog post
        description (str): The description of the blog post
        
    Returns:
        dict: A dictionary containing the post_content and image_description
    """
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
    * **No clichés or hype words:** Avoid terms like "game-changer" or "revolutionize." Just be real.
    * **Be direct:** Get to the point fast. Cut the fluff.
    * **Use a natural voice:** It's okay to start sentences with "But" or "So." Write like you speak.
    * **Focus on value:** Don't oversell. Instead, explain the benefit honestly.
    * **Be human:** Don't fake excitement. Just share what's interesting, surprising, or useful.
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
        import asyncio
        import concurrent.futures
        
        response_data = [None]
        response_error = [None]
        request_completed = [False]
        
        def make_request():
            try:
                logger.info("Starting Gemini API request...")
                response_data[0] = client.models.generate_content(
                    model=TEXT_MODEL,
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

def parse_rss_feed(rss_url):
    """Parse RSS feed and return list of entries
    
    Args:
        rss_url (str): The URL of the RSS feed
        
    Returns:
        list: A list of dictionaries containing RSS entry data
    """
    try:
        import feedparser
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
        logger.error(f"Error parsing RSS feed: {e}")
        return []
