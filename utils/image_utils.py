"""
Utility functions for image generation and processing.
"""
import os
import re
import uuid
import random
import logging
import requests
import time
import hashlib
import textwrap
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageFile

# Configure logging
logger = logging.getLogger(__name__)

# Enable loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_temp_image_dir():
    """Get the directory for temporary images, creating it if needed."""
    temp_img_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static', 'temp')
    os.makedirs(temp_img_dir, exist_ok=True)
    return temp_img_dir

def cleanup_temp_images(max_age_hours=24):
    """Remove temporary images older than the specified age"""
    temp_img_dir = get_temp_image_dir()
    
    try:
        now = time.time()
        count = 0
        
        for filename in os.listdir(temp_img_dir):
            if filename.startswith('generated_') and filename.endswith('.jpg'):
                file_path = os.path.join(temp_img_dir, filename)
                # If file is older than max_age_hours, delete it
                if os.path.isfile(file_path) and os.stat(file_path).st_mtime < (now - max_age_hours * 3600):
                    os.remove(file_path)
                    count += 1
        
        if count > 0:
            logger.info(f"Cleaned up {count} old temporary image files")
    except Exception as e:
        logger.error(f"Error cleaning up temporary images: {e}")

def simplify_image_description(image_description):
    """Create a simplified and shortened image description for better image generation.
    
    Args:
        image_description (str): The original image description
        
    Returns:
        str: A simplified image description for better results
    """
    # If description is too long, simplify it
    if len(image_description) > 100:
        # Take first 100 characters and try to find a sentence boundary
        simplified = image_description[:100].strip()
        if '.' in simplified:
            simplified = simplified.split('.')[0] + '.'
        
        # Clean up any markdown formatting
        simplified = re.sub(r'[*_]', '', simplified)
        
        return simplified
    
    return image_description

def validate_and_fix_image(file_path):
    """Validates image file and attempts to fix any issues
    
    Args:
        file_path (str): Path to the image file
        
    Returns:
        bool: True if image is valid or was fixed, False otherwise
    """
    try:
        import os
        import shutil
        
        if not os.path.exists(file_path):
            logger.error(f"Image file not found: {file_path}")
            return False
            
        if os.path.getsize(file_path) == 0:
            logger.error(f"Image file is empty: {file_path}")
            return False
        
        # Create a backup of the original file
        backup_path = f"{file_path}.bak"
        try:
            shutil.copy2(file_path, backup_path)
        except Exception as backup_err:
            logger.warning(f"Could not create backup of image: {backup_err}")
    
        try:
            # Try to open and verify the image
            with Image.open(file_path) as img:
                # Force load image data
                img.load()
                
                # Check for reasonable dimensions
                if img.width < 10 or img.height < 10:
                    logger.warning(f"Image too small: {img.width}x{img.height}")
                    # Try to restore from backup if dimensions are bad
                    if os.path.exists(backup_path):
                        shutil.copy2(backup_path, file_path)
                    return False
                
                # Ensure it's in a web-friendly format (JPEG)
                if img.format not in ['JPEG', 'PNG']:
                    # Convert to RGB mode if needed and save as JPEG
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(file_path, 'JPEG', quality=95)
                    logger.info(f"Converted image to JPEG: {file_path}")
                
                # If image is too large or too small, resize it to reasonable dimensions
                max_dimension = 1500
                min_dimension = 300
                if img.width > max_dimension or img.height > max_dimension:
                    # Resize while maintaining aspect ratio
                    ratio = min(max_dimension / img.width, max_dimension / img.height)
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size, Image.LANCZOS)
                    img.save(file_path, format=img.format or 'JPEG', quality=95)
                    logger.info(f"Resized large image to {new_size}")
                elif img.width < min_dimension or img.height < min_dimension:
                    # Try to scale up small images
                    ratio = max(min_dimension / img.width, min_dimension / img.height)
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size, Image.LANCZOS)
                    img.save(file_path, format=img.format or 'JPEG', quality=95)
                    logger.info(f"Resized small image to {new_size}")
                    
                # Clean up the backup
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                    
                return True
        except Exception as img_err:
            logger.error(f"Image validation error: {img_err}")
            
            # Try to recover corrupted image
            try:
                # First check if we have a backup
                if os.path.exists(backup_path):
                    logger.info(f"Attempting to restore from backup: {backup_path}")
                    shutil.copy2(backup_path, file_path)
                    os.remove(backup_path)
                    
                    # Try opening the restored image
                    img = Image.open(file_path)
                    img.verify()
                    return True
                    
                # For partially downloaded/corrupted images, try to recover
                img = Image.open(file_path)
                img = img.copy()  # This can sometimes recover corrupted images
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(file_path, 'JPEG', quality=90)
                logger.info(f"Attempted to recover corrupted image: {file_path}")
                return True
            except Exception as recover_err:
                logger.error(f"Failed to recover corrupted image: {file_path} - {recover_err}")
                return False
            
    except Exception as e:
        logger.error(f"Error validating image: {e}")
        return False

def try_unsplash_image(description):
    """Try to get an image from Unsplash based on description
    
    Args:
        description (str): The image description to use for search
        
    Returns:
        dict: A dictionary with image information or None if failed
    """
    try:
        temp_img_dir = get_temp_image_dir()
        
        # Generate a clean search term from the description
        search_terms = description
        unsplash_url = f"https://source.unsplash.com/1200x628/?{search_terms.replace(' ', '+')}"
        logger.info(f"Fetching image from Unsplash with URL: {unsplash_url}")
        
        # Get the image from Unsplash (which redirects to an actual image)
        response = requests.get(unsplash_url, allow_redirects=True, timeout=15)  # Extended timeout
        if response.status_code == 200:
            # Generate a unique filename
            filename = f"generated_{uuid.uuid4().hex}.jpg"
            file_path = os.path.join(temp_img_dir, filename)
            
            # Process and save the image
            try:
                img = Image.open(BytesIO(response.content))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(file_path, "JPEG", quality=95)
                
                # Validate the saved image
                if not validate_and_fix_image(file_path):
                    logger.warning(f"Image validation failed for Unsplash image")
                    return None
                
                logger.info(f"Successfully downloaded and saved image to {file_path}")
                
                # Return the local image URL
                return {
                    "image_url": f"/static/temp/{filename}",
                    "status": "success",
                    "source": "unsplash",
                    "local_path": file_path,
                    "message": "Generated image based on your description"
                }
            except Exception as img_err:
                logger.error(f"Error processing Unsplash image: {img_err}")
                # Try direct save if PIL processing fails
                try:
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    if validate_and_fix_image(file_path):
                        return {
                            "image_url": f"/static/temp/{filename}",
                            "status": "success",
                            "source": "unsplash",
                            "local_path": file_path,
                            "message": "Generated image based on your description"
                        }
                except:
                    pass
                return None
        else:
            logger.warning(f"Failed to get image from Unsplash: Status code {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error downloading image from Unsplash: {e}")
        return None

def try_pexels_image(description):
    """Try to get an image from Pexels based on description
    
    Args:
        description (str): The image description
        
    Returns:
        dict: A dictionary with image information or None if failed
    """
    try:
        temp_img_dir = get_temp_image_dir()
        
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
            "https://images.pexels.com/photos/3861969/pexels-photo-3861969.jpeg",
            # Additional reliable images to improve fallback success rate
            "https://images.pexels.com/photos/268533/pexels-photo-268533.jpeg",
            "https://images.pexels.com/photos/1287145/pexels-photo-1287145.jpeg",
            "https://images.pexels.com/photos/2662116/pexels-photo-2662116.jpeg",
            "https://images.pexels.com/photos/1072179/pexels-photo-1072179.jpeg"
        ]
        
        # Shuffle list for better distribution
        random.shuffle(pexels_images)
        
        # Try multiple images until one works
        for image_url in pexels_images[:3]:  # Try up to 3 images
            logger.info(f"Trying Pexels image: {image_url}")
            
            try:
                # Download the image with longer timeout
                response = requests.get(image_url, timeout=15)
                if response.status_code != 200:
                    logger.warning(f"Failed to get Pexels image: Status code {response.status_code}")
                    continue
                
                # Generate a unique filename
                filename = f"generated_{uuid.uuid4().hex}.jpg"
                file_path = os.path.join(temp_img_dir, filename)
                
                # Save the image
                try:
                    img = Image.open(BytesIO(response.content))
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(file_path, "JPEG", quality=95)
                    
                    # Validate the saved image
                    if not validate_and_fix_image(file_path):
                        logger.warning(f"Image validation failed for Pexels image")
                        continue
                    
                    logger.info(f"Successfully downloaded and saved Pexels image to {file_path}")
                    
                    # Return the local image URL
                    return {
                        "image_url": f"/static/temp/{filename}",
                        "status": "success",
                        "source": "pexels",
                        "local_path": file_path,
                        "message": "Used relevant stock image (Pexels)"
                    }
                except Exception as img_err:
                    logger.error(f"Error processing Pexels image: {img_err}")
                    # Try direct save if PIL processing fails
                    try:
                        with open(file_path, 'wb') as f:
                            f.write(response.content)
                        if validate_and_fix_image(file_path):
                            return {
                                "image_url": f"/static/temp/{filename}",
                                "status": "success",
                                "source": "pexels",
                                "local_path": file_path,
                                "message": "Used relevant stock image (Pexels)"
                            }
                    except:
                        pass
                    continue
            except Exception as req_err:
                logger.error(f"Error requesting Pexels image: {req_err}")
                continue
                
        logger.warning("All Pexels image attempts failed")
        return None
    except Exception as e:
        logger.error(f"Error in Pexels image service: {e}")
        return None

def try_picsum_image(description):
    """Try to get a random image from Picsum
    
    Args:
        description (str): The image description (not used for Picsum)
        
    Returns:
        dict: A dictionary with image information or None if failed
    """
    try:
        temp_img_dir = get_temp_image_dir()
        
        # Try multiple Picsum URLs to increase chances of success
        # Include both specific IDs and random generation
        picsum_urls = [
            f"https://picsum.photos/1200/628",  # Random image
            f"https://picsum.photos/seed/{int(time.time())}/1200/628",  # Seeded random image
            f"https://picsum.photos/id/237/1200/628",  # Specific image (dog)
            f"https://picsum.photos/id/1015/1200/628",  # Specific image (mountain)
            f"https://picsum.photos/id/1018/1200/628"   # Specific image (nature)
        ]
        
        for picsum_url in picsum_urls:
            logger.info(f"Trying Picsum image: {picsum_url}")
            
            try:
                response = requests.get(picsum_url, timeout=15)
                if response.status_code != 200:
                    logger.warning(f"Failed to get image from Picsum: Status code {response.status_code}")
                    continue
                
                # Generate a unique filename
                filename = f"generated_{uuid.uuid4().hex}.jpg"
                file_path = os.path.join(temp_img_dir, filename)
                
                # First try to process with PIL to catch issues early
                try:
                    img = Image.open(BytesIO(response.content))
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(file_path, "JPEG", quality=95)
                except Exception as pil_err:
                    logger.warning(f"PIL processing failed for Picsum image, trying direct save: {pil_err}")
                    # Try direct save as fallback
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                
                # Validate the saved image
                if not validate_and_fix_image(file_path):
                    logger.warning(f"Image validation failed for Picsum image")
                    continue
                
                logger.info(f"Successfully downloaded and saved image from Picsum to {file_path}")
                
                # Return the local image URL
                return {
                    "image_url": f"/static/temp/{filename}",
                    "status": "success",
                    "source": "picsum",
                    "local_path": file_path,
                    "message": "Generated random image (image service fallback)"
                }
            except Exception as req_err:
                logger.error(f"Error with Picsum URL {picsum_url}: {req_err}")
                continue
        
        logger.warning("All Picsum image attempts failed")
        return None
    except Exception as e:
        logger.error(f"Error in Picsum image service: {e}")
        return None

def generate_placeholder_image(image_description):
    """Generate a placeholder image when all other image services fail.
    
    Args:
        image_description (str): Description to use for the placeholder
        
    Returns:
        dict: A dictionary with image information
    """
    try:
        temp_img_dir = get_temp_image_dir()
        logger.info("Falling back to locally generated placeholder image")
        
        # Create a custom placeholder image locally rather than depending on external service
        try:
            # Generate a random seed based on the description
            seed = int(hashlib.md5(image_description.encode()).hexdigest(), 16) % 1000
            random.seed(seed)
            
            # Create a wider range of visually appealing colors
            color_themes = [
                # Blue/Purple gradient
                ((66, 126, 234), (118, 75, 162), (255, 255, 255)),
                # Green/Teal gradient
                ((46, 204, 113), (26, 188, 156), (255, 255, 255)),
                # Orange/Yellow gradient
                ((230, 126, 34), (241, 196, 15), (0, 0, 0)),
                # Red/Pink gradient
                ((231, 76, 60), (233, 30, 99), (255, 255, 255)),
                # Cool Blue gradient
                ((52, 152, 219), (41, 128, 185), (255, 255, 255))
            ]
            
            # Select a random theme
            theme_idx = seed % len(color_themes)
            color1, color2, text_color = color_themes[theme_idx]
            
            # Create gradient image
            width, height = 800, 450
            img = Image.new('RGB', (width, height), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            
            # Draw gradient background
            for y in range(height):
                # Calculate gradient ratio
                ratio = y / height
                r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
                g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
                b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
                
                # Draw horizontal line with calculated color
                draw.line([(0, y), (width, y)], fill=(r, g, b))
            
            # Try to load a font, fall back to default if needed
            font_size = 36
            small_font_size = 24
            font = None
            small_font = None
            
            font_paths = [
                # Standard font locations on various systems
                "DejaVuSans.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/TTF/DejaVuSans.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
                "C:\\Windows\\Fonts\\Arial.ttf",
                # Add more font paths as needed
            ]
            
            for font_path in font_paths:
                try:
                    if os.path.exists(font_path):
                        font = ImageFont.truetype(font_path, font_size)
                        small_font = ImageFont.truetype(font_path, small_font_size)
                        break
                except Exception:
                    continue
            
            if font is None:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # Add a translucent overlay rectangle in center for better text contrast
            overlay_height = 200
            overlay_y = (height - overlay_height) // 2
            
            # Draw a semi-transparent rectangle for better text contrast
            draw.rectangle([(0, overlay_y), (width, overlay_y + overlay_height)], 
                          fill=(255, 255, 255, 128))
            
            # Trim description to fit and wrap text
            short_desc = image_description
            if len(short_desc) > 100:
                short_desc = short_desc[:97] + "..."
            
            wrapped_text = textwrap.fill(short_desc, width=40)
            
            # Calculate text position (centered)
            text_y = height // 2 - (wrapped_text.count('\n') + 1) * font_size // 2
            
            # Add text with a subtle shadow for better visibility
            for line in wrapped_text.split('\n'):
                # Calculate width for centering
                line_width = 0
                if hasattr(font, 'getlength'):
                    line_width = font.getlength(line)
                elif hasattr(font, 'getbbox'):
                    line_width = font.getbbox(line)[2]
                else:
                    line_width = width // 2  # Fallback
                    
                text_x = (width - line_width) // 2
                
                # Draw shadow
                draw.text((text_x + 2, text_y + 2), line, font=font, fill=(0, 0, 0))
                # Draw text
                draw.text((text_x, text_y), line, font=font, fill=text_color)
                
                text_y += font_size + 10
            
            # Add decorative elements
            for i in range(10):
                pos1 = (random.randint(0, width), random.randint(0, height))
                pos2 = (random.randint(0, width), random.randint(0, height))
                draw.line([pos1, pos2], fill=(255, 255, 255), width=1)
            
            # Add attribution
            attribution_text = "AI Generated Placeholder Image"
            draw.text((20, height - small_font_size - 20), attribution_text, font=small_font, fill=text_color)
            
            # Save image
            filename = f"generated_placeholder_{uuid.uuid4().hex}.jpg"
            file_path = os.path.join(temp_img_dir, filename)
            img.save(file_path, "JPEG", quality=95)
            
            # Validate the saved image
            if validate_and_fix_image(file_path):
                return {
                    "image_url": f"/static/temp/{filename}",
                    "status": "placeholder",
                    "source": "local",
                    "local_path": file_path,
                    "message": "Using locally generated placeholder image"
                }
            else:
                logger.error("Failed to validate locally generated placeholder image")
                # If validation fails, fall through to external placeholder
        except Exception as local_err:
            logger.error(f"Error creating local placeholder: {local_err}")
            # Fall back to external placeholder service
        
        # External placeholder fallback
        try:
            # Encode the description to create a custom placeholder image
            encoded_desc = image_description.replace(" ", "+")[:100]  # Limit to 100 chars
            
            # Generate colors for external placeholder
            seed = int(hashlib.md5(image_description.encode()).hexdigest(), 16) % 1000
            bg_color = f"{(seed * 123) % 255:02x}{(seed * 231) % 255:02x}{(seed * 321) % 255:02x}"
            text_color = f"{255-(seed * 123) % 255:02x}{255-(seed * 231) % 255:02x}{255-(seed * 321) % 255:02x}"
            
            image_url = f"https://via.placeholder.com/800x450/{bg_color}/{text_color}?text={encoded_desc}"
            
            return {
                "image_url": image_url,
                "status": "placeholder",
                "source": "external",
                "message": "Using external placeholder image service"
            }
        except Exception as external_err:
            logger.error(f"Error creating external placeholder: {external_err}")
            
    except Exception as e:
        logger.error(f"Error in placeholder image generation: {e}")
    
    # Ultimate fallback if even our custom placeholder fails
    return {
        "image_url": "https://via.placeholder.com/800x450/cccccc/333333?text=Image+Generation+Unavailable",
        "status": "error",
        "source": "fallback",
        "message": "Failed to generate any image"
    }

def generate_image(image_description):
    """Generate an image using various image services based on the description.
    
    Args:
        image_description (str): Description of the image to generate
        
    Returns:
        dict: A dictionary with the image URL and other metadata
    """
    try:
        logger.info(f"Attempting to generate image for description: {image_description[:100]}...")
        
        # Create a simplified and shortened image description for better image generation
        simplified_description = simplify_image_description(image_description)
        logger.info(f"Simplified image description: {simplified_description}")
        
        # Try multiple image services in order of preference
        image_services = [
            try_unsplash_image,
            try_pexels_image,
            try_picsum_image
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
        return generate_placeholder_image(simplified_description)
        
    except Exception as e:
        logger.error(f"Error in image generation: {e}", exc_info=True)
        return generate_placeholder_image(image_description)
