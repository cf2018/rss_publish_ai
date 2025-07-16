# RSS AI Post Generator

A modern Flask web application that transforms RSS feeds or manual input into AI-generated blog posts with accompanying images using Google's Gemini AI.

## Features

- 🎨 **Modern Responsive UI** - Beautiful, mobile-friendly interface built with Tailwind CSS
- 📰 **RSS Feed Integration** - Parse RSS feeds and select articles to transform
- ✍️ **Manual Input** - Create content from custom titles and descriptions
- 🤖 **AI-Powered Content** - Generate engaging blog posts using Google Gemini AI
- 🖼️ **Image Generation** - Create relevant images for your posts (placeholder implementation)
- 📱 **Responsive Design** - Works seamlessly on desktop, tablet, and mobile devices

## Setup Instructions

### 1. Clone and Setup Environment

```bash
cd /home/luch/repos/rss_publish_ai
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Edit the `.env` file and add your API keys:

```env
SECRET_KEY=your-secret-key-here
GEMINI_API_KEY=your-gemini-api-key-here
```

### 3. Get Google Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file

### 4. Run the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Usage

### Manual Input Method
1. Click on the "Manual Input" tab
2. Enter your blog post title
3. Add a brief description
4. Click "Generate Content"

### RSS Feed Method
1. Click on the "RSS Feed" tab
2. Enter an RSS feed URL (e.g., `https://feeds.feedburner.com/example`)
3. Click "Load RSS Feed"
4. Select an article from the dropdown
5. Click "Generate Content from Selection"

## API Endpoints

- `GET /` - Main landing page
- `POST /api/parse-rss` - Parse RSS feed and return entries
- `POST /api/generate-post` - Generate blog post and image from title/description

## Customization

### Adding Image Generation Services

The application currently uses a placeholder for image generation. To integrate with real image generation services:

1. **OpenAI DALL-E**: Add OpenAI API integration in the `generate_image()` method
2. **Stability AI**: Integrate Stable Diffusion API
3. **Midjourney**: Use Midjourney API (if available)

Example integration points are marked in `app.py`.

### Styling Customization

The UI uses Tailwind CSS for styling. Key customization areas:

- **Colors**: Modify the gradient classes in the HTML
- **Layout**: Adjust the grid and spacing classes
- **Components**: Customize cards, buttons, and form elements

## Technologies Used

- **Backend**: Flask (Python)
- **Frontend**: HTML5, Tailwind CSS, JavaScript
- **AI**: Google Gemini AI
- **RSS Parsing**: feedparser
- **Styling**: Tailwind CSS, Font Awesome icons

## Project Structure

```
rss_publish_ai/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── templates/
│   └── index.html        # Main HTML template
└── venv/                 # Virtual environment
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).
