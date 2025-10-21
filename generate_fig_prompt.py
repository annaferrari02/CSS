import PIL
from PIL import Image, ImageDraw, ImageFont
import textwrap

# Create image with white background
width, height = 1200, 650
img = Image.new('RGB', (width, height), color='white')
draw = ImageDraw.Draw(img)

# Draw blue header
header_height = 50
draw.rectangle([(0, 0), (width, header_height)], fill='#0000CC')

# Try to use a monospace font, fallback to default
try:
    font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 20)
    font_text = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 16)
except:
    font_title = ImageFont.load_default()
    font_text = ImageFont.load_default()

# Draw title
draw.text((20, 15), "Mistral-AI Prompt", fill='white', font=font_title)

# Prompt text
prompt_text = """Given these top keywords from a topic in user-chatbot conversations:
{terms_str}

Generate a short, descriptive label (max 6 words) that captures the main theme.
Examples:
- "Affirmations & confirmations"
- "Platform references (Replika, Reddit)"
- "Role-play commands & fantasy RP"
- "Erotic / affectionate descriptions"
- "Greetings & introductions"

Return ONLY the label, nothing else.

Label:"""

# Draw text with proper spacing
y_offset = header_height + 30
line_height = 25

for line in prompt_text.split('\n'):
    draw.text((30, y_offset), line, fill='black', font=font_text)
    y_offset += line_height

# Save image
img.save('mistral_prompt.png')
print("Image saved as 'mistral_prompt.png'")