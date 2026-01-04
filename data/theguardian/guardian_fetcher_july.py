import requests
import json
import hashlib
import re
from datetime import datetime, timedelta
from urllib.parse import quote
import time
import os
from apikey import API_KEY

def tokenize_text(text):
    """
    Simple tokenization that mimics the format in the reference file
    """
    if not text:
        return []
    
    # Convert to lowercase and remove HTML tags
    text = re.sub(r'<[^>]+>', '', text.lower())
    
    # Replace common contractions and special characters
    text = re.sub(r"'s\b", '', text)  # Remove possessive 's
    text = re.sub(r"n't\b", '', text)  # Remove n't
    text = re.sub(r"'re\b", '', text)  # Remove 're
    text = re.sub(r"'ve\b", '', text)  # Remove 've
    text = re.sub(r"'ll\b", '', text)  # Remove 'll
    text = re.sub(r"'d\b", '', text)   # Remove 'd
    
    # Replace punctuation and special characters with spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Split into words and filter out empty strings
    words = [word.strip() for word in text.split() if word.strip()]
    
    return words

def generate_id(url):
    """
    Generate a unique ID for the article using SHA256 hash
    """
    return hashlib.sha256(url.encode('utf-8')).hexdigest()

def count_words(text):
    """
    Count words in text (excluding HTML tags)
    """
    if not text:
        return 0
    
    # Remove HTML tags
    clean_text = re.sub(r'<[^>]+>', '', text)
    
    # Count words
    words = clean_text.split()
    return len(words)

def get_guardian_articles_july():
    """
    Fetch all Guardian articles from July 2025
    """
    # Set date range for July 2025
    start_date_str = "2025-07-01"
    end_date_str = "2025-07-31"
    
    print(f"Fetching Guardian articles from {start_date_str} to {end_date_str}")
    
    # API configuration
    base_url = "https://content.guardianapis.com/search"
    
    # Parameters for the API request
    params = {
        'api-key': API_KEY,
        'from-date': start_date_str,
        'to-date': end_date_str,
        'page-size': 50,  # Maximum allowed
        'show-fields': 'headline,byline,trailText,body,wordcount,thumbnail,shortUrl,publication,lastModified',
        'show-tags': 'contributor,keyword,tone,type',
        'show-elements': 'image',
        'order-by': 'newest'
    }
    
    all_articles = []
    page = 1
    skipped_count = 0
    
    while True:
        params['page'] = page
        
        try:
            print(f"Fetching page {page}...")
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data['response']['status'] != 'ok':
                print(f"API returned error status: {data['response']['status']}")
                break
            
            # Get pagination info
            total_results = data['response']['total']
            total_pages = data['response']['pages']
            current_page = data['response']['currentPage']
            
            print(f"Page {current_page} of {total_pages} (Total articles: {total_results})")
            
            # Add articles from current page
            articles = data['response']['results']
            
            # Filter articles by word count (must have > 100 words)
            filtered_articles = []
            for article in articles:
                body_text = article.get('fields', {}).get('body', '')
                word_count = count_words(body_text)
                
                if word_count > 100:
                    filtered_articles.append(article)
                else:
                    skipped_count += 1
                    if skipped_count <= 5:  # Only show first few skipped articles
                        print(f"Skipping article '{article.get('webTitle', '')}' - only {word_count} words")
            
            all_articles.extend(filtered_articles)
            
            # Check if we've reached the last page
            if current_page >= total_pages:
                break
                
            page += 1
            
            # Add a small delay to be respectful to the API
            time.sleep(0.1)
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {page}: {e}")
            break
        except KeyError as e:
            print(f"Unexpected response format: {e}")
            break
    
    if skipped_count > 5:
        print(f"... and {skipped_count - 5} more articles skipped for having <= 100 words")
    
    return all_articles

def convert_to_required_format(article):
    """
    Convert Guardian API response to the required JSONL format
    """
    # Extract basic info
    url = article.get('webUrl', '')
    title = article.get('webTitle', '')
    api_url = article.get('apiUrl', '')
    
    # Extract fields
    fields = article.get('fields', {})
    body = fields.get('body', '')
    headline = fields.get('headline', title)
    byline = fields.get('byline', '')
    trail_text = fields.get('trailText', '')
    thumbnail = fields.get('thumbnail', '')
    short_url = fields.get('shortUrl', '')
    last_modified = fields.get('lastModified', '')
    
    # Extract authors from byline and tags
    authors = []
    if byline:
        # Clean up byline to extract author names
        byline_clean = re.sub(r'<[^>]+>', '', byline)  # Remove HTML
        authors.append(byline_clean.strip())
    
    # Add domain
    authors.append("www.theguardian.com")
    
    # Add contributor tags
    for tag in article.get('tags', []):
        if tag.get('type') == 'contributor':
            tag_id = tag.get('id', '')
            if '/' in tag_id:
                contributor_slug = tag_id.split('/')[-1]
                authors.append(contributor_slug)
    
    # Get publication date
    pub_date = article.get('webPublicationDate', '')
    if pub_date:
        # Convert to the required format
        try:
            dt = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
            formatted_date = dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            formatted_date = pub_date
    else:
        formatted_date = ''
    
    # Extract image URL from elements
    image_url = ''
    elements = article.get('elements', [])
    for element in elements:
        if element.get('type') == 'image' and 'assets' in element:
            assets = element['assets']
            if assets:
                # Get the largest image
                largest_asset = max(assets, key=lambda x: x.get('typeData', {}).get('width', 0))
                image_url = largest_asset.get('file', '')
                break
    
    # If no image from elements, try thumbnail
    if not image_url:
        image_url = thumbnail
    
    # Create filename from URL
    filename = quote(url, safe='') + '.json'
    
    # Generate article ID
    article_id = generate_id(url)
    
    # Create the main text from headline + trail text + body
    main_parts = []
    if headline and headline != title:
        main_parts.append(headline)
    if trail_text:
        main_parts.append(trail_text)
    if body:
        main_parts.append(body)
    
    maintext = '\n'.join(main_parts) if main_parts else body
    
    # Tokenize the news (title + maintext)
    tokenization_text = title + ' ' + maintext
    tokenized_news = tokenize_text(tokenization_text)
    
    # Create the final object
    result = {
        "authors": authors,
        "date_download": datetime.now().strftime('%Y-%m-%d %H:%M:%S+00:00'),
        "date_modify": last_modified if last_modified else None,
        "date_publish": formatted_date,
        "description": trail_text,
        "filename": filename,
        "image_url": image_url,
        "language": "en",
        "localpath": None,
        "maintext": maintext,
        "source_domain": "www.theguardian.com",
        "title": title,
        "title_page": None,
        "title_rss": None,
        "url": url,
        "id": article_id,
        "tokenized_news": tokenized_news
    }
    
    return result

def save_articles_jsonl(articles, output_dir):
    """
    Save articles in JSONL format to the specified directory
    """
    # Create output directory structure
    theguardian_dir = os.path.join(output_dir, 'theguardian')
    os.makedirs(theguardian_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"www.theguardian.com_july2025_{timestamp}.jsonl"
    filepath = os.path.join(theguardian_dir, filename)
    
    # Convert and save articles
    converted_articles = []
    for article in articles:
        converted = convert_to_required_format(article)
        converted_articles.append(converted)
    
    # Write to JSONL file
    with open(filepath, 'w', encoding='utf-8') as f:
        for article in converted_articles:
            json.dump(article, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"\nSaved {len(converted_articles)} articles to {filepath}")
    
    # Print summary statistics
    print("\n=== SUMMARY ===")
    print(f"Total articles saved: {len(converted_articles)}")
    
    # Count by section
    sections = {}
    for article in articles:
        section = article.get('sectionName', 'Unknown')
        sections[section] = sections.get(section, 0) + 1
    
    print("\nArticles by section (Top 15):")
    for section, count in sorted(sections.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {section}: {count}")
    
    # Word count statistics
    word_counts = []
    for article in converted_articles:
        word_count = count_words(article['maintext'])
        word_counts.append(word_count)
    
    if word_counts:
        avg_words = sum(word_counts) / len(word_counts)
        print(f"\nWord count statistics:")
        print(f"  Average words per article: {avg_words:.0f}")
        print(f"  Minimum words: {min(word_counts)}")
        print(f"  Maximum words: {max(word_counts)}")
    
    return filepath

def main():
    """
    Main function to fetch and process Guardian articles for July 2025
    """
    print("Guardian Article Fetcher - July 2025")
    print("=" * 40)
    
    # Output directory
    output_dir = "/fast/nchandak/forecasting/newsdata"
    
    try:
        # Fetch articles
        print("Fetching articles...")
        articles = get_guardian_articles_july()
        
        if not articles:
            print("No articles found matching criteria.")
            return
        
        # Save in required format
        print("Converting and saving articles...")
        filepath = save_articles_jsonl(articles, output_dir)
        
        print(f"\nProcessing complete! File saved to: {filepath}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 