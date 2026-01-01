# Guardian Articles - July 2025 Data Collection

## Summary

Successfully collected **5,930 Guardian articles** from July 2025 (July 1-31, 2025) using The Guardian's Open Platform API.

## Output File

**Location**: `/fast/nchandak/forecasting/newsdata/theguardian/www.theguardian.com_july2025_20250731_181157.jsonl`

**File Size**: 126 MB

**Format**: JSONL (one JSON object per line)

## Data Collection Details

- **Date Range**: July 1-31, 2025
- **Total Articles**: 5,930 articles 
- **Filtered**: Articles with ≤100 words were excluded (46 articles skipped)
- **API Pages Processed**: 120 pages
- **Processing Time**: ~18 minutes

## Data Format

Each line contains a JSON object with these fields (matching the reference format):

```json
{
  "authors": ["Author Name", "www.theguardian.com", "author-slug"],
  "date_download": "2025-07-31 18:12:00+00:00",
  "date_modify": "2025-07-01T00:30:27Z", 
  "date_publish": "2025-07-01 00:08:56",
  "description": "Article description/trail text",
  "filename": "url_encoded_filename.json",
  "image_url": "https://...",
  "language": "en",
  "localpath": null,
  "maintext": "Full article text with headline and body",
  "source_domain": "www.theguardian.com",
  "title": "Article headline",
  "title_page": null,
  "title_rss": null,
  "url": "https://www.theguardian.com/...",
  "id": "sha256_hash_of_url",
  "tokenized_news": ["tokenized", "words", "array"]
}
```

## Content Statistics

### Articles by Section (Top 15)
- US news: 659 articles
- Sport: 647 articles  
- World news: 568 articles
- Football: 517 articles
- Opinion: 386 articles
- Australia news: 358 articles
- Business: 289 articles
- UK news: 242 articles
- Music: 212 articles
- Environment: 209 articles
- Politics: 203 articles
- Society: 173 articles
- Life and style: 153 articles
- Film: 152 articles
- Television & radio: 152 articles

### Word Count Statistics
- **Average words per article**: 1,160 words
- **Minimum words**: 119 words
- **Maximum words**: 16,277 words

## Scripts Used

1. **`guardian_fetcher_july.py`** - Main collection script
2. **`apikey.py`** - API key configuration
3. **`requirements.txt`** - Dependencies (requests>=2.31.0)

## Features Implemented

- ✅ Automatic date range (July 2025)
- ✅ Word count filtering (>100 words)
- ✅ Complete pagination handling
- ✅ Rich metadata extraction
- ✅ Proper tokenization
- ✅ Error handling and rate limiting
- ✅ JSONL format matching reference specification
- ✅ Unique ID generation using SHA256
- ✅ Author extraction from bylines and contributor tags
- ✅ Image URL extraction from elements and thumbnails

## Data Quality

- All articles have >100 words of content
- Full HTML body text preserved in `maintext` field
- Proper tokenization following reference format
- Complete metadata including authors, publication dates, images
- Unique article IDs for deduplication 