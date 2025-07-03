import logging
import requests
import uuid
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup # For extracting text from HTML if needed for generic website scrape
from datetime import datetime, timezone
import json # for stringifying fallback content

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    'User-Agent': 'MarketIntelligenceAgent/1.0 (SyncHandler; +http://example.com/botinfo)'
}

MAX_ARTICLES_PER_SOURCE_SYNC = 10 # Limit for how many articles to pull in one sync

class SyncedArticle:
    def __init__(self, title: str, url: str, content: str, published_at: Optional[str] = None, source_name: Optional[str] = None):
        self.title = title.strip() if title else "Untitled Document"
        self.url = url
        self.content = content # This should be the main text content
        self.published_at = published_at
        self.source_name = source_name or urlparse(url).netloc # Default source name to domain
        self.original_filename = self.generate_filename()

    def generate_filename(self) -> str:
        # Generate a somewhat readable filename
        name_part = "".join(c if c.isalnum() else "_" for c in self.title.lower().replace(" ", "_"))[:50]
        return f"{name_part}_{uuid.uuid4().hex[:8]}.txt" # Ensure some uniqueness and .txt extension

    def to_document_db_record(self, user_id: str, source_id: str, status: str = "pending_processing") -> Dict[str, Any]:
        # Prepare the record for the 'documents' table
        doc_record = {
            "uploader_id": user_id, # This should be the user_id associated with the data_source
            "source_id": source_id,
            "original_filename": self.original_filename,
            "filename": self.original_filename, # If not saving to disk, this can be same as original_filename
            "file_type": "text/plain", # Content is treated as plain text
            "file_extension": ".txt",
            "status": status, # Initial status
            "upload_time": datetime.now(timezone.utc).isoformat(), # Sync time
            "text": self.content, # Full extracted/fetched content
            "word_count": len(self.content.split()) if self.content else 0,
            # Storing sync metadata in 'analysis' field as it's JSONB.
            # A dedicated 'sync_metadata' field might be cleaner in future schema iterations.
            "analysis": {
                "sync_metadata": {
                    "synced_url": self.url,
                    "synced_title": self.title,
                    "published_at": self.published_at,
                    "retrieved_from_source_name": self.source_name
                }
            }
        }
        # file_size could be len(self.content.encode('utf-8')) if needed
        return doc_record


def fetch_generic_url_content(endpoint_url: str, config: Dict[str, Any]) -> List[SyncedArticle]:
    """Fetches content from a generic URL (API, RSS feed, website page)."""
    articles: List[SyncedArticle] = []
    custom_headers = config.get("headers", {})
    headers = {**DEFAULT_HEADERS, **custom_headers}
    api_key = config.get("apiKey")
    auth_type = config.get("auth_type", "none").lower()

    if api_key:
        if auth_type == "bearer":
            headers["Authorization"] = f"Bearer {api_key}"
        elif auth_type == "header":
            header_name = config.get("header_name", "X-API-Key")
            headers[header_name] = api_key

    try:
        logger.info(f"SyncHandler: Fetching generic URL: {endpoint_url}")
        response = requests.get(endpoint_url, headers=headers, timeout=15)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()

        if "application/json" in content_type:
            data = response.json()
            items_list = data.get("articles", data.get("items", data.get("results", data.get("data", data if isinstance(data, list) else []))))

            if not isinstance(items_list, list):
                items_list = [items_list]

            for item in items_list[:MAX_ARTICLES_PER_SOURCE_SYNC]:
                if not isinstance(item, dict): continue
                title = item.get("title", item.get("name"))
                url = item.get("url", item.get("link", endpoint_url))
                content = item.get("content", item.get("description", item.get("summary", item.get("body", ""))))
                if not content and isinstance(item, dict): content = json.dumps(item, indent=2)

                if title and content:
                    articles.append(SyncedArticle(title=str(title), url=str(url), content=str(content), published_at=item.get("publishedAt", item.get("pubDate", item.get("published_date")))))

        elif "application/rss+xml" in content_type or "application/xml" in content_type or "text/xml" in content_type:
            soup = BeautifulSoup(response.content, features="xml")
            entries = soup.find_all("item") or soup.find_all("entry")
            for entry in entries[:MAX_ARTICLES_PER_SOURCE_SYNC]:
                title = entry.title.string if entry.title else "Untitled Feed Item"
                link_tag = entry.link
                link = ""
                if link_tag:
                    if link_tag.string: link = link_tag.string.strip()
                    elif link_tag.has_attr('href'): link = link_tag['href'].strip()
                if not link: link = endpoint_url

                description_tag = entry.description or entry.summary
                description = description_tag.string if description_tag else ""

                content_encoded_tag = entry.find("content:encoded")
                full_content = content_encoded_tag.string if content_encoded_tag else description

                soup_content = BeautifulSoup(full_content or "", "html.parser")
                plain_content = soup_content.get_text(separator="\n", strip=True)

                pub_date_tag = entry.pubDate or entry.published or entry.updated
                published_at = pub_date_tag.string if pub_date_tag else None

                if title and plain_content:
                    articles.append(SyncedArticle(title=title, url=link, content=plain_content, published_at=published_at))

        elif "text/html" in content_type:
            soup = BeautifulSoup(response.content, "html.parser")
            title = soup.title.string if soup.title else urlparse(endpoint_url).path.split("/")[-1] or "Scraped Page"

            text_parts = []
            for selector in ['article', 'main', 'div[role="main"]', 'body']:
                main_content_tags = soup.select(selector)
                if main_content_tags:
                    for tag in main_content_tags:
                        text_parts.append(tag.get_text(separator="\n", strip=True))
                    if text_parts and any(p.strip() for p in text_parts): break # Stop if content found

            content = "\n\n".join(p for p in text_parts if p.strip())
            if title and content:
                 articles.append(SyncedArticle(title=title, url=endpoint_url, content=content))

        else:
            content = response.text
            title = urlparse(endpoint_url).path.split("/")[-1] or "Synced Text Document"
            if content:
                articles.append(SyncedArticle(title=title, url=endpoint_url, content=content))

        logger.info(f"SyncHandler: Fetched {len(articles)} items from generic URL {endpoint_url}")
        return articles

    except requests.exceptions.RequestException as e:
        logger.error(f"SyncHandler: Error fetching generic URL {endpoint_url}: {e}")
    except Exception as e:
        logger.error(f"SyncHandler: Unexpected error processing generic URL {endpoint_url}: {e}")
    return []


def fetch_newsapi_articles(config: Dict[str, Any], source_name_for_query: str) -> List[SyncedArticle]:
    api_key = config.get("apiKey")
    if not api_key:
        logger.warning("NewsAPI sync: Missing API Key.")
        return []

    articles: List[SyncedArticle] = []
    query = config.get("default_query", source_name_for_query or "latest technology news")

    try:
        logger.info(f"SyncHandler: Fetching NewsAPI for query: {query}")
        response = requests.get(
            config.get("endpoint", "https://newsapi.org/v2/everything"),
            params={"q": query, "pageSize": MAX_ARTICLES_PER_SOURCE_SYNC, "apiKey": api_key, "language": "en", "sortBy": "publishedAt"},
            timeout=15,
            headers=DEFAULT_HEADERS
        )
        response.raise_for_status()
        data = response.json()
        for item in data.get("articles", []):
            title = item.get("title")
            url = item.get("url")
            content = item.get("content", item.get("description", ""))
            if content and "[+ chars]" in content: # Basic check for truncation
                content = item.get("description", "") # Fallback to description if content is truncated

            if title and url and content:
                articles.append(SyncedArticle(title=title, url=url, content=content, published_at=item.get("publishedAt"), source_name=item.get("source", {}).get("name")))
        logger.info(f"SyncHandler: Fetched {len(articles)} articles from NewsAPI for query '{query}'.")
        return articles
    except Exception as e:
        logger.error(f"SyncHandler: Error fetching from NewsAPI: {e}")
    return []


def fetch_tavily_search_results(config: Dict[str, Any], source_name_for_query: str) -> List[SyncedArticle]:
    api_key = config.get("apiKey")
    if not api_key:
        logger.warning("Tavily sync: Missing API Key.")
        return []

    articles: List[SyncedArticle] = []
    query = config.get("default_query", f"latest updates on {source_name_for_query}" if source_name_for_query else "overview of recent market developments")

    try:
        logger.info(f"SyncHandler: Fetching Tavily for query: {query}")
        response = requests.post(
            config.get("endpoint", "https://api.tavily.com/search"),
            json={"api_key": api_key, "query": query, "search_depth": "basic", "max_results": MAX_ARTICLES_PER_SOURCE_SYNC, "include_raw_content": True},
            timeout=20,
            headers=DEFAULT_HEADERS
        )
        response.raise_for_status()
        data = response.json()
        for item in data.get("results", []):
            title = item.get("title")
            url = item.get("url")
            content = item.get("raw_content", item.get("content", ""))
            if title and url and content:
                 articles.append(SyncedArticle(title=title, url=url, content=content, published_at=item.get("publish_date"), source_name=item.get("source", urlparse(url).netloc)))
        logger.info(f"SyncHandler: Fetched {len(articles)} results from Tavily for query '{query}'.")
        return articles
    except Exception as e:
        logger.error(f"SyncHandler: Error fetching from Tavily: {e}")
    return []


# Dispatcher function for syncing
def sync_data_source(source: Dict[str, Any]) -> List[SyncedArticle]:
    source_type = source.get("type", "").lower()
    config = source.get("config", {})
    source_name = source.get("name", "Unknown Source")
    endpoint = config.get("endpoint", "").lower()

    logger.info(f"SyncHandler: Attempting sync for source '{source_name}' (Type: {source_type}, Endpoint: {endpoint[:50]}...)")

    # Determine handler based on type and endpoint characteristics
    if source_type == "newsapi" or "newsapi.org" in endpoint:
        return fetch_newsapi_articles(config, source_name)
    elif source_type == "tavily" or "tavily.com" in endpoint:
        return fetch_tavily_search_results(config, source_name)
    # Add other specific handlers like:
    # elif source_type == "mediastack" or "api.mediastack.com" in endpoint:
    #     return fetch_mediastack_articles(config, source_name)
    elif source_type in ["api", "rss", "website_scrape", "custom_api", "generic_api"] and config.get("endpoint"): # Ensure endpoint exists for generic
        return fetch_generic_url_content(config.get("endpoint"), config)
    else:
        logger.warning(f"SyncHandler: No specific sync handler for source type '{source_type}' or endpoint not configured properly for source '{source_name}'.")
        return []
