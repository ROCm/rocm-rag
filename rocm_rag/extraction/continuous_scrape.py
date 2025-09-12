# This script is used to continuously scrape and index pages from AMD ROCm Blogs.
# in BFS, parallel crawling is used to process all elements in the queue before moving to rest of the code
# this is an optimization on sequential BFS
# all internal links are crawled to make sure coverage, but only blog pages are indexed
import os
from urllib.parse import urlparse
import sqlite3
import hashlib
from datetime import datetime
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
import regex
from enum import Enum
import asyncio
from rocm_rag import config
from rocm_rag.utils.wait_for_service import wait_for_port


conn: sqlite3.Connection | None = None
cursor: sqlite3.Cursor | None = None

class UpdateStatus(Enum):
    NEW = "NEW"
    UPDATED = "UPDATED"
    SKIPPED = "SKIPPED"

def hash_page_content(content: str) -> str:
    """Hash page content using SHA256."""
    normalized = content.strip().lower()
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

def extract_domain(url: str) -> str:
    """Extract site domain from URL."""
    return urlparse(url).netloc

def get_page_hash(url: str, domain: str) -> str | None:
    """Return stored hash for given URL if exists."""
    domain = extract_domain(url)
    cursor.execute("SELECT page_hash FROM page_hashes WHERE domain = ? AND url = ?", (domain, url))
    result = cursor.fetchone()
    return result[0] if result else None

def save_or_update_page_hash(url: str, domain: str, content: str) -> UpdateStatus:
    """Insert or update page hash for a URL."""
    new_hash = hash_page_content(content)

    # if url not exists in the database, insert hash
    cursor.execute("SELECT page_hash FROM page_hashes WHERE domain = ? AND url = ?", (domain, url))
    result = cursor.fetchone()

    if result is None:
        # If no existing hash, insert new record
        cursor.execute("""
            INSERT INTO page_hashes (domain, url, page_hash, last_updated)
            VALUES (?, ?, ?, ?)
        """, (domain, url, new_hash, datetime.now().isoformat()))
        conn.commit()
        print(f"[NEW] Hash saved for {url}")
        return UpdateStatus.NEW
    
    # If existing hash, compare with new hash
    existing_hash = get_page_hash(url, domain)

    if existing_hash == new_hash:
        print(f"[SKIPPED] No change for {url}")
        return UpdateStatus.SKIPPED
    else:
        cursor.execute("""
            INSERT INTO page_hashes (domain, url, page_hash, last_updated)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(domain, url) DO UPDATE SET
                page_hash = excluded.page_hash,
                last_updated = excluded.last_updated
        """, (domain, url, new_hash, datetime.now().isoformat()))
        conn.commit()
        print(f"[UPDATED] Hash saved for {url}")
        return UpdateStatus.UPDATED


def is_valid_page(url: str) -> bool:
    if any(regex.search(pattern, url) for pattern in config.ROCM_RAG_VALID_PAGE_FILTERS):
        return True
    return False

def is_valid_url(url: str, domain: str) -> bool:
    parsed = urlparse(url)
    exet = os.path.splitext(parsed.path.lower())[1]
    if (parsed.netloc == domain) and (not parsed.fragment) and (exet in config.ROCM_RAG_VALID_EXTENSIONS):
        return True
    return False


def required_human_verification(result) -> bool:
    if result.success:
        if any(regex.search(pattern, str(result.markdown)) for pattern in config.ROCM_RAG_REQUIRE_HUMAN_VERIFICATION_FILTERS):
            return True
    return False

def is_page_not_found(result) -> bool:
    if result.success:
        if any(regex.search(pattern, str(result.markdown)) for pattern in config.ROCM_RAG_PAGE_NOT_FOUND_FILTERS):
            return True
    return False

async def get_all_urls(start_url):
    visited = set()
    to_visit = [start_url]
    domain = urlparse(start_url).netloc
    urls = []

    while to_visit and ((not config.ROCM_RAG_SET_MAX_NUM_PAGES) or (config.ROCM_RAG_SET_MAX_NUM_PAGES and len(urls) < config.ROCM_RAG_MAX_NUM_PAGES)):
        results = await crawler.arun_many(urls=to_visit, config=run_conf)
        visited.update(to_visit)  
        to_visit = []  # all URLs in to_visit will be processed in one iteration

        for result in results:
            if is_page_not_found(result):
                print(f"[ERROR] {result.url} not found (404)")
                continue
            if required_human_verification(result):
                print(f"[HUMAN VERIFICATION REQUIRED] {result.url}")
                # visited.discard(result.url)  
                if (result.url not in to_visit):
                    to_visit.append(result.url)
            else:
                url = result.url
                if config.ROCM_RAG_SET_MAX_NUM_PAGES and (len(urls) >= config.ROCM_RAG_MAX_NUM_PAGES):
                    print(f"Reached max number of pages to index: {config.ROCM_RAG_MAX_NUM_PAGES}")
                    break
                if is_valid_page(url):
                    urls.append(url)
                    f.write(url + "\n") 
                    f.flush()
                    hash_update_status = save_or_update_page_hash(url, domain, str(result.markdown))

                    if hash_update_status == UpdateStatus.NEW:
                        # If it's a new page, index it and save to vector db
                        indexing_pipeline.insert_page(url, domain, str(result.markdown))
                    elif hash_update_status == UpdateStatus.UPDATED:
                        # delete the old indexed document from vector db if it exists
                        indexing_pipeline.delete_by_url(url, domain)
                        indexing_pipeline.insert_page(url, domain, str(result.markdown))
                    else:
                        print(f"Skipping indexing for {url} as it has not changed.")
                
                internal_links = [x['href'] for x in result.links["internal"]]
                for link in internal_links:
                    if is_valid_url(link, domain) and (link not in visited): 
                        to_visit.append(link)
                        visited.add(link) 
                        print(f"adding: {link}")    
                    
    return urls


if __name__ == "__main__":
    start_urls = config.ROCM_RAG_START_URLS
    # database to store page hashes
    os.makedirs(os.path.dirname(config.ROCM_RAG_HASH_DIR), exist_ok=True)
    conn = sqlite3.connect(config.ROCM_RAG_HASH_DIR)
    cursor = conn.cursor()
    # Create a shared table with `domain` and `url` as composite primary key
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS page_hashes (
        domain TEXT NOT NULL,
        url TEXT NOT NULL,
        page_hash TEXT NOT NULL,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (domain, url)
    )
    """)
    conn.commit()

    # scraper config
    browser_conf = BrowserConfig(headless=True)  # or False to see the browser
    # scrape single page at a time
    run_conf = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        only_text=True,
        excluded_tags=["script", "style", "head", "title", "meta", "Button", "button", "input", "form", "nav", "footer", "aside"], # Exclude non-content html tags
    )
    crawler = AsyncWebCrawler(config=browser_conf)

    # indexing pipeline
    if config.ROCM_RAG_EXTRACTION_FRAMEWORK == "langgraph":
        from rocm_rag.extraction.langgraph_extraction.indexing_graph import IndexingGraph
        indexing_pipeline = IndexingGraph()
    elif config.ROCM_RAG_EXTRACTION_FRAMEWORK == "haystack":
        from rocm_rag.extraction.haystack_extraction.indexing_pipeline import IndexingPipeline
        indexing_pipeline = IndexingPipeline()
    else:
        raise ValueError("ROCM_RAG_EXTRACTION_FRAMEWORK must be either 'haystack' or 'langgraph'")


    f = open(config.ROCM_RAG_VISITED_URL_FILE, "w")

    wait_for_port("localhost", config.ROCM_RAG_WEAVIATE_PORT, config.ROCM_RAG_WAIT_VECTOR_DB_TIMEOUT)
    wait_for_port("localhost", config.ROCM_RAG_EMBEDDER_API_PORT, config.ROCM_RAG_WAIT_EMBEDDER_TIMEOUT)

    print("Starting continuous scrape and index")
    for start_url in start_urls:
        urls = asyncio.run(get_all_urls(start_url))
        print(f"All URLs found: {len(urls)}")
    f.close()


    
