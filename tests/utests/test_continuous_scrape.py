# test_continuous_scraper.py
import sqlite3
import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock, mock_open

from rocm_rag.extraction import continuous_scrape

@pytest.fixture
def temp_db(monkeypatch):
    """Provide an in-memory sqlite3 db for tests."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE page_hashes (
            domain TEXT NOT NULL,
            url TEXT NOT NULL,
            page_hash TEXT NOT NULL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (domain, url)
        )
    """)
    conn.commit()

    monkeypatch.setattr(continuous_scrape, "conn", conn)
    monkeypatch.setattr(continuous_scrape, "cursor", cursor)
    yield conn
    conn.close()


def test_hash_page_content_consistent():
    h1 = continuous_scrape.hash_page_content("Hello World")
    h2 = continuous_scrape.hash_page_content("  hello world  ")
    assert h1 == h2


def test_save_or_update_page_hash_new(temp_db):
    url = "https://rocm.example.com/blog/post1"
    domain = "rocm.example.com"
    status = continuous_scrape.save_or_update_page_hash(url, domain, "content")
    assert status == continuous_scrape.UpdateStatus.NEW


def test_save_or_update_page_hash_update(temp_db):
    url = "https://rocm.example.com/blog/post2"
    domain = "rocm.example.com"
    # insert initial
    continuous_scrape.save_or_update_page_hash(url, domain, "old content")
    # update with new
    status = continuous_scrape.save_or_update_page_hash(url, domain, "new content")
    assert status == continuous_scrape.UpdateStatus.UPDATED


def test_save_or_update_page_hash_skipped(temp_db):
    url = "https://rocm.example.com/blog/post3"
    domain = "rocm.example.com"
    continuous_scrape.save_or_update_page_hash(url, domain, "same content")
    status = continuous_scrape.save_or_update_page_hash(url, domain, "same content")
    assert status == continuous_scrape.UpdateStatus.SKIPPED


def test_is_valid_url(monkeypatch):
    from urllib.parse import urlparse
    monkeypatch.setattr(continuous_scrape.config, "ROCM_RAG_VALID_EXTENSIONS", [".html", ".htm"])
    url = "https://rocm.example.com/post.html"
    assert continuous_scrape.is_valid_url(url, "rocm.example.com")
    assert not continuous_scrape.is_valid_url("https://other.com/page.html", "rocm.example.com")


def test_extract_domain():
    """Test extract_domain function."""
    assert continuous_scrape.extract_domain("https://rocm.blogs.amd.com/path/to/page") == "rocm.blogs.amd.com"
    assert continuous_scrape.extract_domain("http://example.com") == "example.com"
    assert continuous_scrape.extract_domain("https://subdomain.example.com:8080/path") == "subdomain.example.com:8080"
    assert continuous_scrape.extract_domain("ftp://files.example.org/files") == "files.example.org"


def test_get_page_hash_exists(temp_db):
    """Test get_page_hash when hash exists."""
    url = "https://rocm.example.com/test"
    domain = "rocm.example.com"
    test_hash = "test_hash_123"
    
    # Insert a hash into the database
    continuous_scrape.cursor.execute(
        "INSERT INTO page_hashes (domain, url, page_hash) VALUES (?, ?, ?)",
        (domain, url, test_hash)
    )
    continuous_scrape.conn.commit()
    
    result = continuous_scrape.get_page_hash(url, domain)
    assert result == test_hash


def test_get_page_hash_not_exists(temp_db):
    """Test get_page_hash when hash doesn't exist."""
    url = "https://rocm.example.com/nonexistent"
    domain = "rocm.example.com"
    
    result = continuous_scrape.get_page_hash(url, domain)
    assert result is None


def test_is_valid_page_true(monkeypatch):
    """Test is_valid_page returns True for valid pages."""
    monkeypatch.setattr(continuous_scrape.config, "ROCM_RAG_VALID_PAGE_FILTERS", [r"/blog/", r"/README\.html"])
    
    assert continuous_scrape.is_valid_page("https://rocm.blogs.amd.com/blog/post1")
    assert continuous_scrape.is_valid_page("https://example.com/README.html")


def test_is_valid_page_false(monkeypatch):
    """Test is_valid_page returns False for invalid pages."""
    monkeypatch.setattr(continuous_scrape.config, "ROCM_RAG_VALID_PAGE_FILTERS", [r"/blog/", r"/README\.html"])
    
    assert not continuous_scrape.is_valid_page("https://rocm.blogs.amd.com/contact")
    assert not continuous_scrape.is_valid_page("https://example.com/about.html")


def test_required_human_verification_true():
    """Test required_human_verification returns True when verification required."""
    # Mock result object with success=True and markdown containing verification text
    mock_result = MagicMock()
    mock_result.success = True
    mock_result.markdown = "Please complete verification. Verifying you are human. Continue to site."
    
    with patch.object(continuous_scrape.config, "ROCM_RAG_REQUIRE_HUMAN_VERIFICATION_FILTERS", [r"Verifying you are human"]):
        assert continuous_scrape.required_human_verification(mock_result)


def test_required_human_verification_false():
    """Test required_human_verification returns False when no verification needed."""
    # Mock result object with success=True but no verification text
    mock_result = MagicMock()
    mock_result.success = True
    mock_result.markdown = "This is regular page content without verification."
    
    with patch.object(continuous_scrape.config, "ROCM_RAG_REQUIRE_HUMAN_VERIFICATION_FILTERS", [r"Verifying you are human"]):
        assert not continuous_scrape.required_human_verification(mock_result)


def test_required_human_verification_unsuccessful_result():
    """Test required_human_verification returns False when result is not successful."""
    mock_result = MagicMock()
    mock_result.success = False
    mock_result.markdown = "Verifying you are human"
    
    with patch.object(continuous_scrape.config, "ROCM_RAG_REQUIRE_HUMAN_VERIFICATION_FILTERS", [r"Verifying you are human"]):
        assert not continuous_scrape.required_human_verification(mock_result)


def test_is_page_not_found_true():
    """Test is_page_not_found returns True for 404 pages."""
    mock_result = MagicMock()
    mock_result.success = True
    mock_result.markdown = "Error: 404 - Page Not Found. The requested page could not be found."
    
    with patch.object(continuous_scrape.config, "ROCM_RAG_PAGE_NOT_FOUND_FILTERS", [r"404 - Page Not Found"]):
        assert continuous_scrape.is_page_not_found(mock_result)


def test_is_page_not_found_false():
    """Test is_page_not_found returns False for normal pages."""
    mock_result = MagicMock()
    mock_result.success = True
    mock_result.markdown = "This is a normal page with content."
    
    with patch.object(continuous_scrape.config, "ROCM_RAG_PAGE_NOT_FOUND_FILTERS", [r"404 - Page Not Found"]):
        assert not continuous_scrape.is_page_not_found(mock_result)


def test_is_page_not_found_unsuccessful_result():
    """Test is_page_not_found returns False when result is not successful."""
    mock_result = MagicMock()
    mock_result.success = False
    mock_result.markdown = "404 - Page Not Found"
    
    with patch.object(continuous_scrape.config, "ROCM_RAG_PAGE_NOT_FOUND_FILTERS", [r"404 - Page Not Found"]):
        assert not continuous_scrape.is_page_not_found(mock_result)


def test_is_valid_url_with_fragment(monkeypatch):
    """Test is_valid_url returns False for URLs with fragments."""
    monkeypatch.setattr(continuous_scrape.config, "ROCM_RAG_VALID_EXTENSIONS", [".html", ".htm"])
    url_with_fragment = "https://rocm.example.com/post.html#section1"
    assert not continuous_scrape.is_valid_url(url_with_fragment, "rocm.example.com")


def test_is_valid_url_wrong_extension(monkeypatch):
    """Test is_valid_url returns False for wrong extensions."""
    monkeypatch.setattr(continuous_scrape.config, "ROCM_RAG_VALID_EXTENSIONS", [".html", ".htm"])
    url_with_wrong_ext = "https://rocm.example.com/file.pdf"
    assert not continuous_scrape.is_valid_url(url_with_wrong_ext, "rocm.example.com")


def test_hash_page_content_empty():
    """Test hash_page_content with empty string."""
    h1 = continuous_scrape.hash_page_content("")
    h2 = continuous_scrape.hash_page_content("   ")
    assert h1 == h2


@pytest.mark.asyncio
async def test_get_all_urls_basic(monkeypatch, temp_db):
    """Test get_all_urls with basic functionality."""
    start_url = "https://rocm.blogs.amd.com"
    
    # Mock config values
    monkeypatch.setattr(continuous_scrape.config, "ROCM_RAG_SET_MAX_NUM_PAGES", False)
    monkeypatch.setattr(continuous_scrape.config, "ROCM_RAG_VALID_PAGE_FILTERS", [r"/blog/"])
    monkeypatch.setattr(continuous_scrape.config, "ROCM_RAG_VALID_EXTENSIONS", [".html"])
    monkeypatch.setattr(continuous_scrape.config, "ROCM_RAG_PAGE_NOT_FOUND_FILTERS", [r"404"])
    monkeypatch.setattr(continuous_scrape.config, "ROCM_RAG_REQUIRE_HUMAN_VERIFICATION_FILTERS", [r"Verifying"])
    
    # Mock crawler and results
    mock_crawler = AsyncMock()
    mock_result = MagicMock()
    mock_result.url = "https://rocm.blogs.amd.com/blog/post1.html"
    mock_result.success = True
    mock_result.markdown = "Some blog content"
    mock_result.links = {"internal": [{"href": "https://rocm.blogs.amd.com/blog/post2.html"}]}
    
    mock_crawler.arun_many.return_value = [mock_result]
    
    # Mock indexing pipeline
    mock_indexing_pipeline = MagicMock()
    
    # Set global variables at module level for this test
    continuous_scrape.crawler = mock_crawler
    continuous_scrape.run_conf = MagicMock()
    continuous_scrape.indexing_pipeline = mock_indexing_pipeline
    continuous_scrape.f = mock_open()()
    
    with patch('builtins.print'):  # Suppress print statements
        urls = await continuous_scrape.get_all_urls(start_url)
        
        # Should find the valid URL (may be found twice due to the algorithm processing)
        assert len(urls) >= 1
        assert "https://rocm.blogs.amd.com/blog/post1.html" in urls


@pytest.mark.asyncio 
async def test_get_all_urls_with_max_pages(monkeypatch, temp_db):
    """Test get_all_urls with max pages limit."""
    start_url = "https://rocm.blogs.amd.com" # ROCm Blogs homepage
    
    # Mock config values with max pages limit - need to set both variables
    monkeypatch.setattr(continuous_scrape.config, "ROCM_RAG_SET_MAX_NUM_PAGES", True)
    # Create the ROCM_RAG_MAX_NUM_PAGES attribute since it's conditionally defined
    monkeypatch.setattr(continuous_scrape.config, "ROCM_RAG_MAX_NUM_PAGES", 1, raising=False)
    monkeypatch.setattr(continuous_scrape.config, "ROCM_RAG_VALID_PAGE_FILTERS", [r"/blog/"])
    monkeypatch.setattr(continuous_scrape.config, "ROCM_RAG_VALID_EXTENSIONS", [".html"])
    monkeypatch.setattr(continuous_scrape.config, "ROCM_RAG_PAGE_NOT_FOUND_FILTERS", [r"404"])
    monkeypatch.setattr(continuous_scrape.config, "ROCM_RAG_REQUIRE_HUMAN_VERIFICATION_FILTERS", [r"Verifying"])
    
    # Mock crawler and results
    mock_crawler = AsyncMock()
    mock_result1 = MagicMock()
    mock_result1.url = "https://rocm.blogs.amd.com/blog/post1.html"
    mock_result1.success = True
    mock_result1.markdown = "Blog content 1"
    mock_result1.links = {"internal": []}
    
    mock_crawler.arun_many.return_value = [mock_result1]
    
    # Mock indexing pipeline
    mock_indexing_pipeline = MagicMock()
    
    # Set global variables at module level for this test
    continuous_scrape.crawler = mock_crawler
    continuous_scrape.run_conf = MagicMock()
    continuous_scrape.indexing_pipeline = mock_indexing_pipeline
    continuous_scrape.f = mock_open()()
    
    with patch('builtins.print'):  # Suppress print statements
        urls = await continuous_scrape.get_all_urls(start_url)
        
        # Should respect the max pages limit
        assert len(urls) <= 1


@pytest.mark.asyncio
async def test_get_all_urls_page_not_found(monkeypatch, temp_db):
    """Test get_all_urls handling 404 pages."""
    start_url = "https://rocm.blogs.amd.com"
    
    # Mock config values
    monkeypatch.setattr(continuous_scrape.config, "ROCM_RAG_SET_MAX_NUM_PAGES", False)
    monkeypatch.setattr(continuous_scrape.config, "ROCM_RAG_VALID_PAGE_FILTERS", [r"/blog/"])
    monkeypatch.setattr(continuous_scrape.config, "ROCM_RAG_VALID_EXTENSIONS", [".html"])
    monkeypatch.setattr(continuous_scrape.config, "ROCM_RAG_PAGE_NOT_FOUND_FILTERS", [r"404"])
    monkeypatch.setattr(continuous_scrape.config, "ROCM_RAG_REQUIRE_HUMAN_VERIFICATION_FILTERS", [r"Verifying"])
    
    # Mock crawler and results - 404 page
    mock_crawler = AsyncMock()
    mock_result = MagicMock()
    mock_result.url = "https://rocm.blogs.amd.com/blog/nonexistent.html"
    mock_result.success = True
    mock_result.markdown = "404 - Page Not Found"
    mock_result.links = {"internal": []}
    
    mock_crawler.arun_many.return_value = [mock_result]
    
    # Mock indexing pipeline
    mock_indexing_pipeline = MagicMock()
    
    # Set global variables at module level for this test
    continuous_scrape.crawler = mock_crawler
    continuous_scrape.run_conf = MagicMock()
    continuous_scrape.indexing_pipeline = mock_indexing_pipeline
    continuous_scrape.f = mock_open()()
    
    with patch('builtins.print'):  # Suppress print statements
        urls = await continuous_scrape.get_all_urls(start_url)
        
        # Should return empty list since 404 page is skipped
        assert len(urls) == 0


@pytest.mark.asyncio
async def test_get_all_urls_human_verification(monkeypatch, temp_db):
    """Test get_all_urls handling human verification pages."""
    start_url = "https://rocm.blogs.amd.com"
    
    # Mock config values
    monkeypatch.setattr(continuous_scrape.config, "ROCM_RAG_SET_MAX_NUM_PAGES", False)
    monkeypatch.setattr(continuous_scrape.config, "ROCM_RAG_VALID_PAGE_FILTERS", [r"/blog/"])
    monkeypatch.setattr(continuous_scrape.config, "ROCM_RAG_VALID_EXTENSIONS", [".html"])
    monkeypatch.setattr(continuous_scrape.config, "ROCM_RAG_PAGE_NOT_FOUND_FILTERS", [r"404"])
    monkeypatch.setattr(continuous_scrape.config, "ROCM_RAG_REQUIRE_HUMAN_VERIFICATION_FILTERS", [r"Verifying"])
    
    # Mock crawler and results - human verification page
    mock_crawler = AsyncMock()
    mock_result = MagicMock()
    mock_result.url = "https://rocm.blogs.amd.com/blog/protected.html"
    mock_result.success = True
    mock_result.markdown = "Verifying you are human. Please wait..."
    mock_result.links = {"internal": []}
    
    # First call returns verification page, second call returns empty (simulating retry)
    mock_crawler.arun_many.side_effect = [[mock_result], []]
    
    # Mock indexing pipeline
    mock_indexing_pipeline = MagicMock()
    
    # Set global variables at module level for this test
    continuous_scrape.crawler = mock_crawler
    continuous_scrape.run_conf = MagicMock()
    continuous_scrape.indexing_pipeline = mock_indexing_pipeline
    continuous_scrape.f = mock_open()()
    
    with patch('builtins.print'):  # Suppress print statements
        urls = await continuous_scrape.get_all_urls(start_url)
        
        # Should return empty list since verification page is retried but then empty
        assert len(urls) == 0


def test_save_or_update_page_hash_updated_flow(temp_db):
    """Test the updated flow in save_or_update_page_hash."""
    url = "https://rocm.example.com/blog/test"
    domain = "rocm.example.com"
    
    # Insert initial content
    status1 = continuous_scrape.save_or_update_page_hash(url, domain, "original content")
    assert status1 == continuous_scrape.UpdateStatus.NEW
    
    # Update with different content - this tests the get_page_hash call in the update flow
    status2 = continuous_scrape.save_or_update_page_hash(url, domain, "updated content")
    assert status2 == continuous_scrape.UpdateStatus.UPDATED


@pytest.mark.asyncio
async def test_get_all_urls_max_pages_break(monkeypatch, temp_db):
    """Test get_all_urls breaks when reaching max pages in the middle of processing."""
    start_url = "https://rocm.blogs.amd.com"
    
    # Mock config values with max pages limit
    monkeypatch.setattr(continuous_scrape.config, "ROCM_RAG_SET_MAX_NUM_PAGES", True)
    monkeypatch.setattr(continuous_scrape.config, "ROCM_RAG_MAX_NUM_PAGES", 1, raising=False)
    monkeypatch.setattr(continuous_scrape.config, "ROCM_RAG_VALID_PAGE_FILTERS", [r"/blog/"])
    monkeypatch.setattr(continuous_scrape.config, "ROCM_RAG_VALID_EXTENSIONS", [".html"])
    monkeypatch.setattr(continuous_scrape.config, "ROCM_RAG_PAGE_NOT_FOUND_FILTERS", [r"404"])
    monkeypatch.setattr(continuous_scrape.config, "ROCM_RAG_REQUIRE_HUMAN_VERIFICATION_FILTERS", [r"Verifying"])
    
    # Mock crawler and results - multiple results to trigger the break condition
    mock_crawler = AsyncMock()
    mock_result1 = MagicMock()
    mock_result1.url = "https://rocm.blogs.amd.com/blog/post1.html"
    mock_result1.success = True
    mock_result1.markdown = "Blog content 1"
    mock_result1.links = {"internal": []}
    
    mock_result2 = MagicMock()
    mock_result2.url = "https://rocm.blogs.amd.com/blog/post2.html"
    mock_result2.success = True
    mock_result2.markdown = "Blog content 2"
    mock_result2.links = {"internal": []}
    
    # Return multiple results that would exceed the limit
    mock_crawler.arun_many.return_value = [mock_result1, mock_result2]
    
    # Mock indexing pipeline
    mock_indexing_pipeline = MagicMock()
    
    # Set global variables at module level for this test
    continuous_scrape.crawler = mock_crawler
    continuous_scrape.run_conf = MagicMock()
    continuous_scrape.indexing_pipeline = mock_indexing_pipeline
    continuous_scrape.f = mock_open()()
    
    with patch('builtins.print') as mock_print:  # Capture print statements
        urls = await continuous_scrape.get_all_urls(start_url)
        
        # Should hit the max pages limit
        assert len(urls) == 1
        # Check that the break message was printed
        mock_print.assert_any_call("Reached max number of pages to index: 1")

