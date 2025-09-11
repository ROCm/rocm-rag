from pathlib import Path
import os

def require_env(var_name: str) -> str:
    """Return env variable or raise if missing/empty."""
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(f"Environment variable '{var_name}' must be set and non-empty")
    return value

ROCM_RAG_WORKSPACE = Path(os.getenv("ROCM_RAG_WORKSPACE", "/rag-workspace"))
ROCM_RAG_HASH_DIR = Path(os.getenv(
    "ROCM_RAG_HASH_DIR",
    ROCM_RAG_WORKSPACE / "rocm-rag" / "hash" / "page_hashes"
))

# RAG framework: haystack or langgraph
ROCM_RAG_EXTRACTION_FRAMEWORK = os.getenv("ROCM_RAG_EXTRACTION_FRAMEWORK", "haystack").lower()
if ROCM_RAG_EXTRACTION_FRAMEWORK not in ("haystack", "langgraph"):
    raise ValueError("ROCM_RAG_EXTRACTION_FRAMEWORK must be either 'haystack' or 'langgraph'")
    
ROCM_RAG_RETRIEVAL_FRAMEWORK = os.getenv("ROCM_RAG_RETRIEVAL_FRAMEWORK", "haystack").lower()
if ROCM_RAG_RETRIEVAL_FRAMEWORK not in ("haystack", "langgraph"):
    raise ValueError("ROCM_RAG_RETRIEVAL_FRAMEWORK must be either 'haystack' or 'langgraph'")

# extraction pipeline
ROCM_RAG_HAYSTACK_SERVER_PORT = int(os.getenv("ROCM_RAG_HAYSTACK_SERVER_PORT", 1416))
ROCM_RAG_LANGGRAPH_SERVER_PORT = int(os.getenv("ROCM_RAG_LANGGRAPH_SERVER_PORT", 20000))
ROCM_RAG_EMBEDDER_MODEL = os.getenv("ROCM_RAG_EMBEDDER_MODEL", "intfloat/e5-mistral-7b-instruct")
ROCM_RAG_EMBEDDER_API_BASE_URL = os.getenv("ROCM_RAG_EMBEDDER_API_BASE_URL", "http://localhost")
ROCM_RAG_EMBEDDER_API_PORT = int(os.getenv("ROCM_RAG_EMBEDDER_API_PORT", 10000))
ROCM_RAG_EMBEDDER_MAX_TOKENS = int(os.getenv("ROCM_RAG_EMBEDDER_MAX_TOKENS", 4096))
ROCM_RAG_WEAVIATE_URL = os.getenv("ROCM_RAG_WEAVIATE_URL", "http://localhost")
ROCM_RAG_WEAVIATE_PORT = int(os.getenv("ROCM_RAG_WEAVIATE_PORT", 40000))
ROCM_RAG_WAIT_VECTOR_DB_TIMEOUT = int(os.getenv("ROCM_RAG_WAIT_VECTOR_DB_TIMEOUT", 120))  # seconds
ROCM_RAG_WAIT_EMBEDDER_TIMEOUT = int(os.getenv("ROCM_RAG_WAIT_EMBEDDER_TIMEOUT", 300))  # seconds

ROCM_RAG_START_URLS = os.getenv("ROCM_RAG_START_URLS", "https://rocm.blogs.amd.com/").split(",")
ROCM_RAG_VALID_EXTENSIONS = os.getenv("ROCM_RAG_VALID_EXTENSIONS", ".html,.htm,.css,.xml,.txt").split(",")
ROCM_RAG_VALID_PAGE_FILTERS = os.getenv("ROCM_RAG_VALID_PAGE_FILTERS", r"/README.html").split(",")
ROCM_RAG_REQUIRE_HUMAN_VERIFICATION_FILTERS = os.getenv(
    "ROCM_RAG_REQUIRE_HUMAN_VERIFICATION_FILTERS", r"Verifying you are human"
).split(",")
ROCM_RAG_PAGE_NOT_FOUND_FILTERS = os.getenv(
    "ROCM_RAG_PAGE_NOT_FOUND_FILTERS", r"404 - Page Not Found"
).split(",")

ROCM_RAG_VISITED_URL_FILE = Path(os.getenv(
    "ROCM_RAG_VISITED_URL_FILE",
    ROCM_RAG_WORKSPACE / "rocm-rag" / "logs" / "visited_urls.txt"
))
ROCM_RAG_WEAVIATE_CLASSNAME = os.getenv("ROCM_RAG_WEAVIATE_CLASSNAME", "ROCmRAGOnline")

# limit max num pages to crawl, useful for test purpose
ROCM_RAG_SET_MAX_NUM_PAGES = os.getenv("ROCM_RAG_SET_MAX_NUM_PAGES", "False").lower() in ("true", "1", "t", "y", "yes") # by default do NOT limit max num pages
if ROCM_RAG_SET_MAX_NUM_PAGES:
    ROCM_RAG_MAX_NUM_PAGES = int(os.getenv("ROCM_RAG_MAX_NUM_PAGES", 100)) 

# chunking
ROCM_RAG_MAX_CHUNK_LENGTH = int(os.getenv("ROCM_RAG_MAX_CHUNK_LENGTH", 512))
ROCM_RAG_SIMILARITY_THRESHOLD = float(os.getenv("ROCM_RAG_SIMILARITY_THRESHOLD", 0.5))
ROCM_RAG_LANGGRAPH_BREAKPOINT_THRESHOLD_TYPE= os.getenv("ROCM_RAG_LANGGRAPH_BREAKPOINT_THRESHOLD_TYPE", "percentile") 
ROCM_RAG_LANGGRAPH_BREAKPOINT_THRESHOLD_AMOUNT= float(os.getenv("ROCM_RAG_LANGGRAPH_BREAKPOINT_THRESHOLD_AMOUNT", 50))


# retrieval pipeline
print(os.getenv("ROCM_RAG_USE_EXAMPLE_LLM", "False").lower())
ROCM_RAG_USE_EXAMPLE_LLM = os.getenv("ROCM_RAG_USE_EXAMPLE_LLM", "False").lower() in ("true", "1", "t", "y", "yes") # by default NOT use example LLM, you need to follow README to setup your own LLM server
print(f"ROCM_RAG_USE_EXAMPLE_LLM: {ROCM_RAG_USE_EXAMPLE_LLM}")
if ROCM_RAG_USE_EXAMPLE_LLM:
    print("Using example LLM server for quick testing inside this docker, please refer to README to set up your own LLM server for better performance")
    ROCM_RAG_LLM_API_BASE_URL = os.getenv("ROCM_RAG_LLM_API_BASE_URL", "http://localhost")
    ROCM_RAG_LLM_API_PORT = int(os.getenv("ROCM_RAG_LLM_API_PORT", 30000))
    ROCM_RAG_LLM_MODEL = os.getenv("ROCM_RAG_LLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
    ROCM_RAG_LLM_TP = int(os.getenv("ROCM_RAG_LLM_TP", 8))  # tensor parallel size
else:
    print("Not using example LLM server, please set up your own LLM server and update the config accordingly")
    ROCM_RAG_LLM_API_BASE_URL = require_env("ROCM_RAG_LLM_API_BASE_URL")
    ROCM_RAG_LLM_API_PORT = int(require_env("ROCM_RAG_LLM_API_PORT"))
    ROCM_RAG_LLM_MODEL = require_env("ROCM_RAG_LLM_MODEL")

ROCM_RAG_HAYSTACK_CERTAINTY_THRESHOLD = float(os.getenv("ROCM_RAG_HAYSTACK_CERTAINTY_THRESHOLD", 0.80))
ROCM_RAG_HAYSTACK_TOP_K_RANKING = int(os.getenv("ROCM_RAG_HAYSTACK_TOP_K_RANKING", 10))
ROCM_RAG_LANGGRAPH_TOP_K_RANKING = int(os.getenv("ROCM_RAG_LANGGRAPH_TOP_K_RANKING", 10))

OPENAI_API_KEY = require_env("OPENAI_API_KEY")

