# Base image
FROM rocm/vllm:latest

# Set Python working directory
WORKDIR /rag-workspace/rocm-rag

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    net-tools \
    tmux \
    netcat \
    locales \
    openssl \
    nginx \
    build-essential \
    && locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8

# Copy repository
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel \
    && pip install --upgrade -r requirements.txt

# Setup Crawl4AI
RUN crawl4ai-setup \
    && crawl4ai-doctor

# Setup Open-WebUI
WORKDIR /rag-workspace/rocm-rag/external/open-webui

# Install Node.js 22
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && cp -RPp .env.example .env \
    && npm install \
    && npm run build

# Setup backend
WORKDIR /rag-workspace/rocm-rag/external/open-webui/backend
RUN python3 -m venv venv --system-site-packages \
    && . venv/bin/activate \
    && pip install --upgrade pip \
    && pip install --ignore-installed blinker \
    && pip install -r requirements.txt \
    && pip install --upgrade protobuf==6.31.1 \
    && pip install --upgrade grpcio==1.74.0

# Install Go
WORKDIR /rag-workspace/rocm-rag
RUN wget https://go.dev/dl/go1.24.5.linux-amd64.tar.gz \
    && rm -rf /usr/local/go \
    && tar -C /usr/local -xzf go1.24.5.linux-amd64.tar.gz \
    && rm go1.24.5.linux-amd64.tar.gz

# Add Go to PATH
ENV PATH="/usr/local/go/bin:${PATH}"

# ================================================
# Environment Variables
# ================================================

# Workspace & storage
ENV ROCM_RAG_WORKSPACE="/rag-workspace"
ENV ROCM_RAG_HASH_DIR="/rag-workspace/rocm-rag/hash/page_hashes"
ENV ROCM_RAG_VISITED_URL_FILE="/rag-workspace/rocm-rag/logs/visited_urls.txt"

# Extraction pipeline
ENV ROCM_RAG_EXTRACTION_FRAMEWORK="haystack"
ENV ROCM_RAG_HAYSTACK_SERVER_PORT="1416"
ENV ROCM_RAG_LANGGRAPH_SERVER_PORT="20000"
ENV ROCM_RAG_EMBEDDER_MODEL="intfloat/e5-mistral-7b-instruct"
ENV ROCM_RAG_EMBEDDER_API_BASE_URL="http://localhost"
ENV ROCM_RAG_EMBEDDER_API_PORT="10000"
ENV ROCM_RAG_EMBEDDER_MAX_TOKENS="4096"
ENV ROCM_RAG_WEAVIATE_URL="http://localhost"
ENV ROCM_RAG_WEAVIATE_PORT="40000"
ENV ROCM_RAG_WAIT_VECTOR_DB_TIMEOUT="120"
ENV ROCM_RAG_WAIT_EMBEDDER_TIMEOUT="300"
ENV ROCM_RAG_EMBEDDER_TP="2"
ENV ROCM_RAG_EMBEDDER_GPU_IDS="0,1"

# Crawler
ENV ROCM_RAG_START_URLS="https://rocm.blogs.amd.com/index.html"
ENV ROCM_RAG_VALID_EXTENSIONS=".html,.htm,.css,.xml,.txt"
ENV ROCM_RAG_VALID_PAGE_FILTERS="/README.html"
ENV ROCM_RAG_REQUIRE_HUMAN_VERIFICATION_FILTERS="Verifying you are human"
ENV ROCM_RAG_PAGE_NOT_FOUND_FILTERS="404 - Page Not Found"
ENV ROCM_RAG_WEAVIATE_CLASSNAME="ROCmRAG"
# Set this to True to limit the number of pages to crawl, useful for testing
ENV ROCM_RAG_SET_MAX_NUM_PAGES="False"
ENV ROCM_RAG_MAX_NUM_PAGES=

# Chunking
ENV ROCM_RAG_MAX_CHUNK_LENGTH="512"
ENV ROCM_RAG_SIMILARITY_THRESHOLD="0.5"
ENV ROCM_RAG_LANGGRAPH_BREAKPOINT_THRESHOLD_TYPE="percentile"
ENV ROCM_RAG_LANGGRAPH_BREAKPOINT_THRESHOLD_AMOUNT="50"

# Retrieval pipeline
ENV ROCM_RAG_RETRIEVAL_FRAMEWORK="haystack"
ENV ROCM_RAG_USE_EXAMPLE_LLM="True"
# ATTENTION: default set to example LLM, change to your API and deployed model below if not using example LLM
ENV ROCM_RAG_LLM_API_BASE_URL="http://localhost"
ENV ROCM_RAG_LLM_API_PORT="30000"
ENV ROCM_RAG_LLM_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
ENV ROCM_RAG_LLM_TP="2"
ENV ROCM_RAG_LLM_GPU_IDS="2,3"

ENV ROCM_RAG_HAYSTACK_CERTAINTY_THRESHOLD="0.80"
ENV ROCM_RAG_HAYSTACK_TOP_K_RANKING="10"
ENV ROCM_RAG_LANGGRAPH_TOP_K_RANKING="10"

ENV OPENAI_API_KEY="Your-OpenAI-API-Key"


# ================================================
# End Environment Variables
# ================================================


# Install editable Python package
RUN pip install -e .

RUN chmod +x /rag-workspace/rocm-rag/scripts/*.sh
RUN chmod +x /rag-workspace/rocm-rag/tests/*

# Default command
CMD ["/bin/bash"]
