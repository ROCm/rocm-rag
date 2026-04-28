.. meta::
  :description: Configure ROCm-RAG framework environment variables
  :keywords: RAG, ROCm, install, environment variables, frameworks, LLM

.. _rag-environment-variables:

*************************************************************
Configure environment variables for ROCm-RAG
*************************************************************

You can configure both extraction and retrieval parameters by setting environment variables for the Docker container in :ref:`rocm-rag-installation`.
There are three ways to set environment variables:

.. tab-set::

   .. tab-item:: .env file (recommended)

      1. Start with `default.env <https://github.com/ROCm/rocm-rag/blob/main/default.env>`__ as a base.
      2. Modify the variables as needed and provide the ``.env`` file when running the container:

         .. code-block:: bash

            docker run --env-file <your env file> ...

   .. tab-item:: Docker run

      Set variables individually when starting the container:

      .. code-block:: bash

         docker run -e VAR1=value1 -e VAR2=value2 ...

   .. tab-item:: Export in container

      Export variables inside the container when running in interactive mode:

      .. code-block:: bash

         export VAR1=value1
         export VAR2=value2


Environment variable reference
---------------------------------------------------------------

The following tables list the configurable environment variables for ROCm-RAG.

**Workspace and storage variables**

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Variable
     - Description
   * - ``ROCM_RAG_WORKSPACE``
     - ROCm-RAG workspace directory
   * - ``ROCM_RAG_HASH_DIR``
     - Directory to save page-level hash
   * - ``ROCM_RAG_VISITED_URL_FILE``
     - File to save list of scraped URLs

**Extraction parameters**

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Variable
     - Description
   * - ``ROCM_RAG_EXTRACTION_FRAMEWORK``
     - Extraction RAG framework (``haystack`` or ``langgraph``)
   * - ``ROCM_RAG_HAYSTACK_SERVER_PORT``
     - Haystack pipeline server port
   * - ``ROCM_RAG_LANGGRAPH_SERVER_PORT``
     - LangGraph server port
   * - ``ROCM_RAG_EMBEDDER_MODEL``
     - Embedder model
   * - ``ROCM_RAG_EMBEDDER_API_BASE_URL``
     - Embedder API base URL
   * - ``ROCM_RAG_EMBEDDER_API_PORT``
     - Embedder API port
   * - ``ROCM_RAG_EMBEDDER_MAX_TOKENS``
     - Embedder model max token limit
   * - ``ROCM_RAG_WEAVIATE_URL``
     - Weaviate DB API base URL
   * - ``ROCM_RAG_WEAVIATE_PORT``
     - Weaviate DB API port
   * - ``ROCM_RAG_WEAVIATE_CLASSNAME``
     - Weaviate classname
   * - ``ROCM_RAG_WAIT_VECTOR_DB_TIMEOUT``
     - Wait time for vector DB server to be ready
   * - ``ROCM_RAG_WAIT_EMBEDDER_TIMEOUT``
     - Wait time for embedder server to be ready
   * - ``ROCM_RAG_EMBEDDER_TP``
     - Tensor parallelism for embedder
   * - ``ROCM_RAG_EMBEDDER_GPU_IDS``
     - List of visible GPUs when deploying embedder model
   * - ``ROCM_RAG_START_URLS``
     - Start URL for scraping
   * - ``ROCM_RAG_VALID_EXTENSIONS``
     - List of supported URL extensions to scrape
   * - ``ROCM_RAG_VALID_PAGE_FILTERS``
     - List of regex filters for selecting valid pages to scrape
   * - ``ROCM_RAG_REQUIRE_HUMAN_VERIFICATION_FILTERS``
     - List of regex filters for identifying pages that require human verification
   * - ``ROCM_RAG_PAGE_NOT_FOUND_FILTERS``
     - List of regex filters for identifying not found pages
   * - ``ROCM_RAG_SET_MAX_NUM_PAGES``
     - Enable limit on the maximum number of pages to scrape
   * - ``ROCM_RAG_MAX_NUM_PAGES``
     - Maximum number of pages to scrape
   * - ``ROCM_RAG_MAX_CHUNK_LENGTH``
     - Maximum number of tokens for SemanticChunkMerger
   * - ``ROCM_RAG_SIMILARITY_THRESHOLD``
     - Similarity threshold for SemanticChunkMerger to merge

**Retrieval parameters**

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Variable
     - Description
   * - ``ROCM_RAG_RETRIEVAL_FRAMEWORK``
     - Retrieval RAG framework (``haystack`` or ``langgraph``)
   * - ``ROCM_RAG_USE_EXAMPLE_LLM``
     - Deploy example LLM inference server inside this Docker
   * - ``ROCM_RAG_LLM_API_BASE_URL``
     - LLM API base URL
   * - ``ROCM_RAG_LLM_API_PORT``
     - LLM API port
   * - ``ROCM_RAG_LLM_MODEL``
     - LLM model
   * - ``ROCM_RAG_LLM_TP``
     - Tensor parallelism
   * - ``ROCM_RAG_LLM_GPU_IDS``
     - Visible GPUs, for example, LLM
   * - ``ROCM_RAG_HAYSTACK_CERTAINTY_THRESHOLD``
     - Certainty threshold for retrieval
   * - ``ROCM_RAG_HAYSTACK_TOP_K_RANKING``
     - Top K retrieved documents for Haystack retrieval pipeline
   * - ``ROCM_RAG_LANGGRAPH_TOP_K_RANKING``
     - Top K retrieved documents for LangGraph retrieval pipeline
