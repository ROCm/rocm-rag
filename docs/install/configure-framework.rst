.. meta::
  :description: Set up and configure a ROCm-RAG framework
  :keywords: RAG, ROCm, install, docker, frameworks, LLM

*****************************************
Set up and configure a ROCm-RAG framework
*****************************************

To set up and configure the ROCm-RAG framework, you must:

1. Build a docker image.
2. Choose an RAG framework.
3. Choose an inferencing framework.
4. Configure extraction and retrieval parameters/set environment variables.

Build the docker image
======================

You can build the docker image using dockerfile. The dockerfile comes with all required dependencies and configurations to run the ROCm-RAG framework.

.. code:: bash

  # Clone the repository
  git clone https://github.com/ROCm/rocm-rag.git --recursive
  cd rocm-rag
  # Build docker image 
  docker build -t rocm-rag -f docker/rocm.Dockerfile . 

Pull the pre-built docker image (optional)
------------------------------------------

Alternatively, you can pull the pre-built docker image from DockerHub instead of building the docker image yourself.

.. code:: bash 

  docker pull <image_name>:<tag>

Choose an RAG framework
=======================

The current implementation leverages two widely adopted RAG frameworks:

- `Haystack <https://haystack.deepset.ai/>`__: An open-source framework designed for building search systems, QA pipelines, and RAG workflows.
- `LangGraph <https://www.langchain.com/langgraph>`__: A modular framework tailored for developing applications powered by language models.   

Both frameworks are actively maintained and widely used in the field of LLM-based application development.   
You can configure a framework by setting enironment variables when running the docker container:

.. code:: bash 

  # Options: haystack, langgraph
  ROCM_RAG_EXTRACTION_FRAMEWORK=haystack
  ROCM_RAG_RETRIEVAL_FRAMEWORK=haystack


Choose an inferencing framework
===============================

- `SGLang <https://github.com/sgl-project/sglang.git>`__: An LLM serving engine known for radix tree caching and speculative decoding for ultra-fast inference.
- `vLLM <https://github.com/vllm-project/vllm.git>`__: An efficient LLM inference library built around PagedAttention for fast, memory-optimized serving.
- `llama.cpp <https://github.com/ggml-org/llama.cpp.git>`__: A lightweight C/C++ inference framework for running GGUF-quantized LLMs locally on CPUs and GPUs.

Follow the setup guide to deploy your inference server. If you prefer to test the pipeline *without* deploying your own inference server, enable the example LLM by setting this env variable:

.. code:: bash 
  
  ROCM_RAG_USE_EXAMPLE_LLM=True

.. GPU 2,3? is this correct?
By default, this launches ``Qwen/Qwen3-30B-A3B-Instruct-2507`` using vLLM inside the provided Docker container on GPU 2,3. You can skip the next step if you're using the example LLM model inside this docker.   

If you set ``ROCM_RAG_USE_EXAMPLE_LLM=False``, follow these steps to deploy an LLM inference server outside the ROCm-RAG container.

SGLang
------

Deploy DeepSeek V3:

.. code:: bash 

  # on a separate node
  docker run --cap-add=SYS_PTRACE --ipc=host --privileged=true \
          --shm-size=128GB --network=host --device=/dev/kfd \
          --device=/dev/dri --group-add video -it \
          -v <mount dir>:<mount dir> \
  lmsysorg/sglang:v0.4.4-rocm630

  RCCL_MSCCL_ENABLE=0 CK_MOE=1  HSA_NO_SCRATCH_RECLAIM=1  python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3-0324 --host 0.0.0.0 --port 30000 --tp 8 --trust-remote-code

vLLM
----

See `Accelerated LLM Inference on AMD Instinct™ GPUs with vLLM 0.9.x and ROCm <https://rocm.blogs.amd.com/software-tools-optimization/vllm-0.9.x-rocm/README.html>`__ for more information.    

``llama.cpp``
-------------

1. Deploy unsloth/DeepSeek-V3-0324-GGUF pulling GGUF checkpoints:

    .. code:: bash 

      from huggingface_hub import snapshot_download

      # Define the model repository and destination directory
      model_id = "unsloth/DeepSeek-V3-0324-GGUF"
      local_dir = "<your huggingface cache directory>/hub/models--unsloth--DeepSeek-V3-0324-GGUF"

      # Download only files matching the pattern "DeepSeek-V3-Q4_K_M*"
      snapshot_download(
          repo_id=model_id,
          local_dir=local_dir,
          local_dir_use_symlinks=False,
          allow_patterns=["Q4_K_M/DeepSeek-V3-0324-Q4_K_M*"]
      )

      print(f"Downloaded GGUF file(s) matching pattern to: {local_dir}")

2. Build the ``llama.cpp`` docker image:

    .. code:: bash 

      git clone https://github.com/ROCm/llama.cpp
      cd llama.cpp/
      docker build -t local/llama.cpp:rocm6.4_ubuntu24.04-complete --target build -f .devops/rocm.Dockerfile .

3. Launch the ``llama.cpp`` HTTP server:

    .. code:: bash 

      ./llama-server -m <your huggingface cache directory>/hub/models--unsloth--DeepSeek-V3-0324-GGUF/Q4_K_M/DeepSeek-V3-0324-Q4_K_M-00001-of-00009.gguf -ngl 999 -np 4 --alias unsloth/DeepSeek-V3-0324-Q4_K_M --host 0.0.0.0 --port 30000

Ensure you set the correct APIs for LLM server-related environment variables once you finish setting up your inference server.    

Configure extraction and retrieval parameters
=============================================

You can configure both *extraction* and *retrieval* parameters by setting environment variables for the Docker container:

1. Review the list of environment variables carefully.
2. Set each variable to the correct value based on your configuration and needs.

Use an env file
-----------------

.. The link to default.env will need to change when we move this repo to a public repo.
1. Start with `default.env <https://github.com/ROCm/rocm-rag/blob/main/default.env>`__ as a base. 
2. Modify the variables as needed and provide the env file when running the container:

    .. code:: bash 

      docker run --env-file <your env file> ...

Set variables individually during the docker run
------------------------------------------------

.. code:: bash 

  docker run -e VAR1=value1 -e VAR2=value2 ...

Export variables inside the container 
-------------------------------------

If you're running a container in interactive mode:

.. code:: bash 

  export VAR1=value1
  export VAR2=value2

Ensure all variables are set correctly to ensure the extraction and retrieval pipelines run as expected.

Here's a list of environment variables you may modify as needed:     

Workspace and storage
~~~~~~~~~~~~~~~~~~~~~

.. code:: bash    

  ROCM_RAG_WORKSPACE # ROCm-RAG workspace directory
  ROCM_RAG_HASH_DIR # directory to save page-level hash
  ROCM_RAG_VISITED_URL_FILE # file to save list of scraped URLs

Extraction parameters
~~~~~~~~~~~~~~~~~~~~~

.. code:: bash 

  ROCM_RAG_EXTRACTION_FRAMEWORK # extraction RAG framework
  ROCM_RAG_HAYSTACK_SERVER_PORT # haystack pipeline server port
  ROCM_RAG_LANGGRAPH_SERVER_PORT # langgraph server port
  ROCM_RAG_EMBEDDER_MODEL # embedder model
  ROCM_RAG_EMBEDDER_API_BASE_URL # embedder API base URL
  ROCM_RAG_EMBEDDER_API_PORT # embedder API port
  ROCM_RAG_EMBEDDER_MAX_TOKENS # embedder model max token limit
  ROCM_RAG_WEAVIATE_URL # weaviate db API base URL
  ROCM_RAG_WEAVIATE_PORT # weaviate db API port
  ROCM_RAG_WEAVIATE_CLASSNAME # weaviate classname
  ROCM_RAG_WAIT_VECTOR_DB_TIMEOUT # wait time for vector db server to be ready
  ROCM_RAG_WAIT_EMBEDDER_TIMEOUT # wait time for embedder server to be ready
  ROCM_RAG_EMBEDDER_GPU_IDS # list of visible GPUs when deploy embedder model
  ROCM_RAG_START_URLS # start URL for scraping
  ROCM_RAG_VALID_EXTENSIONS # list of supported URL extensions to scrape
  ROCM_RAG_VALID_PAGE_FILTERS # list of regex filters for selecting valid pages to scrape
  ROCM_RAG_REQUIRE_HUMAN_VERIFICATION_FILTERS # list of regex filters for identifying pages that require human verification
  ROCM_RAG_PAGE_NOT_FOUND_FILTERS # list of regex filters for identifying not found pages
  ROCM_RAG_SET_MAX_NUM_PAGES # enable limit on the maximum number of pages to scrape
  ROCM_RAG_MAX_NUM_PAGES # maximum number of pages to scrape
  ROCM_RAG_MAX_CHUNK_LENGTH # maximum number of tokens for SemanticChunkMerger
  ROCM_RAG_SIMILARITY_THRESHOLD # similarity threshold for SemanticChunkMerger to merge


Retrieval parameters
~~~~~~~~~~~~~~~~~~~~

.. code:: bash 

  ROCM_RAG_RETRIEVAL_FRAMEWORK # retrieval RAG framework
  ROCM_RAG_USE_EXAMPLE_LLM # deploy example LLM inference server inside this docker
  ROCM_RAG_LLM_API_BASE_URL # LLM API base URL
  ROCM_RAG_LLM_API_PORT # LLM API port
  ROCM_RAG_LLM_MODEL # LLM model
  ROCM_RAG_LLM_TP # tensor parallism
  ROCM_RAG_LLM_GPU_IDS # visible GPUs for example LLM
  ROCM_RAG_HAYSTACK_CERTAINTY_THRESHOLD # certainty threshold for retrieval
  ROCM_RAG_HAYSTACK_TOP_K_RANKING # top K retrieved documents for haystack retrieval pipeline
  ROCM_RAG_LANGGRAPH_TOP_K_RANKING # top K retrieved documents for langgraph retrieval pipeline



