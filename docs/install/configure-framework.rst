.. meta::
  :description: Set up and configure a ROCm-RAG framework
  :keywords: RAG, ROCm, install, Docker, frameworks, LLM

***************************************
ROCm-RAG framework on ROCm installation
***************************************

To use the ROCm-RAG framework, it's suggested that you use a Docker image, choose an RAG framework, and then choose an inferencing framework. 
The dependencies required to run these frameworks are preinstalled in the Docker image.
Afterwards, you can configure the extraction and retrieval parameters/set environment variables. 

Once you configure the ROCm-RAG framework, you can deploy the RAG pipelines in :doc:`an interactive session <../how-to/run-interactive-session>`, or you can execute the pipelines :doc:`directly through your terminal <../how-to/direct-execute>`.

System requirements
===================

- ROCm 6.4
- Operating system: Ubuntu 22.04
- These ROCm-RAG frameworks work with the AMD Instinct MI300X GPU:
  
  - If you're hosting the LLM outside the Docker container, then the container requires one MI300X GPU. By default, the GPU ID is ``0``, but this can be changed by setting the environment variables.
  - If you're hosting the LLM inside the container, meaning ``ROCM_RAG_USE_EXAMPLE_LLM`` is set to ``true``, then three MI300X GPUs are required. By default, the GPU IDs are ``0``, ``1``, ``2``, but these can be changed by setting the environment variables.

Using a Docker image with the dependencies preinstalled
=======================================================

The Dockerfile comes with all required dependencies and configurations to run the ROCm-RAG pipeline.

You can pull the prebuilt Docker image from Docker Hub:

.. code:: bash 

  docker pull rocm/rocm-rag:rocm-rag-1.0.0-rocm6.4.1-ubuntu22.04

Alternatively, you can build the Docker image using Dockerfile:

1. Clone the `https://github.com/ROCm/rocm-rag <https://github.com/ROCm/rocm-rag>`__ repository:

   .. code:: bash

     # Clone the repository
     git clone https://github.com/ROCm/rocm-rag.git --recursive
     cd rocm-rag
  
2. Build the Docker image:
  
   .. code:: bash  
    
     # Build docker image 
     docker build -t rocm-rag -f rocm/rocm-rag:rocm-rag-1.0.0-rocm6.4.1-ubuntu22.04 . 

Choose an RAG framework
=======================

The ROCm-RAG implementation leverages two widely adopted RAG frameworks:

- `Haystack <https://haystack.deepset.ai/>`__: An open source framework designed for building search systems, QA pipelines, and RAG workflows.
- `LangGraph <https://www.langchain.com/langgraph>`__: A modular framework tailored for developing applications powered by language models.   

Choose a framework that best suits your preferences and workflow. Both frameworks are actively maintained and widely used in the field of LLM-based application development.   
You can configure a framework by setting environment variables when running the Docker container:

.. code:: bash 

  # Options: haystack, langgraph
  ROCM_RAG_EXTRACTION_FRAMEWORK=haystack
  ROCM_RAG_RETRIEVAL_FRAMEWORK=haystack


Choose an inferencing framework
===============================

There are three inferencing frameworks you can use: 

- `SGLang <https://github.com/sgl-project/sglang.git>`__: An LLM serving engine known for radix tree caching and speculative decoding for fast inference.
- `vLLM <https://github.com/vllm-project/vllm.git>`__: An efficient LLM inference library built around PagedAttention for fast, memory-optimized serving.
- `llama.cpp <https://github.com/ggml-org/llama.cpp.git>`__: A lightweight C/C++ inference framework for running GGUF-quantized LLMs locally on CPUs and GPUs.

Choose the framework that best suits your preferences and workflow. Then follow the setup guide to deploy your inference server. 
If you prefer to test the pipeline *without* deploying your own inference server, enable the example LLM by setting this environment variable:

.. code:: bash 
  
  ROCM_RAG_USE_EXAMPLE_LLM=True

By default, this launches ``Qwen/Qwen3-30B-A3B-Instruct-2507`` using a vLLM inside the provided Docker container, running on GPUs with logical IDs ``1`` and ``2``. 
You can skip the next step if you're using the example LLM model inside this Docker.   

If you set ``ROCM_RAG_USE_EXAMPLE_LLM=False``, follow these steps to deploy an LLM inference server outside the ROCm-RAG container.

SGLang
------

To use the SGLang inferencing framework, deploy DeepSeek V3:

.. code:: bash 

  # on a separate node
  docker run --cap-add=SYS_PTRACE --ipc=host --privileged=true \
          --shm-size=128GB --network=host --device=/dev/kfd \
          --device=/dev/dri --group-add video -it \
  lmsysorg/sglang:v0.5.3rc0-rocm630-mi30x

  RCCL_MSCCL_ENABLE=0 CK_MOE=1  HSA_NO_SCRATCH_RECLAIM=1  python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3.1 --host 0.0.0.0 --port 30000 --tp 8 --trust-remote-code

vLLM
----

To use the vLLM inferencing framework, see `Accelerated LLM Inference on AMD Instinct™ GPUs with vLLM 0.9.x and ROCm <https://rocm.blogs.amd.com/software-tools-optimization/vllm-0.9.x-rocm/README.html>`__ for more information.    

llama.cpp
---------

To use ``llama.cpp``:

1. Deploy unsloth/DeepSeek-V3.1-GGUF:

   .. code:: bash 
      
      from huggingface_hub import snapshot_download
      
      # Define the model repository and destination directory
      model_id = "unsloth/DeepSeek-V3.1-GGUF"
      local_dir = "<your huggingface cache directory>/hub/models--unsloth--DeepSeek-V3.1-GGUF"
      
      # Download only files matching the pattern "DeepSeek-V3.1-Q4_K_M*"
      snapshot_download(
          repo_id=model_id,
          local_dir=local_dir,
          local_dir_use_symlinks=False,
          allow_patterns=["Q4_K_M/DeepSeek-V3.1-Q4_K_M*"]
      )
      
      print(f"Downloaded GGUF file(s) matching pattern to: {local_dir}")

2. Build the ``llama.cpp`` Docker image:

   .. code:: bash 

      git clone https://github.com/ROCm/llama.cpp
      cd llama.cpp/
      docker build -t local/llama.cpp:rocm6.4_ubuntu24.04-complete --target build -f .devops/rocm.Dockerfile .

3. Start your Docker container with your checkpoints directory mounted:

   .. code:: bash

      docker run --cap-add=SYS_PTRACE --ipc=host --privileged=true \
        --shm-size=128GB --network=host --device=/dev/kfd \
        --device=/dev/dri --group-add video -it \
        -v <your huggingface cache directory on host>:<your huggingface cache directory inside container> \
      local/llama.cpp:rocm6.4_ubuntu24.04-complete

4. Launch the ``llama.cpp`` HTTP server:

   .. code:: bash 

      cd /app/build/bin
      ./llama-server -m <your huggingface cache directory inside the container>/hub/models--unsloth--DeepSeek-V3.1-GGUF/Q4_K_M/DeepSeek-V3.1-Q4_K_M-00001-of-00009.gguf -ngl 999 -np 4 --alias unsloth/DeepSeek-V3.1-Q4_K_M --host 0.0.0.0 --port 30000

Ensure you set the correct APIs for LLM server-related environment variables once you finish setting up your inference server.    

Configure the extraction and retrieval parameters
=================================================

You can configure both extraction and retrieval parameters by setting environment variables for the Docker container:

1. Review the list of environment variables carefully.
2. Set each variable to the correct value based on your configuration and needs.

Use an .env file
-----------------

.. The link to default.env will need to change when we move this repo to a public repo.
1. Start with `default.env <https://github.com/AMD-AIOSS/ROCm-RAG-Online/blob/main/default.env>`__ as a base. 
2. Modify the variables as needed and provide the ``.env`` file when running the container:

   .. code:: bash 

      docker run --env-file <your env file> ...

Set variables individually during the Docker run
------------------------------------------------

Use this sample code to individually set variables while the Docker is running:

.. code:: bash 

  docker run -e VAR1=value1 -e VAR2=value2 ...

Export variables inside the container 
-------------------------------------

If you're running a container in interactive mode:

.. code:: bash 

  export VAR1=value1
  export VAR2=value2

Ensure all variables are set correctly to ensure the extraction and retrieval pipelines run as expected.

Customizable environment variables 
----------------------------------

Here's a list of environment variables you may modify as needed.   

Workspace and storage variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Next steps
==========

Now that the ROCm-RAG framework is configured, you can execute the extraction and retrieval pipelines through:

* :doc:`An interactive session <../how-to/run-interactive-session>`
* :doc:`Direct execution with your terminal <../how-to/direct-execute>`
