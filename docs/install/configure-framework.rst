.. meta::
  :description: Set up and configure a ROCm-RAG framework
  :keywords: RAG, ROCm, install, Docker, frameworks, LLM

.. _rocm-rag-installation:

*************************************************************
ROCm-RAG installation
*************************************************************

This topic covers setup and install instructions to help you get started running ROCm-RAG.

System requirements
=============================================================

To use ROCm-RAG `1.0.0 <https://github.com/ROCm/rocm-rag/tree/release/1.0.0>`__, you need the following prerequisites:

- **ROCm version:** `6.4.1 <https://rocm.docs.amd.com/en/docs-6.4.1/>`__
- **Operating system:** Ubuntu 22.04
- **GPU platform:** AMD Instinct™ MI300X

.. note::

   - If you're hosting the LLM outside the Docker container, the container requires one MI300X GPU. By default, the GPU ID is ``0``, but this can be changed by setting the :ref:`environment variables <rag-environment-variables>` for ``ROCM_RAG_EMBEDDER_TP`` and ``ROCM_RAG_EMBEDDER_GPU_IDS``.
   - If you're hosting the LLM inside the container (``ROCM_RAG_USE_EXAMPLE_LLM`` is set to ``true``), three MI300X GPUs are required. By default, the GPU IDs are ``0``, ``1``, ``2``, but these can be changed by setting the :ref:`environment variables <rag-environment-variables>` for ``ROCM_RAG_EMBEDDER_TP``, ``ROCM_RAG_EMBEDDER_GPU_IDS``, ``ROCM_RAG_LLM_TP``, and ``ROCM_RAG_LLM_GPU_IDS``.


Install ROCm-RAG
==============================================================

To install ROCm-RAG on ROCm, you have the following options:

* :ref:`using-docker-with-rag-pre-installed` **(recommended)**
* :ref:`build-rocm-rag-docker-image`

After setting up the container with either option, configure your RAG framework, inferencing framework, and :ref:`environment variables <rag-environment-variables>` before running the pipelines.

.. _using-docker-with-rag-pre-installed:

Use a prebuilt Docker image with ROCm-RAG pre-installed
---------------------------------------------------------------

The prebuilt image contains a fully configured ROCm-RAG installation and all required dependencies.

Pull the Docker image.

.. code-block:: bash 

   docker pull rocm/rocm-rag:rocm-rag-1.0.0-rocm6.4.1-ubuntu22.04

.. _build-rocm-rag-docker-image:

Build from source
---------------------------------------------------------------

ROCm-RAG can be built from source using the provided Dockerfile.

1. Clone the `https://github.com/ROCm/rocm-rag <https://github.com/ROCm/rocm-rag>`__ repository.

   .. code-block:: bash

      git clone https://github.com/ROCm/rocm-rag.git --recursive
      cd rocm-rag
  
2. Build the Docker image.
  
   .. code-block:: bash  
    
      docker build -t rocm-rag -f Dockerfile .


Configure ROCm-RAG
==============================================================

Before running ROCm-RAG, you need to configure the RAG framework, inferencing framework, and :ref:`environment variables <rag-environment-variables>`.

Choose a RAG framework
---------------------------------------------------------------

The ROCm-RAG implementation leverages two widely adopted RAG frameworks:

- `Haystack <https://haystack.deepset.ai/>`__: An open source framework designed for building search systems, QA pipelines, and RAG workflows.
- `LangGraph <https://www.langchain.com/langgraph>`__: A modular framework tailored for developing applications powered by language models.   

Choose a framework that best suits your preferences and workflow. Both frameworks are actively maintained and widely used in the field of LLM-based application development.

You can configure a framework by setting :ref:`environment variables <rag-environment-variables>` when running the Docker container:

.. code-block:: bash 

   # Options: haystack, langgraph
   ROCM_RAG_EXTRACTION_FRAMEWORK=haystack
   ROCM_RAG_RETRIEVAL_FRAMEWORK=haystack


Choose an inferencing framework
---------------------------------------------------------------

ROCm-RAG supports three inferencing frameworks:

- `SGLang <https://github.com/sgl-project/sglang.git>`__: An LLM serving engine known for radix tree caching and speculative decoding for fast inference.
- `vLLM <https://github.com/vllm-project/vllm.git>`__: An efficient LLM inference library built around PagedAttention for fast, memory-optimized serving.
- `llama.cpp <https://github.com/ggml-org/llama.cpp.git>`__: A lightweight C/C++ inference framework for running GGUF-quantized LLMs locally on CPUs and GPUs.

Choose the framework that best suits your preferences and workflow, then follow the setup guide to deploy your inference server.

Using the example LLM (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you prefer to test the pipeline without deploying your own inference server, enable the example LLM by setting this environment variable:

.. code-block:: bash 
  
   ROCM_RAG_USE_EXAMPLE_LLM=True

By default, this launches ``Qwen/Qwen3-30B-A3B-Instruct-2507`` using vLLM inside the provided Docker container, running on GPUs with logical IDs ``1`` and ``2``. 

If you're using the example LLM, you can skip the inferencing framework setup steps below.

Deploy an external inference server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you set ``ROCM_RAG_USE_EXAMPLE_LLM=False``, follow these steps to deploy an LLM inference server outside the ROCm-RAG container.

.. tab-set::

   .. tab-item:: SGLang

      Deploy DeepSeek V3 using SGLang:

      .. code-block:: bash 

         # On a separate node
         docker run --cap-add=SYS_PTRACE --ipc=host --privileged=true \
                 --shm-size=128GB --network=host --device=/dev/kfd \
                 --device=/dev/dri --group-add video -it \
         lmsysorg/sglang:v0.5.3rc0-rocm630-mi30x

         RCCL_MSCCL_ENABLE=0 CK_MOE=1 HSA_NO_SCRATCH_RECLAIM=1 \
         python3 -m sglang.launch_server \
         --model-path deepseek-ai/DeepSeek-V3.1 \
         --host 0.0.0.0 --port 30000 --tp 8 --trust-remote-code

   .. tab-item:: vLLM

      See `Accelerated LLM Inference on AMD Instinct™ GPUs with vLLM 0.9.x and ROCm <https://rocm.blogs.amd.com/software-tools-optimization/vllm-0.9.x-rocm/README.html>`__ for deployment instructions.

   .. tab-item:: llama.cpp

      1. Download the GGUF model files.

         .. code-block:: python
            
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

      2. Build the ``llama.cpp`` Docker image.

         .. code-block:: bash 

            git clone https://github.com/ROCm/llama.cpp
            cd llama.cpp/
            docker build -t local/llama.cpp:rocm6.4_ubuntu24.04-complete \
            --target build -f .devops/rocm.Dockerfile .

      3. Start the Docker container with your checkpoints directory mounted.

         .. code-block:: bash

            docker run --cap-add=SYS_PTRACE --ipc=host --privileged=true \
              --shm-size=128GB --network=host --device=/dev/kfd \
              --device=/dev/dri --group-add video -it \
              -v <your huggingface cache directory on host>:<your huggingface cache directory inside container> \
            local/llama.cpp:rocm6.4_ubuntu24.04-complete

      4. Launch the ``llama.cpp`` HTTP server.

         .. code-block:: bash 

            cd /app/build/bin
            ./llama-server \
            -m <your huggingface cache directory inside the container>/hub/models--unsloth--DeepSeek-V3.1-GGUF/Q4_K_M/DeepSeek-V3.1-Q4_K_M-00001-of-00009.gguf \
            -ngl 999 -np 4 --alias unsloth/DeepSeek-V3.1-Q4_K_M \
            --host 0.0.0.0 --port 30000

After setting up your inference server, ensure you set the correct API endpoints for LLM server-related :ref:`environment variables <rag-environment-variables>`.


Run ROCm-RAG
==============================================================

Once the ROCm-RAG framework is configured, you can execute the extraction and retrieval pipelines through:

* :doc:`An interactive session <../how-to/run-interactive-session>`
* :doc:`Direct execution with your terminal <../how-to/direct-execute>`


Next steps
---------------------------------------------------------------

Now that you have ROCm-RAG configured on your AMD Instinct GPU, you can:

* Explore different RAG frameworks (Haystack and LangGraph)
* Experiment with different inferencing frameworks (SGLang, vLLM, llama.cpp)
* Customize extraction and retrieval parameters for your use case
* Build custom RAG pipelines for specialized tasks
* Integrate ROCm-RAG into your AI applications

