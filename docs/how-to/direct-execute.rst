.. meta::
  :description: Run the ROCm-RAG extraction and retrieval pipelines automatically without entering a container
  :keywords: RAG, ROCm, extraction, pipelines, how-to, container, Open-WebUI

.. _rag-direct-execute:

**********************************
Run ROCm-RAG with direct execution 
**********************************

Use direct execution to run the extraction and retrieval pipelines automatically without needing to enter a container. 

.. note::

   You can also limit the scope of scraping. For more information, see :ref:`extract`.

Run the pipelines
==================

Use the following command in your terminal to directly execute the ROCm-RAG pipelines:

.. code-block:: bash 

   docker run --env-file <your env file> --cap-add=SYS_PTRACE --ipc=host --privileged=true \
             --shm-size=128GB --network=host \
             --device=/dev/kfd --device=/dev/dri \
             --group-add video -it \
             -v <mount dir>:<mount dir> \
             rocm-rag:latest /bin/bash -c \
             "cd /rag-workspace/rocm-rag/scripts && \
              bash run-extraction.sh && \
              bash run-retrieval.sh"

View logs
=========

The logs are saved to ``/rag-workspace/rocm-rag/logs`` inside the Docker container. To view logs on your host machine, mount this directory to a local path.

.. important::

   Running the container directly will not include HTTPS setup, so your microphone might be blocked by your browser with this approach. 
   For more information, see :ref:`configure`.


