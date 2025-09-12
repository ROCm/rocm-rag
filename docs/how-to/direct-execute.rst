.. meta::
  :description: Run the ROCm-RAG extraction and retrieval pipelines automatically without entering a container
  :keywords: RAG, ROCm, extraction, pipelines, how-to, container, Open-WebUI

**********************************
Run ROCm-RAG with direct execution 
**********************************

Use direct execution to run the extraction and retrieval pipelines automatically without entering the container. 
You can also limit the scope of scraping (see `Run the extraction pipeline <run-docker-container.rst#run-the-extraction-pipeline>`__ for more information).

.. code:: bash 

  docker run --env-file <your env file> --cap-add=SYS_PTRACE --ipc=host --privileged=true \
            --shm-size=128GB --network=host \
            --device=/dev/kfd --device=/dev/dri \
            --group-add video -it \
            -v <mount dir>:<mount dir> \
            rocm-rag:latest /bin/bash -c \
            "cd /rag-workspace/rocm-rag && \
              bash run-extraction.sh && \
              bash run-retrieval.sh"

Logs save to ``/rag-workspace/rocm-rag/logs`` inside the docker. Mount that directory to your host directory to visualize logs. 
Directly running the container will *not* include HTTPs setup, so your microphone might be blocked by your browser in this approach. 

Set up Open-WebUI
=================

This workflow uses `Open-WebUI <https://github.com/open-webui/open-webui>`__ as an example frontend. 
*After* your retrieval pipeline is up-and-running (make sure all components are ready by checking the logs), 
you can access the Open-WebUI frontend by navigating to ``https://\<Your deploy machine IP\>`` or ``http://\<Your deploy machine IP\>:8080``.   

When you set up a new Open-WebUI account, your user data is saved to ``/rag-workspace/rocm-rag/external/open-webui/backend/data``:   

.. image:: ../images/new_user_signup.png

To set up Open-WebUI:

1. Go to the admin panel.   

    .. image:: ../images/main_page.png

2. Check **Enable User Sign Ups** to allow new user registration. User data saves to ``/rag-workspace/rocm-rag/external/open-webui/backend/data/webui.db``.

    .. image:: ../images/admin_general.png

3. Add APIs to the RAG server (by default, haystack is running on port ``1416`` and langgraph is running on port ``20000``).

    .. image:: ../images/admin_add_new_connection_rag.png

4. Add APIs to the example LLM server (if ``ROCM_RAG_USE_EXAMPLE_LLM=True``, otherwise replace this API URL with your inference server URL).

    .. image:: ../images/admin_add_new_connection_llm.png

- Here are the IPs used in this example:

   .. code::

     http://<Your deploy machine IP>:1416 -> haystack server, provides ROCm-RAG-Haystack if Haystack is chosen as RAG framework
     http://<Your deploy machine IP>:20000/v1 -> langgraph server, provides ROCm-RAG-Langgraph if Langgraph is chosen as RAG framework
     http://<Your deploy machine IP>:30000/v1 -> Qwen/Qwen3-30B-A3B-Instruct-2507 inferencing server if ROCM_RAG_USE_EXAMPLE_LLM=True

- Here's a list of models provided by these APIs. This is retrieved by calling ``http GET API_URL:PORT/v1/models``.   

   .. image:: ../images/admin_list_of_models.png

- These are the model settings. By default, models are only accessible by admin. Make sure you share the model to public (all registered users) or private groups.   

   .. image:: ../images/admin_model_settings.png

5. Select which fast whisper model (here's a `list of models provided by fast-whisper <https://github.com/SYSTRAN/faster-whisper/blob/d3bfd0a305eb9d97c08047c82149c1998cc90fcb/faster_whisper/utils.py#L12>`__) and which TTS voice to use.   

    .. image:: ../images/admin_audio.png

.. Expand on these examples

Here are some examples:

- Asking Qwen3-30B the definition of ROCm-DS:

    .. image:: ../images/direct_llm_response.png

- Asking RAG pipeline the definition of ROCm-DS:

    .. image:: ../images/rag_response.png

- The ROCm-DS blog page:

    .. image:: ../images/rocm_ds_blog.png

