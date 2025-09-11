.. meta::
  :description: Run ROCm-RAG from a docker container in interactive mode
  :keywords: RAG, ROCm, extraction, how-to, docker, retrieval

************************************
Run ROCm-RAG with a docker container
************************************

You can run the ROCm RAG container in an interactive session. Use interactive mode to explore and debug the container manually.   

Start the container and open a bash session
===========================================

.. code:: bash 

  docker run --cap-add=SYS_PTRACE --ipc=host --privileged=true \
          --shm-size=128GB --network=host --device=/dev/kfd \
          --device=/dev/dri --group-add video -it \
          -v <mount dir>:<mount dir> \
  rocm-rag:latest

Run the extraction pipeline
===========================

By default, the scraper will scrape all blog pages starting from the `ROCm Blogs Home <https://rocm.blogs.amd.com/index.html>`__. 
You can limit the scope of scraping by setting the following env variables. Here's an example of scraping a single blog page about ROCm-DS, 
so the extraction pipeline will build the knowledge base *only* from this blog page:

.. code:: bash 

  export ROCM_RAG_START_URLS="https://rocm.blogs.amd.com/software-tools-optimization/introducing-rocm-ds-revolutionizing-data-processing-with-amd-instinct-gpus/README.html"
  export ROCM_RAG_SET_MAX_NUM_PAGES=True
  export ROCM_RAG_MAX_NUM_PAGES=1

Then start the extraction pipeline. By default, the page hash database is under ``/rag-workspace/rocm-rag/hash``, and the Weaviate persistent data is under ``/rag-workspace/rocm-rag/data``. 
Ensure you mount these directories to your host directories so that you can reuse the extracted data in the future. 

.. code:: bash

  cd rocm-rag/scripts
  bash run-extraction-tmux.sh # OR bash run-extraction.sh

You can switch to a different bash terminal window with ``tmux select-window -t <window ID>``.   
Logs for extraction components can be found under ``/rag-workspace/rocm-rag/logs``.   
The extraction script crawls, scrapes, chunks, indexes, and saves the content to the database for future use. 
At the end of scraping and indexing, you'll see a message ``All URLs found: <your starting URL>``. Once finished, you can run a quick test to retrieve content from the Weaviate vector database.

.. code:: bash 

  cd /rag-workspace/rocm-rag/tests/examples
  python weaviate_client_example.py

Configure the SSL certificate and enable HTTPs    
==============================================

Access to the ASR feature requires microphone permissions for some browsers, which in turn necessitates a secure HTTPS connection. 
You can configure HTTPS using a self-signed certificate for testing purposes *only*. 

.. warning::
  
  This approach must *not* be used in production environments. For production deployments, configure a valid domain name and obtain an SSL/TLS certificate from a trusted certificate authority.

1. Get the machine IP. 

    .. code:: bash

      # get private IP inside LAN
      apt install net-tools && ifconfig

2. Once you get the IP address, create a self-signed certificate with the SAN included. This creates a self-signed SSL certificate valid for 365 days saved to ``/etc/nginx/ssl/selfsigned.crt``. It's saved together with a new RSA private key in ``/etc/nginx/ssl/selfsigned.key``:

    .. code:: bash

      mkdir -p /etc/nginx/ssl
      openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout /etc/nginx/ssl/selfsigned.key \
        -out /etc/nginx/ssl/selfsigned.crt \
        -subj "/C=US/ST=Local/L=Local/O=Local/CN=<your machine IP>" \
        -addext "subjectAltName=IP:<your machine IP>"

 
3. Start nginx (if it isn't already running):

    .. code:: bash 
      
      nginx

4. Configure nginx to enable HTTPs:

    .. code:: bash 

      cat <<'EOF' >> /etc/nginx/sites-available/default
      server {
          listen 443 ssl;
          listen [::]:443 ssl;
          server_name <your IP address>;  # Accept any hostname

          ssl_certificate     /etc/nginx/ssl/selfsigned.crt;
          ssl_certificate_key /etc/nginx/ssl/selfsigned.key;

          location / {
              proxy_pass http://localhost:8080;
              proxy_set_header Host $host;
              proxy_set_header X-Real-IP $remote_addr;
              proxy_set_header Accept-Encoding "";
              proxy_set_header X-Forwarded-Scheme $scheme;
              proxy_set_header X-Forwarded-Proto $scheme;
              proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

              # Websockets
              proxy_http_version 1.1;
              proxy_set_header Upgrade $http_upgrade;
              proxy_set_header Connection "upgrade";
              ##
              # Disable buffering for the streaming responses (SSE)
              chunked_transfer_encoding off;
              proxy_buffering off;
              proxy_cache off;
              ##
              # Conections Timeouts (1hr)
              keepalive_timeout 3600;
              proxy_connect_timeout 3600;
              proxy_read_timeout 3600;
              proxy_send_timeout 3600;
              ##
          }
      }
      EOF
    
5. Reload nginx:

    .. code:: bash

      reload nginx

6. Test and reload the nginx configuration. ``-t`` checks for syntax errors in the config files, and ``-s`` reloads the configuration without stopping the service:
      
    .. code:: bash

      nginx -t
      nginx -s reload

Run the retrieval pipeline 
==========================

.. code:: bash 

  cd rocm-rag/scripts
  bash run-retrieval-tmux.sh # OR bash run-retrieval.sh

You can switch to a different bash terminal window with ``tmux select-window -t <window ID>``.
The retrieval component logs are under ``/rag-workspace/rocm-rag/logs``.
