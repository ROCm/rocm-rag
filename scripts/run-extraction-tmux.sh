#!/bin/bash

SESSION="rocm_rag_extraction"

# Create tmux session for extraction pipeline
tmux new-session -d -s $SESSION -n weaviate "cd ${ROCM_RAG_WORKSPACE}/rocm-rag/external/weaviate/cmd/weaviate-server && QUERY_DEFAULTS_LIMIT=20 AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true DEFAULT_VECTORIZER_MODULE=none ENABLE_MODULES=backup-filesystem BACKUP_FILESYSTEM_PATH="${ROCM_RAG_WORKSPACE}/rocm-rag/backup" PERSISTENCE_DATA_PATH="${ROCM_RAG_WORKSPACE}/rocm-rag/data/${ROCM_RAG_EXTRACTION_FRAMEWORK}/weaviate" DEFAULT_PORT=${ROCM_RAG_WEAVIATE_PORT} TLS_DISABLED=true go run main.go --host 0.0.0.0 --port ${ROCM_RAG_WEAVIATE_PORT} --scheme http 2>&1 | tee ${ROCM_RAG_WORKSPACE}/rocm-rag/logs/weaviate_extraction.log"
tmux new-window -t $SESSION -n embedder-vllm "vllm serve ${ROCM_RAG_EMBEDDER_MODEL} --task embed --port ${ROCM_RAG_EMBEDDER_API_PORT} --tensor-parallel-size ${ROCM_RAG_EMBEDDER_TP} --trust-remote-code 2>&1 | tee ${ROCM_RAG_WORKSPACE}/rocm-rag/logs/embedder_extraction.log"
tmux new-window -t $SESSION -n extraction-pipeline "cd ${ROCM_RAG_WORKSPACE}/rocm-rag/rocm_rag/extraction && python3 continuous_scrape.py 2>&1 | tee ${ROCM_RAG_WORKSPACE}/rocm-rag/logs/extraction.log"

