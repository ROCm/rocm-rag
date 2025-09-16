#!/bin/bash
set -euo pipefail

# Keep track of PIDs
pids=()
extraction_pid=""

# Cleanup function: gracefully stop all child processes
cleanup() {
    echo "Shutting down services..."
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo " → Stopping background service (PID $pid)"
            kill -9 -"$pid" 2>/dev/null  # note the minus sign
        else
            echo " → Background service (PID $pid) already stopped"
        fi
    done
    wait
    echo "All background processes have been closed."
}
trap cleanup EXIT

# Start Weaviate

cd "${ROCM_RAG_WORKSPACE}/rocm-rag/external/weaviate/cmd/weaviate-server" || exit

export QUERY_DEFAULTS_LIMIT=20 
export AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true 
export DEFAULT_VECTORIZER_MODULE=none 
# export ENABLE_MODULES=backup-filesystem
# export BACKUP_FILESYSTEM_PATH="${ROCM_RAG_WORKSPACE}/rocm-rag/backup"
export PERSISTENCE_DATA_PATH="${ROCM_RAG_WORKSPACE}/rocm-rag/data/${ROCM_RAG_EXTRACTION_FRAMEWORK}/weaviate" 
export DEFAULT_PORT="${ROCM_RAG_WEAVIATE_PORT}" 
export TLS_DISABLED=true 

setsid go run main.go --host 0.0.0.0 --port "${ROCM_RAG_WEAVIATE_PORT}" --scheme http \
> "${ROCM_RAG_WORKSPACE}/rocm-rag/logs/weaviate_extraction.log" &
pids+=($!)

# Start Embedder

setsid vllm serve "${ROCM_RAG_EMBEDDER_MODEL}" --task embed --port "${ROCM_RAG_EMBEDDER_API_PORT}" \
--tensor-parallel-size "${ROCM_RAG_EMBEDDER_TP}" --trust-remote-code \
> "${ROCM_RAG_WORKSPACE}/rocm-rag/logs/embedder_extraction.log" &
pids+=($!)

# Start extraction pipeline
cd "${ROCM_RAG_WORKSPACE}/rocm-rag/rocm_rag/extraction" || exit
setsid python3 continuous_scrape.py \
> "${ROCM_RAG_WORKSPACE}/rocm-rag/logs/extraction.log" &
extraction_pid=$!
pids+=($extraction_pid)

# Wait for *any* process to exit
wait -n
EXIT_CODE=$?

# Check if extraction process exited and log accordingly
if ! kill -0 "$extraction_pid" 2>/dev/null; then
    # Extraction process has exited - get its specific exit code
    wait "$extraction_pid"
    EXTRACTION_EXIT_CODE=$?
    
    if [ $EXTRACTION_EXIT_CODE -eq 0 ]; then
        echo "Extraction process (continuous_scrape.py) exited normally."
        echo ""
        echo "================================================================"
        echo "||                                                            ||"
        echo "||          *** EXTRACTION COMPLETED SUCCESSFULLY! ***        ||"
        echo "||                                                            ||"
        echo "================================================================"
        echo ""
    else
        echo "Extraction process (continuous_scrape.py) exited with error code $EXTRACTION_EXIT_CODE."
    fi
else
    echo "A background process (not extraction) exited with code $EXIT_CODE."
fi

cleanup
exit $EXIT_CODE