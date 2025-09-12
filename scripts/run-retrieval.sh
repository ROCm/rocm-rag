#!/bin/bash

# Keep track of PIDs
pids=()

# Cleanup function: kill all child processes
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


# Open WebUI backend
cd "${ROCM_RAG_WORKSPACE}/rocm-rag/external/open-webui/backend" || exit
source venv/bin/activate
setsid sh dev.sh > "${ROCM_RAG_WORKSPACE}/rocm-rag/logs/openwebui_backend.log" & 
pids+=($!)

# Weaviate server
cd "${ROCM_RAG_WORKSPACE}/rocm-rag/external/weaviate/cmd/weaviate-server" || exit

export QUERY_DEFAULTS_LIMIT=20
export AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true
export DEFAULT_VECTORIZER_MODULE=none
export ENABLE_MODULES=backup-filesystem
export BACKUP_FILESYSTEM_PATH="${ROCM_RAG_WORKSPACE}/rocm-rag/backup"
export PERSISTENCE_DATA_PATH="${ROCM_RAG_WORKSPACE}/rocm-rag/data/${ROCM_RAG_RETRIEVAL_FRAMEWORK}/weaviate"
export DEFAULT_PORT=${ROCM_RAG_WEAVIATE_PORT}
export TLS_DISABLED=true

setsid go run main.go --host 0.0.0.0 --port "${ROCM_RAG_WEAVIATE_PORT}" --scheme http \
> "${ROCM_RAG_WORKSPACE}/rocm-rag/logs/weaviate_retrieval.log" &
pids+=($!)

# Embedder vLLM
export HIP_VISIBLE_DEVICES=${ROCM_RAG_EMBEDDER_GPU_IDS}
setsid vllm serve ${ROCM_RAG_EMBEDDER_MODEL} --task embed --tensor-parallel-size ${ROCM_RAG_EMBEDDER_TP} --port ${ROCM_RAG_EMBEDDER_API_PORT} --trust-remote-code \
> "${ROCM_RAG_WORKSPACE}/rocm-rag/logs/embedder_retrieval.log" & 
pids+=($!)

# Retrieval framework
if [[ "${ROCM_RAG_RETRIEVAL_FRAMEWORK,,}" == "haystack" ]]; then
    # Haystack server
    cd "${ROCM_RAG_WORKSPACE}/rocm-rag/rocm_rag/retrieval/haystack" || exit
    setsid hayhooks run --host 0.0.0.0 --port "${ROCM_RAG_HAYSTACK_SERVER_PORT}" \
    > "${ROCM_RAG_WORKSPACE}/rocm-rag/logs/haystack_retrieval.log" & 
    pids+=($!)

    # Haystack pipeline deployment (waits for server)
    cd "${ROCM_RAG_WORKSPACE}/rocm-rag/rocm_rag/retrieval/haystack/ROCm-RAG-Haystack" || exit
    while ! nc -z localhost "${ROCM_RAG_HAYSTACK_SERVER_PORT}"; do sleep 1; done
    hayhooks pipeline deploy-files . -n ROCm-RAG-Haystack --overwrite

elif [[ "${ROCM_RAG_RETRIEVAL_FRAMEWORK,,}" == "langgraph" ]]; then
    # LangGraph server (waits for Weaviate)
    cd "${ROCM_RAG_WORKSPACE}/rocm-rag/rocm_rag/retrieval/langgraph/ROCm-RAG-Langgraph" || exit
    while ! nc -z localhost "${ROCM_RAG_WEAVIATE_PORT}"; do sleep 1; done
    setsid uvicorn serve:app --host 0.0.0.0 --port "${ROCM_RAG_LANGGRAPH_SERVER_PORT}" --reload \
    > "${ROCM_RAG_WORKSPACE}/rocm-rag/logs/langgraph_retrieval.log" & 
    pids+=($!)

else
    echo "Unsupported retrieval framework: ${ROCM_RAG_RETRIEVAL_FRAMEWORK}. Supported frameworks are 'haystack' and 'langgraph'."
    exit 1
fi

# Optional example LLM
if [[ "${ROCM_RAG_USE_EXAMPLE_LLM,,}" == "true" || "${ROCM_RAG_USE_EXAMPLE_LLM,,}" == 1 ]]; then
    export HIP_VISIBLE_DEVICES=${ROCM_RAG_LLM_GPU_IDS}
    export VLLM_USE_V1=1
    export VLLM_ROCM_USE_AITER=1
    export VLLM_ROCM_USE_AITER_RMSNORM=0
    export VLLM_ROCM_USE_AITER_MHA=0
    export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1

    setsid vllm serve ${ROCM_RAG_LLM_MODEL} \
        --tensor-parallel-size ${ROCM_RAG_LLM_TP} \
        --disable-log-requests \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port ${ROCM_RAG_LLM_API_PORT} \
    > "${ROCM_RAG_WORKSPACE}/rocm-rag/logs/example_llm_retrieval.log" & 
    pids+=($!)
fi
# wait for all background processes
wait -n

cleanup

exit 0
