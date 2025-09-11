#!/bin/bash

SESSION="rocm_rag_retrieval"
# Create tmux session
tmux new-session -d -s $SESSION -n open-webui "cd ${ROCM_RAG_WORKSPACE}/rocm-rag/external/open-webui/backend && source venv/bin/activate && sh dev.sh 2>&1 | tee ${ROCM_RAG_WORKSPACE}/rocm-rag/logs/open-webui_retrieval.log"
tmux new-window -t $SESSION -n weaviate "cd ${ROCM_RAG_WORKSPACE}/rocm-rag/external/weaviate/cmd/weaviate-server && QUERY_DEFAULTS_LIMIT=20 AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true DEFAULT_VECTORIZER_MODULE=none ENABLE_MODULES=none PERSISTENCE_DATA_PATH="${ROCM_RAG_WORKSPACE}/rocm-rag/data/haystack/weaviate" DEFAULT_PORT=${ROCM_RAG_WEAVIATE_PORT} TLS_DISABLED=true go run main.go --host 0.0.0.0 --port ${ROCM_RAG_WEAVIATE_PORT} --scheme http 2>&1 | tee ${ROCM_RAG_WORKSPACE}/rocm-rag/logs/weaviate_retrieval.log"
tmux new-window -t $SESSION -n embedder-vllm "HIP_VISIBLE_DEVICES=${ROCM_RAG_EMBEDDER_GPU_IDS} vllm serve ${ROCM_RAG_EMBEDDER_MODEL} --task embed --port ${ROCM_RAG_EMBEDDER_API_PORT} --tensor-parallel-size ${ROCM_RAG_EMBEDDER_TP} --trust-remote-code 2>&1 | tee ${ROCM_RAG_WORKSPACE}/rocm-rag/logs/embedder_retrieval.log"

if [[ "${ROCM_RAG_RETRIEVAL_FRAMEWORK,,}" == "haystack" ]]; then
    tmux new-window -t $SESSION -n ROCm-RAG-Haystack "cd ${ROCM_RAG_WORKSPACE}/rocm-rag/rocm_rag/retrieval/haystack && hayhooks run --host 0.0.0.0 --port ${ROCM_RAG_HAYSTACK_SERVER_PORT} 2>&1 | tee ${ROCM_RAG_WORKSPACE}/rocm-rag/logs/haystack_retrieval.log"
    tmux new-window -t $SESSION -n haystack-retrieval-pipeline "cd ${ROCM_RAG_WORKSPACE}/rocm-rag/rocm_rag/retrieval/haystack/ROCm-RAG-Haystack && while ! nc -z localhost ${ROCM_RAG_HAYSTACK_SERVER_PORT}; do sleep 1; done && hayhooks pipeline deploy-files . -n ROCm-RAG-Haystack --overwrite"
elif [[ "${ROCM_RAG_RETRIEVAL_FRAMEWORK,,}" == "langgraph" ]]; then
    tmux new-window -t $SESSION -n ROCm-RAG-LangGraph "cd ${ROCM_RAG_WORKSPACE}/rocm-rag/rocm_rag/retrieval/langgraph/ROCm-RAG-Langgraph &&  while ! nc -z localhost ${ROCM_RAG_WEAVIATE_PORT}; do sleep 1; done && uvicorn serve:app --host 0.0.0.0 --port ${ROCM_RAG_LANGGRAPH_SERVER_PORT} --reload 2>&1 | tee ${ROCM_RAG_WORKSPACE}/rocm-rag/logs/langgraph_retrieval.log"
else
    echo "Unsupported retrieval framework: ${ROCM_RAG_RETRIEVAL_FRAMEWORK}. Supported frameworks are 'haystack' and 'langgraph'."
    exit 1
fi

if [[ "${ROCM_RAG_USE_EXAMPLE_LLM,,}" == "true" || "${ROCM_RAG_USE_EXAMPLE_LLM,,}" == 1 ]]; then
    tmux new-window -t $SESSION -n example-llm "HIP_VISIBLE_DEVICES=${ROCM_RAG_LLM_GPU_IDS} VLLM_USE_V1=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_RMSNORM=0 VLLM_ROCM_USE_AITER_MHA=0 VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 vllm serve ${ROCM_RAG_LLM_MODEL} --tensor-parallel-size ${ROCM_RAG_LLM_TP} --disable-log-requests --trust-remote-code --host 0.0.0.0 --port ${ROCM_RAG_LLM_API_PORT} 2>&1 | tee ${ROCM_RAG_WORKSPACE}/rocm-rag/logs/example_llm_retrieval.log"
fi


