#!/bin/bash

# Ray Multi-node Setup Script
# This script helps set up a stable Ray cluster for multi-node training

set -e

# Configuration
HEAD_NODE_IP="172.22.8.5"
RAY_PORT="6379"
DASHBOARD_PORT="8265"

# Environment setup
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_DEDUP_LOGS=0
export RAY_IGNORE_UNHANDLED_ERRORS=1
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DISABLE_STRICT_VERSION_CHECK=1
export RAY_CLIENT_FORCE_CLEANUP=1

# Additional stability settings
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=INFO
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

function show_usage() {
    echo "Usage: $0 [head|worker|status|stop|restart]"
    echo "  head    - Start Ray head node"
    echo "  worker  - Start Ray worker node"
    echo "  status  - Show cluster status"
    echo "  stop    - Stop Ray cluster"
    echo "  restart - Restart Ray cluster"
    exit 1
}

function start_head() {
    echo "Starting Ray head node..."
    
    # Stop any existing Ray processes
    ray stop --force || true
    sleep 3
    
    # Start head node with optimized settings
    ray start \
        --head \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=$DASHBOARD_PORT \
        --port=$RAY_PORT \
        --num-cpus=384 \
        --num-gpus=8 \
        --object-store-memory=100000000000 \
        --plasma-directory=/tmp \
        --temp-dir=/tmp/ray \
        --disable-usage-stats \
        --verbose
    
    echo "Head node started successfully!"
    echo "Dashboard available at: http://$HEAD_NODE_IP:$DASHBOARD_PORT"
    echo "To connect worker nodes, run:"
    echo "  ray start --address='$HEAD_NODE_IP:$RAY_PORT'"
}

function start_worker() {
    echo "Starting Ray worker node..."
    
    # Stop any existing Ray processes
    ray stop --force || true
    sleep 3
    
    # Start worker node
    ray start \
        --address="$HEAD_NODE_IP:$RAY_PORT" \
        --num-cpus=384 \
        --num-gpus=8 \
        --object-store-memory=100000000000 \
        --plasma-directory=/tmp \
        --temp-dir=/tmp/ray \
        --disable-usage-stats \
        --verbose
    
    echo "Worker node started successfully!"
}

function show_status() {
    echo "Ray cluster status:"
    ray status --address="$HEAD_NODE_IP:$RAY_PORT" || {
        echo "Failed to get cluster status. Is the cluster running?"
        exit 1
    }
}

function stop_cluster() {
    echo "Stopping Ray cluster..."
    ray stop --force
    echo "Ray cluster stopped."
}

function restart_cluster() {
    echo "Restarting Ray cluster..."
    stop_cluster
    sleep 5
    start_head
}

function check_prerequisites() {
    # Check if Ray is installed
    if ! command -v ray &> /dev/null; then
        echo "Error: Ray is not installed or not in PATH"
        exit 1
    fi
    
    # Check if we're in the right environment
    if [[ "$CONDA_DEFAULT_ENV" != "forecast" ]]; then
        echo "Warning: Not in 'forecast' conda environment. Consider activating it."
    fi
    
    # Check GPU availability
    if ! command -v nvidia-smi &> /dev/null; then
        echo "Warning: nvidia-smi not found. GPU support may not be available."
    else
        echo "Available GPUs:"
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits
    fi
}

# Main script logic
if [ $# -eq 0 ]; then
    show_usage
fi

check_prerequisites

case "$1" in
    "head")
        start_head
        ;;
    "worker")
        start_worker
        ;;
    "status")
        show_status
        ;;
    "stop")
        stop_cluster
        ;;
    "restart")
        restart_cluster
        ;;
    *)
        show_usage
        ;;
esac 