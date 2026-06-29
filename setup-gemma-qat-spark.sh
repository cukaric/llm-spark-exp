#!/usr/bin/env bash
# setup-gemma-qat-spark.sh
# Setup Gemma 4 QAT models on DGX Spark (ARM64)
# Supports Ollama, llama.cpp, or vLLM

set -euo pipefail

MODEL_DIR="${HOME}/models/gemma-4-qat"
OLLAMA_MODEL_NAME="gemma-4-e4b-qat"
VLLM_MODEL_NAME="google/gemma-4-E4B-it-qat-w4a16-ct"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_arch() {
    ARCH=$(uname -m)
    if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
        log_info "Detected ARM64 architecture (DGX Spark compatible)."
    else
        log_warn "Detected architecture: $ARCH. DGX Spark is ARM64 — script will continue but verify compatibility."
    fi
}

check_dependencies() {
    log_info "Checking dependencies..."

    # Check for Python
    if ! command -v python3 &> /dev/null; then
        log_error "python3 not found. Please install Python first."
        exit 1
    fi

    # Detect Python path (prefer uv-managed python)
    if command -v uv &> /dev/null; then
        PYTHON_CMD=$(uv python find 2>/dev/null || echo "python3")
    else
        PYTHON_CMD="python3"
    fi
    log_info "Using Python: $PYTHON_CMD"

    # Ensure huggingface-hub is installed
    if ! "$PYTHON_CMD" -c "import huggingface_hub" 2>/dev/null; then
        log_info "Installing huggingface-hub..."
        if command -v uv &> /dev/null; then
            uv pip install huggingface-hub
        elif command -v pip3 &> /dev/null; then
            pip3 install --user huggingface-hub
        else
            log_error "No package manager found. Install uv or pip."
            exit 1
        fi
    fi

    # Ensure git is available for llama.cpp
    if ! command -v git &> /dev/null; then
        log_error "git is required but not installed."
        exit 1
    fi
}

download_model() {
    log_info "Downloading Gemma 4 E4B QAT Q4_0 GGUF model..."
    mkdir -p "$MODEL_DIR"

    "$PYTHON_CMD" -m huggingface_hub.cli download \
        google/gemma-4-E4B-it-qat-q4_0-gguf \
        --include "*.gguf" \
        --local-dir "$MODEL_DIR" \
        --local-dir-use-symlinks False

    log_info "Model downloaded to $MODEL_DIR"
}

setup_ollama() {
    log_info "Setting up Ollama..."

    if ! command -v ollama &> /dev/null; then
        log_info "Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
    else
        log_info "Ollama already installed."
    fi

    # Find the downloaded GGUF file
    GGUF_FILE=$(find "$MODEL_DIR" -name "*.gguf" | head -n 1)
    if [[ -z "$GGUF_FILE" ]]; then
        log_error "No GGUF file found in $MODEL_DIR"
        exit 1
    fi

    log_info "Using GGUF file: $GGUF_FILE"

    # Create Modelfile
    cat > /tmp/Modelfile << EOF
FROM $GGUF_FILE

TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

"""

PARAMETER stop <|eot_id|>
PARAMETER stop <|end_of_text|>
PARAMETER temperature 1.0
PARAMETER top_p 0.95
PARAMETER top_k 64
EOF

    log_info "Creating Ollama model '$OLLAMA_MODEL_NAME'..."
    ollama create "$OLLAMA_MODEL_NAME" -f /tmp/Modelfile

    log_info "Ollama setup complete!"
    log_info "Run with: ollama run $OLLAMA_MODEL_NAME"
}

setup_llamacpp() {
    log_info "Setting up llama.cpp..."

    LLAMA_DIR="${HOME}/llama.cpp"

    if [[ ! -d "$LLAMA_DIR" ]]; then
        log_info "Cloning llama.cpp..."
        git clone https://github.com/ggerganov/llama.git "$LLAMA_DIR" || \
        git clone https://github.com/ggerganov/llama.cpp.git "$LLAMA_DIR"
    else
        log_info "llama.cpp already cloned at $LLAMA_DIR"
    fi

    cd "$LLAMA_DIR"

    # Detect CUDA
    if command -v nvcc &> /dev/null || [[ -d /usr/local/cuda ]]; then
        log_info "CUDA detected — building llama.cpp with GPU support..."
        cmake -B build -DGGML_CUDA=ON
    else
        log_warn "CUDA not detected — building CPU-only version..."
        cmake -B build
    fi

    log_info "Building llama.cpp (this may take a few minutes)..."
    cmake --build build --config Release -j$(nproc)

    # Find the downloaded GGUF file
    GGUF_FILE=$(find "$MODEL_DIR" -name "*.gguf" | head -n 1)
    if [[ -z "$GGUF_FILE" ]]; then
        log_error "No GGUF file found in $MODEL_DIR"
        exit 1
    fi

    log_info "llama.cpp setup complete!"
    log_info "GGUF file: $GGUF_FILE"

    # Create convenience runner script
    cat > /tmp/run-gemma.sh << EOF
#!/usr/bin/env bash
# Quick runner for Gemma 4 E4B QAT via llama.cpp

MODEL="$GGUF_FILE"
BUILD_DIR="$LLAMA_DIR/build"

if [[ -f "\$BUILD_DIR/bin/llama-cli" ]]; then
    "\$BUILD_DIR/bin/llama-cli" \\
        -m "\$MODEL" \\
        -p "\${1:-You are a helpful assistant. Write a short joke about saving RAM.}" \\
        -n 512 \\
        --temp 1.0 \\
        --top-p 0.95 \\
        --top-k 64
elif [[ -f "\$BUILD_DIR/bin/llama" ]]; then
    "\$BUILD_DIR/bin/llama" \\
        -m "\$MODEL" \\
        -p "\${1:-You are a helpful assistant. Write a short joke about saving RAM.}" \\
        -n 512 \\
        --temp 1.0 \\
        --top-p 0.95 \\
        --top-k 64
else
    echo "llama-cli not found in \$BUILD_DIR/bin"
    exit 1
fi
EOF
    chmod +x /tmp/run-gemma.sh
    mv /tmp/run-gemma.sh "$MODEL_DIR/run-gemma.sh"

    log_info "Run with: $MODEL_DIR/run-gemma.sh"
    log_info "Or run server: $LLAMA_DIR/build/bin/llama-server -m $GGUF_FILE --port 8080"
}

setup_vllm() {
    log_info "Setting up vLLM with compressed-tensors QAT (w4a16)..."

    # Detect CUDA
    if ! command -v nvcc &> /dev/null && [[ ! -d /usr/local/cuda ]]; then
        log_warn "CUDA not detected. vLLM requires GPU support for compressed-tensors inference."
        log_warn "If you have a GPU, ensure CUDA toolkit is installed."
    fi

    # Install vLLM
    if ! "$PYTHON_CMD" -c "import vllm" 2>/dev/null; then
        log_info "Installing vLLM..."
        if command -v uv &> /dev/null; then
            uv pip install vllm
        else
            pip3 install --user vllm
        fi
    else
        log_info "vLLM already installed."
    fi

    # Install compressed-tensors support if not present
    if ! "$PYTHON_CMD" -c "import compressed_tensors" 2>/dev/null; then
        log_info "Installing compressed-tensors..."
        if command -v uv &> /dev/null; then
            uv pip install compressed-tensors
        else
            pip3 install --user compressed-tensors
        fi
    fi

    # Download the w4a16 model (safetensors format)
    log_info "Downloading Gemma 4 E4B QAT w4a16 compressed-tensors model..."
    mkdir -p "$MODEL_DIR"

    "$PYTHON_CMD" -m huggingface_hub.cli download \
        "$VLLM_MODEL_NAME" \
        --local-dir "$MODEL_DIR/vllm" \
        --local-dir-use-symlinks False

    log_info "Model downloaded to $MODEL_DIR/vllm"

    # Create convenience server script
    cat > "$MODEL_DIR/run-vllm-server.sh" << 'EOF'
#!/usr/bin/env bash
# vLLM server for Gemma 4 E4B QAT w4a16

MODEL_DIR="$HOME/models/gemma-4-qat/vllm"
PORT="${1:-8080}"

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --port "$PORT" \
    --dtype auto \
    --tensor-parallel-size 1 \
    --quantization "compressed-tensors"
EOF
    chmod +x "$MODEL_DIR/run-vllm-server.sh"

    # Create convenience client script
    cat > "$MODEL_DIR/run-vllm-chat.sh" << 'EOF'
#!/usr/bin/env bash
# Quick chat with vLLM OpenAI-compatible API

PORT="${1:-8080}"
PROMPT="${2:-You are a helpful assistant. Write a short joke about saving RAM.}"

curl http://localhost:"$PORT"/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"gemma-4-e4b-qat\",
    \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}],
    \"temperature\": 1.0,
    \"top_p\": 0.95,
    \"max_tokens\": 512
  }"
EOF
    chmod +x "$MODEL_DIR/run-vllm-chat.sh"

    log_info "vLLM setup complete!"
    log_info "Start server: $MODEL_DIR/run-vllm-server.sh [port]"
    log_info "Chat:       $MODEL_DIR/run-vllm-chat.sh [port] 'Your prompt'"
    log_info ""
    log_info "Example:"
    log_info "  $MODEL_DIR/run-vllm-server.sh 8080"
    log_info "  $MODEL_DIR/run-vllm-chat.sh 8080 'What is 2+2?'"
}

print_summary() {
    echo ""
    echo "========================================"
    echo "  Gemma 4 QAT Setup Complete!"
    echo "========================================"
    echo ""
    if [[ "$CHOICE" == "3" ]]; then
        echo "Model: Gemma 4 E4B IT QAT w4a16 (compressed-tensors)"
        echo "Location: $MODEL_DIR/vllm"
        echo ""
        echo "Tool: vLLM"
        echo "Server: $MODEL_DIR/run-vllm-server.sh [port]"
        echo "Chat:   $MODEL_DIR/run-vllm-chat.sh [port] 'Your prompt'"
        echo ""
        echo "API endpoint: http://localhost:8080/v1/chat/completions"
    else
        echo "Model: Gemma 4 E4B IT QAT Q4_0 GGUF"
        echo "Location: $MODEL_DIR"
        echo ""
        if [[ "$CHOICE" == "1" ]]; then
            echo "Tool: Ollama"
            echo "Run:  ollama run $OLLAMA_MODEL_NAME"
        else
            echo "Tool: llama.cpp"
            echo "Run:  $MODEL_DIR/run-gemma.sh"
            echo "API:  llama-server -m <gguf> --port 8080"
        fi
    fi
    echo ""
    echo "Recommended sampling params:"
    echo "  temperature=1.0, top_p=0.95, top_k=64"
    echo ""
    echo "========================================"
}

# Main
main() {
    echo "========================================"
    echo "  Gemma 4 QAT Setup for DGX Spark"
    echo "========================================"
    echo ""

    check_arch
    check_dependencies

    echo ""
    echo "Choose your runtime:"
    echo "  1) Ollama (easiest, chat interface, GGUF)"
    echo "  2) llama.cpp (max control, server API, GGUF)"
    echo "  3) vLLM (fastest serving, OpenAI-compatible API, w4a16 compressed-tensors)"
    echo ""
    read -p "Enter choice [1-3]: " CHOICE

    if [[ "$CHOICE" == "3" ]]; then
        setup_vllm
    else
        download_model
        if [[ "$CHOICE" == "1" ]]; then
            setup_ollama
        elif [[ "$CHOICE" == "2" ]]; then
            setup_llamacpp
        else
            log_error "Invalid choice. Run script again and select 1, 2, or 3."
            exit 1
        fi
    fi

    print_summary
}

main "$@"
