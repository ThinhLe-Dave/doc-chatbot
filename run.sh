#!/bin/bash

# Doc Chatbot Environment Setup & Run Script

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

check_python() {
    if ! command -v python3 &> /dev/null; then
        echo "❌ Error: python3 is not installed. Please install Python 3.10+ and try again."
        exit 1
    fi
}

setup_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        echo "📦 Creating virtual environment in .venv..."
        python3 -m venv "$VENV_DIR"
    fi
}

install_dependencies() {
    echo "🔌 Activating virtual environment..."
    source "$VENV_DIR/bin/activate"

    echo "📥 Installing dependencies..."
    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/requirements.txt"
}

init_directories() {
    mkdir -p "$PROJECT_DIR/database"
}

print_success() {
    echo "------------------------------------------------"
    echo "✅ Setup finished successfully!"
    echo "------------------------------------------------"
    echo "💡 To activate the environment manually: source .venv/bin/activate"
    echo "🔍 To search via CLI:                  ./run.sh search"
    echo "📄 To scan a PDF:                      ./run.sh pdf-scan <path_to_pdf>"
    echo "To scrape a website:                   ./run.sh scrape <url>"
    echo "🌐 To run the web UI:                  ./run.sh serve --help"
    echo "🧪 To run tests:                       ./run.sh test"
    echo "🔨 To compile/check all .py files:     ./run.sh compile"
    echo "🗑️  To clear the database (testing):   ./run.sh clear-db --force"
    echo "------------------------------------------------"
}

run_app() {
    source "$VENV_DIR/bin/activate"
    python3 "$PROJECT_DIR/app.py" "$@"
}

run_tests() {
    source "$VENV_DIR/bin/activate"
    export PYTHONPATH="$PROJECT_DIR"
    python3 -m unittest discover -s "$PROJECT_DIR/test" -p "test_*.py" -v
}

run_compile() {
    source "$VENV_DIR/bin/activate"
    echo "🔨 Compiling all .py files..."
    failed=0
    while IFS= read -r file; do
        if ! python3 -m py_compile "$file" 2>/dev/null; then
            echo "❌ FAIL: $file"
            failed=$((failed + 1))
        fi
    done < <(find "$PROJECT_DIR" -type f -name "*.py" ! -path "*/.venv/*" ! -path "*/__pycache__/*")
    if [ "$failed" -eq 0 ]; then
        echo "✅ All .py files compiled successfully."
    else
        echo "❌ $failed file(s) failed to compile."
        exit 1
    fi
}

run_server() {
    source "$VENV_DIR/bin/activate"
    PORT="${1:-8000}"
    HOST="${2:-127.0.0.1}"
    echo "🚀 Starting Doc Chatbot web server at http://$HOST:$PORT"
    "$VENV_DIR/bin/python" -m uvicorn web_frontend.fastapi_app:app --host "$HOST" --port "$PORT" --reload
}

init() {
    echo "------------------------------------------------"
    echo "🚀 Setting up Doc Chatbot environment..."
    echo "------------------------------------------------"

    check_python
    setup_venv
    install_dependencies
    init_directories
    print_success
}

main() {
    if [[ "$1" == "setup" ]]; then
        init
        exit 0
    fi

    if [[ "$1" == "serve" ]]; then
        if [ ! -d "$VENV_DIR" ]; then
            init
        fi
        shift
        run_server "$@"
        exit 0
    fi

    if [[ "$1" == "pdf-scan" ]]; then
        shift
        run_app pdf-scan "$@"
        exit 0
    fi

    if [[ "$1" == "scrape" ]]; then
        shift
        run_app scrape "$@"
        exit 0
    fi

    if [[ "$1" == "test" ]]; then
        shift
        run_tests "$@"
        exit 0
    fi

    if [[ "$1" == "compile" ]]; then
        run_compile
        exit 0
    fi

    if [[ "$1" == "build-chunks" ]]; then
        shift
        run_app build_chunks "$@"
        exit 0
    fi

    if [[ "$1" == "clear-db" ]]; then
        shift
        run_app clear-db "$@"
        exit 0
    fi

    if [ ! -d "$VENV_DIR" ]; then
        init
    fi

    run_app "$@"
}

main "$@"