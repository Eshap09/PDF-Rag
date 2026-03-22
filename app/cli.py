import os
import sys
import time
import webbrowser
import threading
from pathlib import Path


RAG_HOME   = Path.home() / ".rag"
CHROMA_DIR = RAG_HOME / "chromadb"
PORT       = 8000
URL        = f"http://localhost:{PORT}"


# ── COMMANDS ──────────────────────────────────────────────────────────────────

def start():
    _banner()

    # create chroma directory
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    _ok("chromadb ready", str(CHROMA_DIR))

    # set env vars before importing anything from app/
    os.environ["CHROMA_PERSIST_DIR"] = str(CHROMA_DIR)

    # load GROQ_API_KEY
    _load_key()

    # warm up embedding model
    _step("loading embedding model...")
    from app.rag.embedder import get_embedder
    get_embedder()
    _ok("model ready", "all-MiniLM-L6-v2")

    # open browser after delay
    threading.Thread(target=_open_browser, daemon=True).start()

    _ok("server starting", URL)
    print()

    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=PORT,
        log_level="warning",
    )


def config():
    _banner()

    RAG_HOME.mkdir(parents=True, exist_ok=True)
    env_file = RAG_HOME / ".env"

    # show current state
    existing_key = _read_key()
    if existing_key:
        masked = existing_key[:8] + "..." + existing_key[-4:]
        print(f"  Current GROQ_API_KEY: {masked}")
        print()
        overwrite = input("  Overwrite? (y/N): ").strip().lower()
        if overwrite != "y":
            print("  No changes made.")
            return

    print()
    print("  Get your free API key at: https://console.groq.com")
    print()
    key = input("  Paste your GROQ_API_KEY: ").strip()

    if not key:
        _warn("No key entered. Aborted.")
        return

    if not key.startswith("gsk_"):
        _warn("That doesn't look like a Groq key (should start with gsk_)")
        confirm = input("  Save anyway? (y/N): ").strip().lower()
        if confirm != "y":
            return

    env_file.write_text(f"GROQ_API_KEY={key}\n")
    _ok("saved", str(env_file))
    print()
    print("  Run 'rag start' to launch the app.")
    print()


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _load_key():
    """Load GROQ_API_KEY from env or ~/.rag/.env. Prompt if missing."""
    if os.environ.get("GROQ_API_KEY"):
        return

    key = _read_key()
    if key:
        os.environ["GROQ_API_KEY"] = key
        return

    # not found anywhere — prompt
    print()
    _warn("GROQ_API_KEY not set.")
    print()
    run_config = input("  Set it now? (Y/n): ").strip().lower()
    if run_config != "n":
        config()
        key = _read_key()
        if key:
            os.environ["GROQ_API_KEY"] = key
    print()


def _read_key() -> str:
    """Read GROQ_API_KEY from ~/.rag/.env. Returns empty string if not found."""
    env_file = RAG_HOME / ".env"
    if not env_file.exists():
        return ""
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line.startswith("GROQ_API_KEY="):
            return line.split("=", 1)[1].strip()
    return ""


def _open_browser():
    time.sleep(2)
    webbrowser.open(URL)


def _banner():
    print()
    print("  \033[1mrag\033[0m v1.0.0")
    print()


def _step(msg):
    print(f"  \033[2m⋯\033[0m  {msg}", flush=True)


def _ok(label, value):
    print(f"  \033[32m✓\033[0m  \033[1m{label:<18}\033[0m {value}")


def _warn(msg):
    print(f"  \033[33m⚠\033[0m  {msg}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    if not args or args[0] == "start":
        start()
    elif args[0] == "config":
        config()
    elif args[0] == "version":
        print("rag 1.0.0")
    elif args[0] == "help":
        print("Usage: rag <command>")
        print()
        print("Commands:")
        print("  start      start the server and open browser")
        print("  config     set your Groq API key")
        print("  version    show version")
        print("  help       show this help")
    else:
        print(f"Unknown command: {args[0]}")
        print("Run 'rag help' for usage.")
        sys.exit(1)


if __name__ == "__main__":
    main()