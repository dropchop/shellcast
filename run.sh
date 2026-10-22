#!/usr/bin/env bash
# ShellCast web UI. Usage: ./run.sh [port]   (default 5070)
# Serves on localhost only. The UI has no login: HOST=0.0.0.0 ./run.sh opens it to everyone on your network.
set -e
cd "$(dirname "$0")"
python3 -c "import trimesh, manifold3d, igl, skimage, fastapi" 2>/dev/null || pip install --user -e ".[dev]"
PORT="${1:-${PORT:-5070}}"
HOST="${HOST:-127.0.0.1}"
echo "ShellCast → http://${HOST}:${PORT}"
exec python3 -m shellcast serve --host "$HOST" --port "$PORT"
