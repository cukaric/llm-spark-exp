#!/usr/bin/env bash
set -euo pipefail

SERVICES=(
  hermes-llama-qwen.service
  hermes-signal-cli.service
  hermes-gateway-signal.service
)

wait_for_url() {
  local name="$1"
  local url="$2"
  local attempts="${3:-60}"

  printf 'Waiting for %s' "$name"
  for _ in $(seq 1 "$attempts"); do
    if curl --fail --silent --show-error --max-time 2 "$url" >/dev/null 2>&1; then
      printf '\n'
      return 0
    fi
    printf '.'
    sleep 1
  done

  printf '\n%s did not become ready at %s\n' "$name" "$url" >&2
  return 1
}

usage() {
  cat <<'USAGE'
Usage:
  scripts/hermes-signal-stack.sh start
  scripts/hermes-signal-stack.sh stop
  scripts/hermes-signal-stack.sh restart
  scripts/hermes-signal-stack.sh status
  scripts/hermes-signal-stack.sh logs [lines]
  scripts/hermes-signal-stack.sh disable-autostart

Commands:
  start             Start llama.cpp, signal-cli, then Hermes gateway.
  stop              Stop Hermes gateway, signal-cli, then llama.cpp.
  restart           Stop and start the full stack.
  status            Show systemd status for all services.
  logs [lines]      Show recent logs; defaults to 120 lines.
  disable-autostart Ensure services are not enabled at login/boot.
USAGE
}

start_stack() {
  systemctl --user start hermes-llama-qwen.service
  systemctl --user start hermes-signal-cli.service
  wait_for_url "llama.cpp" "http://127.0.0.1:18080/health"
  wait_for_url "signal-cli" "http://127.0.0.1:18081/api/v1/check"
  systemctl --user restart hermes-gateway-signal.service
  systemctl --user --no-pager --full status "${SERVICES[@]}"
}

stop_stack() {
  systemctl --user stop hermes-gateway-signal.service
  systemctl --user stop hermes-signal-cli.service
  systemctl --user stop hermes-llama-qwen.service
}

case "${1:-}" in
  start)
    start_stack
    ;;
  stop)
    stop_stack
    ;;
  restart)
    stop_stack
    start_stack
    ;;
  status)
    systemctl --user --no-pager --full status "${SERVICES[@]}"
    ;;
  logs)
    journalctl --user \
      -u hermes-llama-qwen.service \
      -u hermes-signal-cli.service \
      -u hermes-gateway-signal.service \
      -n "${2:-120}" \
      --no-pager
    ;;
  disable-autostart)
    systemctl --user disable "${SERVICES[@]}"
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    echo "Unknown command: $1" >&2
    usage >&2
    exit 2
    ;;
esac
