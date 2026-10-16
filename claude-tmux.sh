#!/bin/bash
# Claude Code with tmux integration
# Usage: ./claude-tmux.sh [--attach-tmux] [other claude args]
#
# Examples:
#   ./claude-tmux.sh --resume          # Show session picker (passes --resume to Claude)
#   ./claude-tmux.sh --attach-tmux     # Attach to existing tmux session, or create new one
#   ./claude-tmux.sh --attach-tmux --session-id <id>  # Resume specific session in tmux

ATTACH_TMUX=false
SESSION_NAME="claude-session"
ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --attach-tmux)
      ATTACH_TMUX=true
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

# If not using tmux, just run Claude directly (passes all args through)
if [ "$ATTACH_TMUX" = false ]; then
  cd "$(pwd)"
  exec claude "${ARGS[@]}"
fi

# Otherwise, manage tmux session
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  # Session exists - attach to it
  echo "Attaching to existing Claude session: $SESSION_NAME"
  tmux attach-session -t "$SESSION_NAME"
else
  # No session exists - create one
  echo "Creating new Claude session in tmux: $SESSION_NAME"
  tmux new-session -d -s "$SESSION_NAME" "cd '$(pwd)' && claude ${ARGS[@]}"
  tmux attach-session -t "$SESSION_NAME"
fi
