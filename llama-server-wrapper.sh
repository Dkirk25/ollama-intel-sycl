#!/bin/bash
# Ollama stops llama-server with SIGKILL. On Intel xe GPUs, killing it right
# after GPU work can hang the next model load (copy engine reset). Run the real
# server as a child that gets a graceful SIGTERM when this wrapper dies.
real=/usr/lib/ollama/llama-server.real

setpriv --pdeathsig TERM "$real" "$@" &
child=$!
trap 'kill -TERM "$child" 2>/dev/null' TERM INT

wait "$child"
rc=$?
# wait returns early when a trapped signal arrives; wait again for the real exit
while kill -0 "$child" 2>/dev/null; do
    wait "$child"
    rc=$?
done
exit $rc
