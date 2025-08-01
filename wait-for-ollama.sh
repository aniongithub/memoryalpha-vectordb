# wait-for-ollama.sh
#!/bin/bash
echo "🖖 Waiting for Ollama at $OLLAMA_URL..."

until curl -s "$OLLAMA_URL/api/tags" > /dev/null; do
  sleep 1
done

echo "✅ Ollama is ready."
exec "$@"
