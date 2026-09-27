from settings import build_chat_model, build_embeddings, build_chroma_client

reply = build_chat_model().invoke("Just reply: CHAT READY")
print("Chat model reply:", reply.content)

vector = build_embeddings().embed_query("solar panel warranty")
print("Embedding dimensions:", len(vector))

try:
    heartbeat = build_chroma_client().heartbeat()
    print("ChromaDB heartbeat:", heartbeat)
except Exception as e:
    raise SystemExit(f"ChromaDB heartbeat failed: {e}") from e