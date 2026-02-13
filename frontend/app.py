import os

import httpx
import streamlit as st


def api_base() -> str:
    return os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")


def _client() -> httpx.Client:
    return httpx.Client(base_url=api_base(), timeout=120.0)


st.set_page_config(page_title="AI Knowledge Assistant", page_icon="📚", layout="wide")
st.title("AI Knowledge Assistant")
st.caption("Upload documents, then ask questions grounded in your files (RAG).")

with st.sidebar:
    st.subheader("API")
    st.markdown(f"**Base URL:** `{api_base()}`")
    st.caption("Set env `API_BASE_URL` (e.g. `http://api:8000` in Docker).")
    if st.button("Check health"):
        try:
            with _client() as c:
                r = c.get("/health")
                r.raise_for_status()
                data = r.json()
            st.success(
                f"Status: {data.get('status')} — OpenAI configured: {data.get('openai_configured')}"
            )
        except Exception as e:
            st.error(f"Health check failed: {e}")

tab_upload, tab_chat = st.tabs(["Upload documents", "Ask questions"])

with tab_upload:
    st.markdown("Supported: **PDF**, **TXT**, **Markdown**.")
    uploaded = st.file_uploader(
        "Choose files", type=["pdf", "txt", "md"], accept_multiple_files=True
    )
    if uploaded and st.button("Upload to knowledge base", type="primary"):
        progress = st.progress(0.0, text="Starting…")
        ok, bad = 0, []
        for i, f in enumerate(uploaded):
            try:
                with httpx.Client(base_url=api_base(), timeout=300.0) as client:
                    files = {
                        "file": (
                            f.name,
                            f.getvalue(),
                            f.type or "application/octet-stream",
                        )
                    }
                    resp = client.post("/documents", files=files)
                    resp.raise_for_status()
                ok += 1
            except Exception as ex:
                bad.append((f.name, str(ex)))
            progress.progress((i + 1) / len(uploaded), text=f"Uploaded {i + 1}/{len(uploaded)}")
        progress.empty()
        if ok:
            st.success(f"Uploaded {ok} file(s) successfully.")
        for name, err in bad:
            st.error(f"{name}: {err}")

    st.divider()
    st.subheader("Indexed uploads (on disk)")
    try:
        with _client() as client:
            r = client.get("/documents")
            r.raise_for_status()
            docs = r.json()
        if not docs:
            st.info("No documents listed yet. Upload files above.")
        else:
            for d in docs:
                st.write(f"**{d.get('filename')}** — `{d.get('doc_id')}`")
    except Exception as e:
        st.warning(f"Could not list documents: {e}")

with tab_chat:
    docs_count = 0
    try:
        with _client() as client:
            docs_resp = client.get("/documents")
            docs_resp.raise_for_status()
            docs_count = len(docs_resp.json() or [])
    except Exception:
        docs_count = 0

    if docs_count == 0:
        st.warning(
            "No documents are indexed yet. Upload at least one file in 'Upload documents' "
            "before asking questions."
        )

    q = st.text_input("Your question", placeholder="What do the documents say about …?")
    k = st.slider("Retrieval top-k (optional)", min_value=1, max_value=20, value=8)
    show_context = st.checkbox("Show retrieved context", value=False)
    if st.button("Ask", type="primary", disabled=(docs_count == 0)) and q.strip():
        try:
            with httpx.Client(base_url=api_base(), timeout=120.0) as client:
                r = client.post("/query", json={"question": q.strip(), "k": k})
                r.raise_for_status()
                data = r.json()
            st.markdown("### Answer")
            st.write(data.get("answer", ""))
            if data.get("context_empty"):
                st.info(
                    "No relevant chunks passed the relevance threshold. Try lowering "
                    "`MIN_RELEVANCE_SCORE` or uploading more targeted documents."
                )
            srcs = data.get("sources") or []
            if srcs:
                with st.expander(f"Sources ({len(srcs)} chunks)", expanded=True):
                    for s in srcs:
                        chunk_index = s.get("chunk_index")
                        chunk_label = (
                            f"chunk {chunk_index}"
                            if isinstance(chunk_index, int)
                            else "chunk ?"
                        )
                        st.markdown(
                            f"**{s.get('source')}** ({chunk_label}) — doc `{s.get('doc_id')}` — "
                            f"score `{s.get('relevance_score', 0):.3f}`"
                        )
                        st.caption(s.get("snippet", ""))
                if show_context:
                    st.markdown("### Retrieved context")
                    for i, s in enumerate(srcs, start=1):
                        chunk_index = s.get("chunk_index")
                        chunk_label = (
                            f"chunk {chunk_index}"
                            if isinstance(chunk_index, int)
                            else "chunk ?"
                        )
                        with st.expander(f"{i}. {s.get('source')} ({chunk_label})"):
                            st.write(s.get("chunk_text", ""))
            elif not data.get("context_empty"):
                st.info("No relevant information found.")
        except Exception as e:
            st.error(f"Query failed: {e}")
