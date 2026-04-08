"""
Chat Completions-based thread manager for the QA pipeline.

Uses the standard OpenAI Chat Completions API (compatible with any
OpenAI-compatible endpoint via API_BASE_URL / MODEL_NAME / HF_TOKEN).
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai import OpenAI
from constants import OPENAI_API_KEY, API_BASE_URL, DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS


class SimpleAssistantThreadManager:
    """Manages a conversation thread using the Chat Completions API."""

    def __init__(self, assistant_id: str = ""):
        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)
        self.thread_id = None
        self.messages: list[dict] = [
            {
                "role": "system",
                "content": (
                    "You are an expert Quality Assurance engineer. "
                    "You analyze software documentation (FRDs, Technical Design docs, API specs) "
                    "and generate comprehensive, structured test scenarios and detailed test cases. "
                    "Always respond with valid JSON when asked to."
                ),
            }
        ]

    def start_thread(self):
        """Start a fresh conversation thread."""
        if self.thread_id is None:
            import uuid
            self.thread_id = str(uuid.uuid4())
            self.messages = self.messages[:1]  # keep only system message
        return self.thread_id

    def extract_pdf_content(self, file_paths):
        """Extract text content from files (plain-text or PDF)."""
        if not file_paths:
            return {"status": "no_files", "content": "", "files_processed": 0}

        all_content = []
        for file_path in file_paths:
            if not os.path.exists(file_path):
                continue
            filename = os.path.basename(file_path)
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read().strip()
                if text:
                    all_content.append(f"\n=== DOCUMENT: {filename} ===\n{text}")
            except Exception:
                try:
                    import PyPDF2
                    with open(file_path, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        pages = []
                        for i, page in enumerate(reader.pages):
                            t = page.extract_text()
                            if t and t.strip():
                                pages.append(f"Page {i+1}: {t.strip()}")
                        if pages:
                            all_content.append(
                                f"\n=== DOCUMENT: {filename} ===\n" + "\n\n".join(pages)
                            )
                except Exception:
                    pass

        combined = "\n\n".join(all_content)
        return {
            "status": "completed" if all_content else "no_content",
            "content": combined,
            "files_processed": len(all_content),
        }

    def invoke_assistant(self, prompt: str) -> str:
        """Send a prompt and return the assistant's response via Chat Completions."""
        self.messages.append({"role": "user", "content": prompt})

        max_ctx = min(DEFAULT_MAX_TOKENS, 4096)

        try:
            response = self.client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=self.messages,
                temperature=DEFAULT_TEMPERATURE,
                max_tokens=max_ctx,
            )
            reply = response.choices[0].message.content or ""
        except Exception as e:
            reply = f"LLM error: {e}"

        self.messages.append({"role": "assistant", "content": reply})

        if len(self.messages) > 20:
            self.messages = self.messages[:1] + self.messages[-18:]

        return reply
