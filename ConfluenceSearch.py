#!/usr/bin/env python3
"""
confluence_search_gui_faiss.py

PyQt5 GUI to fetch Confluence pages, embed them locally with Sentence‑Transformers,
build a FAISS index (cosine similarity via inner‑product after L2 normalisation),
run semantic search, and persist / reload the index.

Requirements (all wheels available for Python ≥3.11, no C/C++ compile on Windows):
    pip install sentence-transformers faiss-cpu requests beautifulsoup4 pyqt5

How to run:
    python confluence_search_gui_faiss.py
"""

import os
import pickle
import sys
from typing import List, Tuple

import requests
from bs4 import BeautifulSoup
from requests.auth import HTTPBasicAuth
from sentence_transformers import SentenceTransformer
import faiss
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# ───────────────────────────────────────── Configuration ─────────────────────────────────────────
INDEX_PATH = "confluence_faiss.index"
MAP_PATH = "id_to_page.pkl"
MODEL_DEFAULT = "all-MiniLM-L6-v2"

# ───────────────────────────────────────── Confluence fetch ────────────────────────────────────────

def fetch_confluence_pages(base_url: str, space_key: str, auth: HTTPBasicAuth) -> List[Tuple[str, str, str]]:
    """Return list of (page_id, title, plain_text)."""
    pages: List[Tuple[str, str, str]] = []
    url = f"{base_url}/rest/api/content"
    start = 0
    limit = 50
    while True:
        params = {
            "spaceKey": space_key,
            "limit": limit,
            "start": start,
            "expand": "body.storage",
        }
        resp = requests.get(url, params=params, auth=auth, timeout=30)
        resp.raise_for_status()
        obj = resp.json()
        results = obj.get("results", [])
        if not results:
            break
        for page in results:
            pid = page["id"]
            title = page["title"]
            html = page["body"]["storage"]["value"]
            text = BeautifulSoup(html, "html.parser").get_text("\n").strip()
            pages.append((pid, title, text))
        start += limit
    return pages

# ───────────────────────────────────────── GUI App ────────────────────────────────────────────────

class ConfluenceSearchApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Confluence Semantic Search (FAISS)")
        self.resize(850, 600)

        # Runtime state
        self.model = None  # SentenceTransformer
        self.index = None  # FAISS index
        self.id_to_page = {}  # internal id -> (page_id, title)
        self.next_id = 0

        self._build_ui()

    # ─────────────────────────────── UI construction ───────────────────────────────
    def _build_ui(self):
        main = QVBoxLayout(self)
        tabs = QTabWidget()
        main.addWidget(tabs)

        # Index tab
        t_index = QWidget()
        form = QFormLayout()
        self.base_url = QLineEdit("https://your-domain.atlassian.net/wiki")
        self.space_key = QLineEdit("SPACEKEY")
        self.username = QLineEdit()
        self.api_token = QLineEdit()
        self.api_token.setEchoMode(QLineEdit.Password)
        self.batch_size = QSpinBox(); self.batch_size.setRange(10, 1000); self.batch_size.setValue(100)
        self.model_name = QLineEdit(MODEL_DEFAULT)
        self.progress = QProgressBar(); self.progress.setValue(0)

        btn_fetch = QPushButton("Fetch & Index")
        btn_fetch.clicked.connect(self.fetch_and_index)
        btn_load = QPushButton("Load Existing Index")
        btn_load.clicked.connect(self.load_index)

        form.addRow("Base URL", self.base_url)
        form.addRow("Space Key", self.space_key)
        form.addRow("Username", self.username)
        form.addRow("API Token", self.api_token)
        form.addRow("SBERT Model", self.model_name)
        form.addRow("Batch size", self.batch_size)
        form.addRow(btn_fetch, btn_load)
        form.addRow("Progress", self.progress)
        t_index.setLayout(form)
        tabs.addTab(t_index, "Indexing")

        # Search tab
        t_search = QWidget()
        vbox = QVBoxLayout()
        hl = QHBoxLayout()
        self.query = QLineEdit(); self.query.returnPressed.connect(self.perform_search)
        self.top_k = QSpinBox(); self.top_k.setRange(1, 50); self.top_k.setValue(5)
        btn_search = QPushButton("Search"); btn_search.clicked.connect(self.perform_search)
        hl.addWidget(QLabel("Query:")); hl.addWidget(self.query)
        hl.addWidget(QLabel("Top K:")); hl.addWidget(self.top_k)
        hl.addWidget(btn_search)
        self.results = QTextEdit(); self.results.setReadOnly(True)
        vbox.addLayout(hl); vbox.addWidget(self.results)
        t_search.setLayout(vbox)
        tabs.addTab(t_search, "Search")

    # ─────────────────────────────── Index routines ───────────────────────────────

    def _lazy_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name.text())
        return self.model

    def _new_index(self, dim: int):
        """Create a new cosine FAISS index (FlatIP with L2‑norm)."""
        idx = faiss.IndexFlatIP(dim)
        return idx

    def load_index(self):
        idx_file, _ = QFileDialog.getOpenFileName(self, "Select FAISS index", "", "FAISS index (*.index)")
        map_file, _ = QFileDialog.getOpenFileName(self, "Select mapping pickle", "", "Pickle (*.pkl)")
        if not idx_file or not map_file:
            return
        try:
            self.index = faiss.read_index(idx_file)
            with open(map_file, "rb") as f:
                self.id_to_page = pickle.load(f)
            self.next_id = len(self.id_to_page)
            self.model = self._lazy_model()  # ensure model loaded
            QMessageBox.information(self, "Loaded", f"Index loaded with {self.next_id} embeddings.")
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    def fetch_and_index(self):
        # Prepare model + new index
        model = self._lazy_model()
        dim = model.get_sentence_embedding_dimension()
        self.index = self._new_index(dim)
        self.id_to_page = {}
        self.next_id = 0

        # Fetch pages
        auth = HTTPBasicAuth(self.username.text(), self.api_token.text())
        try:
            pages = fetch_confluence_pages(self.base_url.text().rstrip("/"), self.space_key.text(), auth)
        except Exception as exc:
            QMessageBox.critical(self, "Confluence error", str(exc)); return
        total = len(pages)
        if total == 0:
            QMessageBox.warning(self, "No pages", "Space returned 0 pages."); return

        # Batch encode and add to index
        bs = self.batch_size.value()
        for start in range(0, total, bs):
            batch = pages[start : start + bs]
            inputs = [f"{title}\n{text}" for (_, title, text) in batch]
            embs = model.encode(inputs, convert_to_numpy=True).astype("float32")
            faiss.normalize_L2(embs)
            self.index.add(embs)
            for inc, (pid, title, _) in enumerate(batch):
                self.id_to_page[self.next_id + inc] = (pid, title)
            self.next_id += len(batch)
            pct = int(100 * self.next_id / total)
            self.progress.setValue(pct)
            QApplication.processEvents()

        # Persist
        faiss.write_index(self.index, INDEX_PATH)
        with open(MAP_PATH, "wb") as f:
            pickle.dump(self.id_to_page, f)
        QMessageBox.information(self, "Done", f"Indexed {self.next_id} pages.")

    # ─────────────────────────────── Search ───────────────────────────────
    def perform_search(self):
        if self.index is None:
            QMessageBox.warning(self, "No index", "Load or build an index first."); return
        query_text = self.query.text().strip()
        if not query_text:
            return
        k = self.top_k.value()
        emb = self._lazy_model().encode([query_text], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(emb)
        distances, ids = self.index.search(emb, k)
        self.results.clear()
        for idx, dist in zip(ids[0], distances[0]):
            if idx < 0:  # FAISS returns -1 when fewer than k vectors exist
                continue
            pid, title = self.id_to_page[int(idx)]
            score = dist  # already cosine similarity
            self.results.append(f"[{pid}] {title} (score: {score:.3f})")

# ───────────────────────────────────────── Entry point ────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    w = ConfluenceSearchApp(); w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
