#!/usr/bin/env python3
"""
confluence_search_gui_faiss.py
Fully-local semantic search over a Confluence space, with FAISS for ANN,
and a PyQt5 GUI.  NEW: "List titles only" mode for connectivity/debugging.
"""

import os
import pickle
import requests
from requests.auth import HTTPBasicAuth
from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QTextEdit,
    QComboBox, QSpinBox, QVBoxLayout, QHBoxLayout, QTabWidget, QFileDialog,
    QMessageBox, QProgressBar, QFormLayout, QCheckBox
)
from PyQt5.QtCore import Qt
import sys


# ───────────────────────────────────────────────────────────────────────────────
# Low-level: fetch pages from Confluence
# ───────────────────────────────────────────────────────────────────────────────
def fetch_confluence_pages(base_url: str, space_key: str, auth) -> list:
    """
    Returns list of tuples: (page_id, title, text)
    """
    pages, start, limit = [], 0, 50
    url = f"{base_url.rstrip('/')}/rest/api/content"

    while True:
        params = {
            "spaceKey": space_key,
            "limit": limit,
            "start": start,
            "expand": "body.storage",
        }
        r = requests.get(url, params=params, auth=auth, timeout=30)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])
        if not results:
            break

        for p in results:
            pid = p["id"]
            title = p["title"]
            html = p["body"]["storage"]["value"]
            text = BeautifulSoup(html, "html.parser").get_text("\n").strip()
            pages.append((pid, title, text))

        start += limit
    return pages


# ───────────────────────────────────────────────────────────────────────────────
# GUI
# ───────────────────────────────────────────────────────────────────────────────
class ConfluenceSearchApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Confluence Semantic Search (FAISS)")
        self.resize(900, 600)

        # Runtime state
        self.faiss_index = None
        self.id_to_page = {}        # internal-id -> (page_id, title)
        self.model = None

        # Build UI
        self._build_ui()

    # -------------------------------------------------------------------------
    # UI construction
    # -------------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)
        tabs = QTabWidget()

        # ---------------- Indexing tab ----------------
        tab_index = QWidget()
        form = QFormLayout()

        self.base_url   = QLineEdit("https://innovmetric.atlassian.net/wiki")
        self.space_key  = QLineEdit( "Equipe4" )
        self.username   = QLineEdit( os.getenv('USERNAME') + "@innovmetric.com" )
        tok = os.getenv('CONFLUENCE_TOKEN')
        self.api_token  = QLineEdit( tok ); self.api_token.setEchoMode(QLineEdit.Password)
        self.api_token = QLineEdit()
        self.api_token.setEchoMode(QLineEdit.Password)

        self.model_box  = QComboBox()
        self.model_box.addItems([
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "paraphrase-albert-small-v2"
        ])

        self.embed_title_text = QCheckBox("Embed title + body (untick = title only)")
        self.embed_title_text.setChecked(False)

        # ★ new: dry-run checkbox
        self.titles_only = QCheckBox("List titles only (skip embedding / FAISS)")
        self.titles_only.setChecked(True)

        # FAISS indexing params
        self.faiss_nlist  = QSpinBox(); self.faiss_nlist.setRange(1, 4096); self.faiss_nlist.setValue(1)
        self.faiss_nprobe = QSpinBox(); self.faiss_nprobe.setRange(1, 64);  self.faiss_nprobe.setValue(1)

        btn_load  = QPushButton("Load Existing Index")
        btn_index = QPushButton("Fetch & Index")

        btn_load.clicked.connect(self.load_index)
        btn_index.clicked.connect(self.fetch_and_index)

        self.progress = QProgressBar(); self.progress.setValue(0)

        # layout rows
        form.addRow("Confluence base URL:", self.base_url)
        form.addRow("Space key:",          self.space_key)
        form.addRow("Username (e-mail):",  self.username)
        form.addRow("API token:",          self.api_token)
        form.addRow("Embedding model:",    self.model_box)
        form.addRow(self.embed_title_text)
        form.addRow(self.titles_only)                # ★ new row
        form.addRow("FAISS nlist (IVF):",  self.faiss_nlist)
        form.addRow("FAISS nprobe:",       self.faiss_nprobe)
        form.addRow(btn_load, btn_index)
        form.addRow("Progress:", self.progress)

        tab_index.setLayout(form)

        # ---------------- Search tab -----------------
        tab_search = QWidget()
        sv = QVBoxLayout()
        hl = QHBoxLayout()
        self.query = QLineEdit()
        self.top_k = QSpinBox(); self.top_k.setRange(1, 100); self.top_k.setValue(5)
        btn_search = QPushButton("Search")
        btn_search.clicked.connect(self.do_search)
        hl.addWidget(QLabel("Query:")); hl.addWidget(self.query)
        hl.addWidget(QLabel("Top-k:")); hl.addWidget(self.top_k); hl.addWidget(btn_search)
        self.results = QTextEdit(); self.results.setReadOnly(True)
        sv.addLayout(hl); sv.addWidget(self.results)
        tab_search.setLayout(sv)

        # assemble
        tabs.addTab(tab_index,  "Indexing / Fetch")
        tabs.addTab(tab_search, "Search")
        layout.addWidget(tabs)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _lazy_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_box.currentText())
        return self.model

    # -------------------------------------------------------------------------
    # Actions
    # -------------------------------------------------------------------------
    def load_index(self):
        idx_path, _ = QFileDialog.getOpenFileName(self, "FAISS index (*.index)", "", "FAISS index (*.index)")
        map_path, _ = QFileDialog.getOpenFileName(self, "Page map (*.pkl)", "", "Pickle (*.pkl)")
        if not idx_path or not map_path:
            return
        self.faiss_index = faiss.read_index(idx_path)
        with open(map_path, "rb") as f:
            self.id_to_page = pickle.load(f)
        QMessageBox.information(self, "Loaded", f"Loaded {len(self.id_to_page)} vectors.")

    def fetch_and_index(self):
        auth = HTTPBasicAuth(self.username.text(), self.api_token.text())
        try:
            pages = fetch_confluence_pages(self.base_url.text(), self.space_key.text(), auth)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Confluence fetch failed:\n{e}")
            return

        # -------- Titles-only debug mode ------------
        if self.titles_only.isChecked():
            print("\n=== Page titles =============================")
            for pid, title, _ in pages:
                print(f"[{pid}] {title}")
            print(f"=== {len(pages)} pages total =================\n")
            QMessageBox.information(self, "Titles listed",
                                    f"Fetched {len(pages)} pages.\n"
                                    f"Titles printed to the console.\n"
                                    f"(No embeddings or index created.)")
            return
        # --------------------------------------------

        # Normal embedding / FAISS path
        model = self._lazy_model()
        dim = model.get_sentence_embedding_dimension()
        texts = [
            (f"{title}\n{text}" if self.embed_title_text.isChecked() else title)
            for _, title, text in pages
        ]

        # batched encode
        batch_size = 128
        vectors = []
        for i in range(0, len(texts), batch_size):
            self.progress.setValue(int(i / len(texts) * 100))
            QApplication.processEvents()
            vectors.extend(model.encode(texts[i:i + batch_size], convert_to_numpy=True))
        self.progress.setValue(100)

        vectors = np.asarray(vectors, dtype="float32")
        faiss.normalize_L2(vectors)

        # Build FAISS IVF-HNSW index (configurable nlist/nprobe)
        nlist = self.faiss_nlist.value()
        quantizer = faiss.IndexHNSWFlat(dim, 32)   # inner product (cosine)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(vectors)
        index.add(vectors)
        index.nprobe = self.faiss_nprobe.value()

        # Persist
        faiss.write_index(index, "confluence_faiss.index")
        with open("id_to_page.pkl", "wb") as f:
            pickle.dump({i: (pid, title) for i, (pid, title, _) in enumerate(pages)}, f)

        # Update runtime
        self.faiss_index = index
        self.id_to_page = {i: (pid, title) for i, (pid, title, _) in enumerate(pages)}

        QMessageBox.information(self, "Done", f"Indexed {len(pages)} pages.")

    def do_search(self):
        if not self.faiss_index:
            QMessageBox.warning(self, "No index", "Load or build an index first.")
            return
        q = self.query.text()
        model = self._lazy_model()
        q_vec = model.encode([q], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_vec)
        dist, ids = self.faiss_index.search(q_vec, self.top_k.value())
        self.results.clear()
        for rank, (idx, d) in enumerate(zip(ids[0], dist[0]), 1):
            pid, title = self.id_to_page[int(idx)]
            score = 1 - d  # cosine similarity
            self.results.append(f"{rank}. [{pid}] {title}  (sim={score:.3f})")


# ───────────────────────────────────────────────────────────────────────────────
# Entry point
# ───────────────────────────────────────────────────────────────────────────────
def main() -> None:
    app = QApplication(sys.argv)
    w = ConfluenceSearchApp()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
