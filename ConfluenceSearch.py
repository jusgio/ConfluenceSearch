#!/usr/bin/env python3
"""
confluence_search_gui_faiss.py
PyQt5 GUI for:
  • Fetching a Confluence space
  • (Optionally) embedding with Sentence-Transformers
  • Building a FAISS IVF-HNSW index
NEW:
  • Right-hand log window (QTextEdit) so you don’t need the debug console
  • Default values for URL, spaceKey, username, token
"""

import os
import sys
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


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def fetch_confluence_pages(base_url: str, space_key: str, auth) -> list:
    """
    Return list of tuples: (page_id, title, text)
    """
    pages, start, limit = [], 0, 50
    url = f"{base_url.rstrip('/')}/rest/api/content"

    while True:
        params = {
            "spaceKey": space_key,
            "limit": limit,
            "start": start,
            "expand": "body.storage"
        }
        r = requests.get(url, params=params, auth=auth, timeout=30)
        r.raise_for_status()
        data = r.json()
        items = data.get("results", [])
        if not items:
            break

        for p in items:
            pid = p["id"]
            title = p["title"]
            html = p["body"]["storage"]["value"]
            text = BeautifulSoup(html, "html.parser").get_text("\n").strip()
            pages.append((pid, title, text))
        start += limit
    return pages


# ──────────────────────────────────────────────────────────────────────────────
# GUI
# ──────────────────────────────────────────────────────────────────────────────
class ConfluenceSearchApp(QWidget):
    # ────────────────────────────── UI ───────────────────────────────────────
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Confluence Semantic Search (FAISS)")
        self.resize(1100, 650)

        # runtime
        self.faiss_index = None
        self.id_to_page = {}
        self.model = None

        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)                         # split: tabs | log
        tabs = QTabWidget()

        # ── Indexing tab ──────────────────────────────────────────────────
        tab_index = QWidget()
        form = QFormLayout()

        # default values you requested
        self.base_url = QLineEdit("https://innovmetric.atlassian.net/wiki")
        self.space_key = QLineEdit("Equipe4")
        self.username = QLineEdit(f"{os.getenv('USERNAME')}@innovmetric.com")
        self.api_token = QLineEdit(os.getenv("CONFLUENCE_TOKEN") or "")
        self.api_token.setEchoMode(QLineEdit.Password)

        self.model_box = QComboBox()
        self.model_box.addItems([
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "paraphrase-albert-small-v2"
        ])

        self.embed_title_body = QCheckBox("Embed title + body text")
        self.embed_title_body.setChecked(False)              # per request

        self.titles_only = QCheckBox("List titles only (skip embed/index)")
        self.titles_only.setChecked(True)                    # per request

        # FAISS params
        self.nlist = QSpinBox();  self.nlist.setRange(1, 4096); self.nlist.setValue(1)
        self.nprobe = QSpinBox(); self.nprobe.setRange(1, 64);  self.nprobe.setValue(1)

        btn_load  = QPushButton("Load Existing Index")
        btn_build = QPushButton("Fetch & Index")
        btn_load.clicked.connect(self.load_index)
        btn_build.clicked.connect(self.fetch_and_index)

        self.progress = QProgressBar(); self.progress.setValue(0)

        form.addRow("Confluence base URL:", self.base_url)
        form.addRow("Space key:",           self.space_key)
        form.addRow("Username (email):",    self.username)
        form.addRow("API token:",           self.api_token)
        form.addRow("Embedding model:",     self.model_box)
        form.addRow(self.embed_title_body)
        form.addRow(self.titles_only)
        form.addRow("FAISS nlist:",         self.nlist)
        form.addRow("FAISS nprobe:",        self.nprobe)
        form.addRow(btn_load, btn_build)
        form.addRow("Progress:",            self.progress)
        tab_index.setLayout(form)

        # ── Search tab ─────────────────────────────────────────────────────
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

        tabs.addTab(tab_index,  "Index / Fetch")
        tabs.addTab(tab_search, "Search")

        # ── Log pane ───────────────────────────────────────────────────────
        log_layout = QVBoxLayout()
        log_label  = QLabel("Log")
        self.log_widget = QTextEdit(); self.log_widget.setReadOnly(True)
        log_layout.addWidget(log_label); log_layout.addWidget(self.log_widget)

        # assemble split view
        root.addWidget(tabs, stretch=3)
        root.addLayout(log_layout, stretch=2)

    # ────────────────────── utility: write to log pane ────────────────────
    def log(self, msg: str):
        self.log_widget.append(msg)
        sys.stdout.write(msg + "\n")
        sys.stdout.flush()

    # ────────────────────── lazy model loader ─────────────────────────────
    def _lazy_model(self):
        if self.model is None:
            self.log(f"Loading model: {self.model_box.currentText()} …")
            self.model = SentenceTransformer(self.model_box.currentText())
            self.log("Model loaded.")
        return self.model

    # ────────────────────── menu actions ──────────────────────────────────
    def load_index(self):
        idx_path, _ = QFileDialog.getOpenFileName(self, "FAISS index (*.index)", "", "Index (*.index)")
        map_path, _ = QFileDialog.getOpenFileName(self, "Page map (*.pkl)", "", "Pickle (*.pkl)")
        if not idx_path or not map_path:
            return
        self.faiss_index = faiss.read_index(idx_path)
        with open(map_path, "rb") as f:
            self.id_to_page = pickle.load(f)
        self.log(f"Loaded index with {len(self.id_to_page)} vectors.")

    def fetch_and_index(self):
        self.progress.setValue(0)
        self.log("Fetching pages …")
        auth = HTTPBasicAuth(self.username.text(), self.api_token.text())
        try:
            pages = fetch_confluence_pages(self.base_url.text(), self.space_key.text(), auth)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Confluence fetch failed:\n{e}")
            self.log(f"ERROR: {e}")
            return
        self.log(f"Fetched {len(pages)} pages.")

        # ­­—— titles-only mode
        if self.titles_only.isChecked():
            self.log("=== Page titles ==================================")
            for pid, title, _ in pages:
                self.log(f"[{pid}] {title}")
            self.log(f"=== {len(pages)} titles printed ==================\n")
            QMessageBox.information(self, "Titles only", "Titles listed in log pane.")
            return

        # ­­—— embedding + FAISS
        model = self._lazy_model()
        texts = [
            (f"{title}\n{text}" if self.embed_title_body.isChecked() else title)
            for _, title, text in pages
        ]
        dim = model.get_sentence_embedding_dimension()

        self.log("Encoding …")
        batch, vecs = 128, []
        for i in range(0, len(texts), batch):
            self.progress.setValue(int(i / len(texts) * 100))
            QApplication.processEvents()
            vecs.extend(model.encode(texts[i:i + batch], convert_to_numpy=True))
        self.progress.setValue(100)
        vecs = np.asarray(vecs, dtype="float32")
        faiss.normalize_L2(vecs)
        self.log("Building FAISS index …")

        nlist = self.nlist.value()
        quantizer = faiss.IndexHNSWFlat(dim, 32)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(vecs)
        index.add(vecs)
        index.nprobe = self.nprobe.value()

        faiss.write_index(index, "confluence_faiss.index")
        with open("id_to_page.pkl", "wb") as f:
            pickle.dump({i: (pid, title) for i, (pid, title, _) in enumerate(pages)}, f)

        self.faiss_index = index
        self.id_to_page = {i: (pid, title) for i, (pid, title, _) in enumerate(pages)}

        self.log("Index built and saved (confluence_faiss.index).")
        QMessageBox.information(self, "Done", f"Indexed {len(pages)} pages.")

    def do_search(self):
        if not self.faiss_index:
            QMessageBox.warning(self, "No index", "Load or build an index first.")
            return
        q = self.query.text()
        if not q.strip():
            return
        self.log(f"Searching for: {q!r}")
        model = self._lazy_model()
        q_vec = model.encode([q], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_vec)
        dist, ids = self.faiss_index.search(q_vec, self.top_k.value())
        self.results.clear()
        for rank, (idx, d) in enumerate(zip(ids[0], dist[0]), 1):
            pid, title = self.id_to_page[int(idx)]
            score = 1 - d
            self.results.append(f"{rank}. [{pid}] {title}  (sim={score:.3f})")
        self.log("Search finished.")


# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    app = QApplication(sys.argv)
    w = ConfluenceSearchApp()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
