#!/usr/bin/env python3
"""
confluence_search_gui.py

A PyQt5 GUI application for:
1) Fetching Confluence pages (titles + body text) from a specified space.
2) Generating embeddings locally with Sentence-Transformers.
3) Building & querying an HNSW index via hnswlib.
4) Dynamic updates and search with sub-second retrieval.
"""

import sys
import os
import pickle
import requests
from requests.auth import HTTPBasicAuth
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import hnswlib
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit,
    QPushButton, QTextEdit, QComboBox, QSpinBox,
    QVBoxLayout, QHBoxLayout, QTabWidget, QFileDialog,
    QMessageBox, QProgressBar, QFormLayout, QCheckBox
)
from PyQt5.QtCore import Qt

def fetch_confluence_pages(base_url, space_key, auth):
    """
    Fetches all pages in a Confluence space.
    Returns a list of (page_id, title, text).
    """
    pages = []
    url = f"{base_url}/rest/api/content"
    start = 0
    limit = 50
    while True:
        params = {
            "spaceKey": space_key,
            "limit": limit,
            "start": start,
            "expand": "body.storage"
        }
        resp = requests.get(url, params=params, auth=auth)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            break
        for p in results:
            pid = p["id"]
            title = p["title"]
            html = p["body"]["storage"]["value"]
            text = BeautifulSoup(html, "html.parser") \
                    .get_text(separator="\n").strip()
            pages.append((pid, title, text))
        start += limit
    return pages

class ConfluenceSearchApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Confluence Semantic Search")
        self.resize(800, 600)

        # State
        self.model = None
        self.idx = None
        self.id_to_page = {}
        self.next_id = 0

        # Build UI
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        tabs = QTabWidget()

        # ── Indexing Tab ─────────────────────────────────────────────────────
        tab_index = QWidget()
        form = QFormLayout()

        # Confluence settings
        self.base_url = QLineEdit("https://your-domain.atlassian.net/wiki")
        self.space_key = QLineEdit("SPACEKEY")
        self.username = QLineEdit()
        self.token = QLineEdit()
        self.token.setEchoMode(QLineEdit.Password)

        # Embedding settings
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "clip-ViT-B-32"
        ])
        self.title_text_cb = QCheckBox("Embed Title + Text")
        self.title_text_cb.setChecked(True)

        # HNSW parameters
        self.max_e = QSpinBox()
        self.max_e.setRange(100, 1000000)
        self.max_e.setValue(20000)
        self.ef_con = QSpinBox()
        self.ef_con.setRange(10, 1000)
        self.ef_con.setValue(200)
        self.m_links = QSpinBox()
        self.m_links.setRange(2, 64)
        self.m_links.setValue(16)
        self.ef_search = QSpinBox()
        self.ef_search.setRange(10, 1000)
        self.ef_search.setValue(50)

        # Buttons & progress
        load_btn = QPushButton("Load Existing Index")
        load_btn.clicked.connect(self.load_index)
        fetch_btn = QPushButton("Fetch & Index")
        fetch_btn.clicked.connect(self.fetch_and_index)
        self.progress = QProgressBar()
        self.progress.setValue(0)

        # Assemble form
        form.addRow("Base URL:", self.base_url)
        form.addRow("Space Key:", self.space_key)
        form.addRow("Username:", self.username)
        form.addRow("API Token:", self.token)
        form.addRow("Embedding Model:", self.model_combo)
        form.addRow(self.title_text_cb)
        form.addRow("Max Elements:", self.max_e)
        form.addRow("EF Construction:", self.ef_con)
        form.addRow("M (links):", self.m_links)
        form.addRow("EF Search:", self.ef_search)
        form.addRow(load_btn, fetch_btn)
        form.addRow("Progress:", self.progress)

        tab_index.setLayout(form)

        # ── Search Tab ───────────────────────────────────────────────────────
        tab_search = QWidget()
        sv_layout = QVBoxLayout()
        hl = QHBoxLayout()
        self.query = QLineEdit()
        self.top_k = QSpinBox()
        self.top_k.setRange(1, 100)
        self.top_k.setValue(5)
        search_btn = QPushButton("Search")
        search_btn.clicked.connect(self.perform_search)
        hl.addWidget(QLabel("Query:"))
        hl.addWidget(self.query)
        hl.addWidget(QLabel("Top K:"))
        hl.addWidget(self.top_k)
        hl.addWidget(search_btn)
        self.results = QTextEdit()
        self.results.setReadOnly(True)
        sv_layout.addLayout(hl)
        sv_layout.addWidget(self.results)
        tab_search.setLayout(sv_layout)

        # Add tabs to main layout
        tabs.addTab(tab_index, "Indexing")
        tabs.addTab(tab_search, "Search")
        layout.addWidget(tabs)

    def load_index(self):
        idx_file, _ = QFileDialog.getOpenFileName(self, "Select HNSW Index File", "", "Index Files (*.idx)")
        map_file, _ = QFileDialog.getOpenFileName(self, "Select Mapping File", "", "Pickle Files (*.pkl)")
        if not idx_file or not map_file:
            return
        try:
            model_name = self.model_combo.currentText()
            self.model = SentenceTransformer(model_name)
            dim = self.model.get_sentence_embedding_dimension()
            self.idx = hnswlib.Index(space='cosine', dim=dim)
            self.idx.load_index(idx_file)
            self.idx.set_ef(self.ef_search.value())
            with open(map_file, 'rb') as f:
                self.id_to_page = pickle.load(f)
            self.next_id = len(self.id_to_page)
            QMessageBox.information(self, "Loaded", f"Index loaded with {self.next_id} items.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def fetch_and_index(self):
        # Reset index
        model_name = self.model_combo.currentText()
        self.model = SentenceTransformer(model_name)
        dim = self.model.get_sentence_embedding_dimension()
        self.idx = hnswlib.Index(space='cosine', dim=dim)
        self.idx.init_index(
            max_elements=self.max_e.value(),
            ef_construction=self.ef_con.value(),
            M=self.m_links.value()
        )
        self.idx.set_ef(self.ef_search.value())
        self.id_to_page = {}
        self.next_id = 0

        # Fetch pages
        auth = HTTPBasicAuth(self.username.text(), self.token.text())
        pages = fetch_confluence_pages(self.base_url.text(), self.space_key.text(), auth)
        total = len(pages)

        # Batch encode & add
        batch_size = 100
        for offset in range(0, total, batch_size):
            batch = pages[offset: offset + batch_size]
            texts = [
                (f"{t}\n{txt}" if self.title_text_cb.isChecked() else t)
                for (_, t, txt) in batch
            ]
            embs = self.model.encode(texts, convert_to_numpy=True)
            ids = range(self.next_id, self.next_id + len(batch))
            self.idx.add_items(embs, ids)
            for i, (pid, title, _) in zip(ids, batch):
                self.id_to_page[i] = (pid, title)
            self.next_id += len(batch)
            self.progress.setValue(int((self.next_id / total) * 100))
            QApplication.processEvents()

        # Save to disk
        self.idx.save_index("confluence_hnsw.idx")
        with open("id_to_page.pkl", "wb") as f:
            pickle.dump(self.id_to_page, f)
        QMessageBox.information(self, "Done", f"Indexed {self.next_id} pages.")

    def perform_search(self):
        if not self.idx or not self.model:
            QMessageBox.warning(self, "Warning", "Index not loaded or built.")
            return
        q = self.query.text()
        k = self.top_k.value()
        q_emb = self.model.encode([q], convert_to_numpy=True)
        labels, distances = self.idx.knn_query(q_emb, k=k)
        self.results.clear()
        for lbl, dist in zip(labels[0], distances[0]):
            pid, title = self.id_to_page[int(lbl)]
            sim = 1 - dist
            self.results.append(f"[{pid}] {title} (score: {sim:.3f})")

def main():
    app = QApplication(sys.argv)
    win = ConfluenceSearchApp()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

