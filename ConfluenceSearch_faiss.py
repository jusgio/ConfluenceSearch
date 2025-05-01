# -*- coding: utf-8 -*-
from __future__ import unicode_literals

# This script rewrites the original PyQt5 GUI-based Confluence search app to use FAISS instead of hnswlib.

import sys
import os
import pickle
import requests
from requests.auth import HTTPBasicAuth
from bs4 import BeautifulSoup
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QTextEdit,
    QComboBox, QSpinBox, QVBoxLayout, QHBoxLayout, QTabWidget, QFileDialog,
    QMessageBox, QProgressBar, QFormLayout, QCheckBox
)
from PyQt5.QtCore import Qt

def fetch_confluence_pages(base_url, space_key, auth):
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
            text = BeautifulSoup(html, "html.parser").get_text(separator="\n").strip()
            pages.append((pid, title, text))
        start += limit
    return pages

class ConfluenceFAISSGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Confluence Search with FAISS")
        self.resize(800, 600)
        self.model = None
        self.index = None
        self.id_to_page = {}
        self.embeddings = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        tabs = QTabWidget()

        # Indexing Tab
        tab_index = QWidget()
        form = QFormLayout()
        self.base_url = QLineEdit("https://your-domain.atlassian.net/wiki")
        self.space_key = QLineEdit("SPACEKEY")
        self.username = QLineEdit()
        self.token = QLineEdit()
        self.token.setEchoMode(QLineEdit.Password)
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "clip-ViT-B-32"
        ])
        self.title_text_cb = QCheckBox("Embed Title + Text")
        self.title_text_cb.setChecked(True)
        load_btn = QPushButton("Load Existing Index")
        load_btn.clicked.connect(self.load_index)
        fetch_btn = QPushButton("Fetch & Index")
        fetch_btn.clicked.connect(self.fetch_and_index)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        form.addRow("Base URL:", self.base_url)
        form.addRow("Space Key:", self.space_key)
        form.addRow("Username:", self.username)
        form.addRow("API Token:", self.token)
        form.addRow("Embedding Model:", self.model_combo)
        form.addRow(self.title_text_cb)
        form.addRow(load_btn, fetch_btn)
        form.addRow("Progress:", self.progress)
        tab_index.setLayout(form)

        # Search Tab
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

        tabs.addTab(tab_index, "Indexing")
        tabs.addTab(tab_search, "Search")
        layout.addWidget(tabs)

    def load_index(self):
        index_file, _ = QFileDialog.getOpenFileName(self, "Select FAISS Index", "", "*.index")
        map_file, _ = QFileDialog.getOpenFileName(self, "Select Mapping File", "", "*.pkl")
        if not index_file or not map_file:
            return
        self.index = faiss.read_index(index_file)
        with open(map_file, "rb") as f:
            self.id_to_page = pickle.load(f)
        QMessageBox.information(self, "Loaded", f"Loaded FAISS index with {len(self.id_to_page)} entries.")

    def fetch_and_index(self):
        model_name = self.model_combo.currentText()
        self.model = SentenceTransformer(model_name)
        auth = HTTPBasicAuth(self.username.text(), self.token.text())
        pages = fetch_confluence_pages(self.base_url.text(), self.space_key.text(), auth)
        total = len(pages)
        texts = []
        for _, title, text in pages:
            content = f"{title}\n{text}" if self.title_text_cb.isChecked() else title
            texts.append(content)
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.embeddings = embeddings
        self.id_to_page = {i: (pid, title) for i, (pid, title, _) in enumerate(pages)}
        self.progress.setValue(100)
        faiss.write_index(self.index, "faiss_confluence.index")
        with open("id_to_page.pkl", "wb") as f:
            pickle.dump(self.id_to_page, f)
        QMessageBox.information(self, "Done", f"Indexed {len(pages)} pages with FAISS.")

    def perform_search(self):
        if not self.index or not self.model:
            QMessageBox.warning(self, "Warning", "Index not loaded.")
            return
        q = self.query.text()
        top_k = self.top_k.value()
        q_emb = self.model.encode([q], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        scores, indices = self.index.search(q_emb, top_k)
        self.results.clear()
        for idx, score in zip(indices[0], scores[0]):
            pid, title = self.id_to_page.get(idx, ("?", "?"))
            self.results.append(f"[{pid}] {title} (score: {score:.3f})")

def main():
    app = QApplication(sys.argv)
    win = ConfluenceFAISSGUI()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
