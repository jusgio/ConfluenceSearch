#!/usr/bin/env python3
"""
Confluence semantic-search GUI (FAISS backend)
"""

import json, os, sys, pickle, requests
from pathlib import Path
from requests.auth import HTTPBasicAuth
from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QTextEdit,
    QComboBox, QSpinBox, QVBoxLayout, QHBoxLayout, QTabWidget, QFileDialog,
    QMessageBox, QProgressBar, QFormLayout, QCheckBox, QTextBrowser
)
from PyQt5.QtCore import Qt


CACHE_FILE  = Path("confluence_spaces.json")
INDEXES_DIR = Path("indexes")


def fetch_spaces(base_url: str, auth) -> list:
    out, start, limit = [], 0, 50
    url = f"{base_url.rstrip('/')}/rest/api/space"
    while True:
        r = requests.get(url, params={"limit": limit, "start": start}, auth=auth, timeout=30)
        r.raise_for_status()
        data = r.json()
        out.extend({"key": s["key"], "name": s["name"]} for s in data.get("results", []))
        if not data.get("_links", {}).get("next"):
            break
        start += limit
    return out


def fetch_pages(base_url: str, space_key: str, auth) -> list:
    pages, start, limit = [], 0, 50
    url = f"{base_url.rstrip('/')}/rest/api/content"
    while True:
        params = {"spaceKey": space_key, "limit": limit, "start": start, "expand": "body.storage"}
        r = requests.get(url, params=params, auth=auth, timeout=30)
        r.raise_for_status()
        data = r.json()
        items = data.get("results", [])
        if not items:
            break
        for p in items:
            pid, title = p["id"], p["title"]
            page_url = base_url.rstrip("/") + p["_links"]["webui"]
            html = p["body"]["storage"]["value"]
            text = BeautifulSoup(html, "html.parser").get_text("\n").strip()
            pages.append((pid, title, page_url, text))
        start += limit
    return pages


class ConfluenceSearch(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Confluence Semantic Search (FAISS)")
        self.resize(1180, 650)

        self.faiss_index = None
        self.id_to_page, self.model = {}, None

        self._build_ui()
        self._load_spaces_cache()

    def _build_ui(self):
        root = QHBoxLayout(self)
        tabs = QTabWidget()

        t_idx, form = QWidget(), QFormLayout()
        self.base_url = QLineEdit("https://innovmetric.atlassian.net/wiki")
        self.username = QLineEdit(f"{os.getenv('USERNAME')}@innovmetric.com")
        self.api_token = QLineEdit(os.getenv("CONFLUENCE_TOKEN") or "")
        self.api_token.setEchoMode(QLineEdit.Password)

        self.space_box = QComboBox()
        btn_spaces = QPushButton("Load")
        btn_spaces.setFixedWidth(100)
        btn_spaces.clicked.connect(self.load_spaces)

        h_space = QHBoxLayout()
        h_space.addWidget(self.space_box)
        h_space.addWidget(btn_spaces)

        self.include_personal = QCheckBox("Include personal spaces (~)")
        self.include_personal.setChecked(False)

        self.model_box = QComboBox()
        self.model_box.addItems([
            "all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-albert-small-v2"
        ])
        self.only_title = QCheckBox("Just title")
        self.only_title.setChecked(True)
        self.titles_only = QCheckBox("List titles only (skip embed/index)")
        self.titles_only.setChecked(True)

        self.nlist = QSpinBox(); self.nlist.setRange(1, 4096); self.nlist.setValue(1)
        self.nprobe = QSpinBox(); self.nprobe.setRange(1, 64); self.nprobe.setValue(1)

        btn_load = QPushButton("Load Existing Index")
        btn_build = QPushButton("Fetch & Index")
        btn_load.clicked.connect(self.load_index)
        btn_build.clicked.connect(self.fetch_and_index)

        self.prog = QProgressBar(); self.prog.setValue(0)

        form.addRow("Base URL:", self.base_url)
        form.addRow("Username:", self.username)
        form.addRow("API token:", self.api_token)
        form.addRow(QLabel("Space:"), h_space)
        form.addRow(self.include_personal)
        form.addRow("Embedding model:", self.model_box)
        form.addRow(self.only_title)
        form.addRow(self.titles_only)
        form.addRow("FAISS nlist:", self.nlist)
        form.addRow("FAISS nprobe:", self.nprobe)
        form.addRow(btn_load, btn_build)
        form.addRow("Progress:", self.prog)
        t_idx.setLayout(form)

        t_search, sv = QWidget(), QVBoxLayout()
        hl = QHBoxLayout()

        self.query = QLineEdit()
        self.top_k = QSpinBox(); self.top_k.setRange(1, 100); self.top_k.setValue(5)
        btn_search = QPushButton("Search")

        # ← connect signals
        btn_search.clicked.connect(self.do_search)
        self.query.returnPressed.connect(self.do_search)

        hl.addWidget(QLabel("Query:"))
        hl.addWidget(self.query)
        hl.addWidget(QLabel("Top-k:"))
        hl.addWidget(self.top_k)
        hl.addWidget(btn_search)

        self.results = QTextBrowser(); self.results.setReadOnly(True)
        self.results.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.results.setOpenExternalLinks(True)

        sv.addLayout(hl); sv.addWidget(self.results); t_search.setLayout(sv)

        tabs.addTab(t_idx, "Index / Fetch")
        tabs.addTab(t_search, "Search")

        log_box = QVBoxLayout()
        log_box.addWidget(QLabel("Log"))
        self.log = QTextEdit(); self.log.setReadOnly(True)
        log_box.addWidget(self.log)

        root.addWidget(tabs, stretch=3)
        root.addLayout(log_box, stretch=2)

    def _clear_log(self):
        self.log.clear()

    def _log(self, msg: str):
        self.log.append(msg); print(msg, flush=True)

    def _lazy_model(self):
        if self.model is None:
            self._log(f"Loading model {self.model_box.currentText()} …")
            self.model = SentenceTransformer(self.model_box.currentText())
            self._log("Model ready.")
        return self.model

    def _load_spaces_cache(self):
        if CACHE_FILE.exists():
            try:
                spaces = json.loads(CACHE_FILE.read_text())
                self._fill_space_combo(spaces)
                self._log(f"Loaded {len(spaces)} cached spaces.")
            except Exception as e:
                self._log(f"Cache read error: {e}")

    def _write_spaces_cache(self, spaces: list):
        try:
            CACHE_FILE.write_text(json.dumps(spaces, indent=2))
            self._log(f"Saved {len(spaces)} spaces to cache.")
        except Exception as e:
            self._log(f"Cache write error: {e}")

    def _fill_space_combo(self, spaces: list):
        include_personal = self.include_personal.isChecked()
        self.space_box.clear()
        for s in spaces:
            if not include_personal and s["key"].startswith("~"):
                continue
            self.space_box.addItem(f"{s['key']} — {s['name']}", userData=s["key"])
        if self.space_box.count() == 0:
            self.space_box.addItem("— no spaces —")

    def load_spaces(self):
        self._clear_log()
        auth = HTTPBasicAuth(self.username.text(), self.api_token.text())
        try:
            spaces = fetch_spaces(self.base_url.text(), auth)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not list spaces:\n{e}")
            self._log(f"ERROR: {e}"); return
        self._fill_space_combo(spaces)
        self._write_spaces_cache(spaces)
        QMessageBox.information(self, "Spaces", f"{self.space_box.count()} spaces loaded.")

    def load_index(self):
        self._clear_log()
        # Default to the indexes folder when it exists
        default_dir = str(INDEXES_DIR) if INDEXES_DIR.exists() else ""
        idx_path, _ = QFileDialog.getOpenFileName(self, "FAISS index", default_dir, "Index (*.index)")
        map_path, _ = QFileDialog.getOpenFileName(self, "Page map",   default_dir, "Pickle (*.pkl)")
        if not idx_path or not map_path: return
        self.faiss_index = faiss.read_index(idx_path)
        self.id_to_page = pickle.load(open(map_path, "rb"))
        self._log(f"Loaded index with {len(self.id_to_page)} vectors.")

    def fetch_and_index(self):
        self._clear_log()
        self.prog.setValue(0)
        auth = HTTPBasicAuth(self.username.text(), self.api_token.text())
        space_key = self.space_box.currentData() or self.space_box.currentText().split(" — ")[0]
        self._log(f"Fetching pages from space {space_key} …")

        try:
            pages = fetch_pages(self.base_url.text(), space_key, auth)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Fetch failed:\n{e}")
            self._log(f"ERROR: {e}"); return

        self._log(f"{len(pages)} pages fetched.")

        if self.titles_only.isChecked():
            for pid, title, _, _ in pages: self._log(f"[{pid}] {title}")
            QMessageBox.information(self, "Done", "Titles listed in log pane."); return

        model = self._lazy_model()
        texts = [title if self.only_title.isChecked() else f"{title}\n{body}"
                 for _, title, _, body in pages]
        dim = model.get_sentence_embedding_dimension()

        vecs = []
        for i in range(0, len(texts), 128):
            self.prog.setValue(int(i/len(texts)*100)); QApplication.processEvents()
            vecs.extend(model.encode(texts[i:i+128], convert_to_numpy=True))
        self.prog.setValue(100)
        vecs = np.asarray(vecs, dtype="float32"); faiss.normalize_L2(vecs)

        self._log("Building FAISS index …")
        quantizer = faiss.IndexHNSWFlat(dim, 32)
        index = faiss.IndexIVFFlat(quantizer, dim, self.nlist.value(), faiss.METRIC_INNER_PRODUCT)
        index.train(vecs); index.add(vecs); index.nprobe = self.nprobe.value()

        # save inside indexes/ sub-folder
        safe_key = "".join(ch if ch.isalnum() else "_" for ch in space_key)
        INDEXES_DIR.mkdir(exist_ok=True)

        idx_path = INDEXES_DIR / f"confluence_{safe_key}.index"
        map_path = INDEXES_DIR / f"id_to_page_{safe_key}.pkl"

        faiss.write_index(index, str(idx_path))
        pickle.dump({i: (pid, title, url) for i, (pid, title, url, _) in enumerate(pages)},
                    open(map_path, "wb"))

        self.faiss_index = index
        self.id_to_page  = {i: (pid, title, url) for i, (pid, title, url, _) in enumerate(pages)}
        self._log(f"Index built and saved ({idx_path}).")

        QMessageBox.information(self, "Done", f"Indexed {len(pages)} pages.")

    def do_search(self):
        if not self.faiss_index:
            QMessageBox.warning(self, "No index", "Load or build an index."); return
        q = self.query.text().strip()
        if not q: return
        self._log(f"Query: {q!r}")
        vec = self._lazy_model().encode([q], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(vec)
        dist, ids = self.faiss_index.search(vec, self.top_k.value())
        self.results.clear()
        
        self.results.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.results.setOpenExternalLinks(True)

        html = ""
        for rank, (idx, d) in enumerate(zip(ids[0], dist[0]), 1):
            if idx == -1 or idx not in self.id_to_page:
                continue          
            pid, title, url = self.id_to_page[int(idx)]
            score = 1 - d  # cosine sim
            html += f"{rank}. [{pid}] <a href=\"{url}\">{title}</a> (sim={score:.3f})<br><br>\n"
        self.results.setHtml(html or "No results found.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ConfluenceSearch(); win.show()
    sys.exit(app.exec_())
