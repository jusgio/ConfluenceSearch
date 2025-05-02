#!/usr/bin/env python3
"""
Confluence semantic-search GUI (FAISS backend)

• Spaces combo is filled from a local JSON cache at startup.
• “Load Spaces” refreshes the list from Confluence and rewrites the cache.
• Personal spaces (~username) are omitted unless you tick
  “Include personal spaces (~)”.
• You can create a FAISS index for any Confluence space; each index is stored
  under ./indexes/ as a .index / .pkl pair.
• The Search tab lets you load multiple indexes at once (all, or selected),
  perform a semantic query across them, and shows the results ranked globally.
• When pages are fetched we now also capture each page’s URL.  The search
  results list displays the page title as a clickable link that opens in your
  default browser.
"""

import json
import os
import sys
import pickle
import requests
from pathlib import Path

from requests.auth import HTTPBasicAuth
from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QTextBrowser,
    QComboBox,
    QSpinBox,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QFileDialog,
    QMessageBox,
    QProgressBar,
    QFormLayout,
    QCheckBox,
    QListWidget,
    QAbstractItemView,
)
from PyQt5.QtCore import Qt

# ───────────────────────────── constants ──────────────────────────────
CACHE_FILE = Path("confluence_spaces.json")
INDEXES_DIR = Path("indexes")


# ───────────────────────────── helpers ────────────────────────────────
def fetch_spaces(base_url: str, auth) -> list:
    """Return list[{key,name}] for all spaces visible to the user."""
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
    """
    Return list of tuples:
        (page_id, title, body_text, page_url)
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
        items = data.get("results", [])
        if not items:
            break
        for p in items:
            pid, title = p["id"], p["title"]
            html = p["body"]["storage"]["value"]
            text = BeautifulSoup(html, "html.parser").get_text("\n").strip()
            rel = p.get("_links", {}).get("webui", f"/pages/{pid}")
            page_url = f"{base_url.rstrip('/')}{rel}"
            pages.append((pid, title, text, page_url))
        start += limit
    return pages


# ──────────────────────── main GUI class ──────────────────────────────
class ConfluenceSearch(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Confluence Semantic Search (FAISS)")
        self.resize(1180, 650)

        # runtime
        self.model = None
        # each element: {"key": safe_key, "index": faiss_index, "map": id_to_page}
        self.loaded_indices = []

        self._build_ui()
        self._load_spaces_cache()
        self._refresh_available_indexes()

    # ────────────────────── UI construction ───────────────────────────
    def _build_ui(self):
        root = QHBoxLayout(self)
        tabs = QTabWidget()

        # ────────── Index / Fetch tab ──────────
        t_idx, form = QWidget(), QFormLayout()

        self.base_url = QLineEdit("https://innovmetric.atlassian.net/wiki")
        self.username = QLineEdit(f"{os.getenv('USERNAME', '')}@innovmetric.com")
        self.api_token = QLineEdit(os.getenv("CONFLUENCE_TOKEN") or "")
        self.api_token.setEchoMode(QLineEdit.Password)

        # space selector
        self.space_box = QComboBox()
        btn_spaces = QPushButton("Load")
        btn_spaces.setFixedWidth(100)
        btn_spaces.clicked.connect(self.load_spaces)
        h_space = QHBoxLayout()
        h_space.addWidget(self.space_box)
        h_space.addWidget(btn_spaces)

        self.include_personal = QCheckBox("Include personal spaces (~)")

        self.model_box = QComboBox()
        self.model_box.addItems(
            [
                "all-MiniLM-L6-v2",
                "all-mpnet-base-v2",
                "paraphrase-albert-small-v2",
            ]
        )

        self.only_title = QCheckBox("Just title")
        self.only_title.setChecked(False)  # default: unchecked

        self.titles_only = QCheckBox("List titles only (skip embed/index)")
        self.titles_only.setChecked(False)  # default: unchecked

        self.nlist = QSpinBox()
        self.nlist.setRange(1, 4096)
        self.nlist.setValue(1)
        self.nprobe = QSpinBox()
        self.nprobe.setRange(1, 64)
        self.nprobe.setValue(1)

        btn_build = QPushButton("Fetch & Index")
        btn_build.clicked.connect(self.fetch_and_index)

        self.prog = QProgressBar()
        self.prog.setValue(0)

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
        form.addRow(btn_build)
        form.addRow("Progress:", self.prog)
        t_idx.setLayout(form)

        # ────────── Search tab ──────────
        t_search = QWidget()
        sv = QVBoxLayout()

        # section: available indexes
        avail_layout = QHBoxLayout()
        avail_layout.addWidget(QLabel("Available Indexes:"))
        self.available_list = QListWidget()
        self.available_list.setSelectionMode(QAbstractItemView.MultiSelection)
        avail_layout.addWidget(self.available_list, stretch=1)
        vbtns = QVBoxLayout()
        self.btn_load_sel = QPushButton("Load Selected Indexes")
        self.btn_load_all = QPushButton("Load All Available Indexes")
        self.btn_load_sel.clicked.connect(self.load_selected_indexes)
        self.btn_load_all.clicked.connect(self.load_all_available_indexes)
        vbtns.addWidget(self.btn_load_sel)
        vbtns.addWidget(self.btn_load_all)
        avail_layout.addLayout(vbtns)
        sv.addLayout(avail_layout)

        # section: query line
        q_line = QHBoxLayout()
        self.query = QLineEdit()
        self.query.returnPressed.connect(self.do_search)
        self.top_k = QSpinBox()
        self.top_k.setRange(1, 100)
        self.top_k.setValue(5)
        btn_search = QPushButton("Search")
        btn_search.clicked.connect(self.do_search)
        q_line.addWidget(QLabel("Query:"))
        q_line.addWidget(self.query, stretch=1)
        q_line.addWidget(QLabel("Top-k:"))
        q_line.addWidget(self.top_k)
        q_line.addWidget(btn_search)
        sv.addLayout(q_line)

        # results
        self.results = QTextBrowser()
        self.results.setOpenExternalLinks(True)
        sv.addWidget(self.results)
        t_search.setLayout(sv)

        tabs.addTab(t_idx, "Index / Fetch")
        tabs.addTab(t_search, "Search")

        # log pane
        log_layout = QVBoxLayout()
        log_layout.addWidget(QLabel("Log"))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        log_layout.addWidget(self.log)

        root.addWidget(tabs, stretch=3)
        root.addLayout(log_layout, stretch=2)

    # ────────────────────── convenience ──────────────────────────────
    def _clear_log(self):
        self.log.clear()

    def _log(self, msg: str):
        self.log.append(msg)
        print(msg, flush=True)

    def _lazy_model(self) -> SentenceTransformer:
        if self.model is None:
            name = self.model_box.currentText()
            self._log(f"Loading model {name} …")
            self.model = SentenceTransformer(name)
            self._log("Model ready.")
        return self.model

    # ────────────────────── space cache ──────────────────────────────
    def _load_spaces_cache(self):
        if CACHE_FILE.exists():
            try:
                spaces = json.loads(CACHE_FILE.read_text())
                self._fill_space_combo(spaces)
                self._log(f"Loaded {len(spaces)} cached spaces.")
            except Exception as e:
                self._log(f"Cache read error: {e}")

    def _write_spaces_cache(self, spaces):
        try:
            CACHE_FILE.write_text(json.dumps(spaces, indent=2))
            self._log(f"Saved {len(spaces)} spaces to cache.")
        except Exception as e:
            self._log(f"Cache write error: {e}")

    def _fill_space_combo(self, spaces):
        include_pers = self.include_personal.isChecked()
        self.space_box.clear()
        for s in spaces:
            if not include_pers and s["key"].startswith("~"):
                continue
            self.space_box.addItem(f"{s['key']} — {s['name']}", userData=s["key"])
        if self.space_box.count() == 0:
            self.space_box.addItem("— no spaces —")

    # ────────────────────── UI actions ───────────────────────────────
    def load_spaces(self):
        self._clear_log()
        auth = HTTPBasicAuth(self.username.text(), self.api_token.text())
        try:
            spaces = fetch_spaces(self.base_url.text(), auth)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not list spaces:\n{e}")
            self._log(f"ERROR: {e}")
            return
        self._fill_space_combo(spaces)
        self._write_spaces_cache(spaces)
        QMessageBox.information(self, "Spaces", f"{self.space_box.count()} spaces loaded.")

    # ────────── index-loading helpers ──────────
    def _refresh_available_indexes(self):
        self.available_list.clear()
        if not INDEXES_DIR.exists():
            return
        loaded_keys = {d["key"] for d in self.loaded_indices}
        for idx_file in INDEXES_DIR.glob("confluence_*.index"):
            key = idx_file.stem[len("confluence_") :]
            if key in loaded_keys:
                continue
            if not (INDEXES_DIR / f"id_to_page_{key}.pkl").exists():
                continue
            self.available_list.addItem(key)

    def _load_index_pair(self, key: str):
        idx_file = INDEXES_DIR / f"confluence_{key}.index"
        map_file = INDEXES_DIR / f"id_to_page_{key}.pkl"
        index = faiss.read_index(str(idx_file))
        id_map = pickle.load(open(map_file, "rb"))
        self.loaded_indices.append({"key": key, "index": index, "map": id_map})
        self._log(f"Loaded index '{key}' ({len(id_map)} vectors).")

    def load_selected_indexes(self):
        keys = [item.text() for item in self.available_list.selectedItems()]
        if not keys:
            QMessageBox.information(self, "Load Selected", "No indexes selected.")
            return
        for key in keys:
            self._load_index_pair(key)
        self._refresh_available_indexes()

    def load_all_available_indexes(self):
        keys = [self.available_list.item(i).text() for i in range(self.available_list.count())]
        if not keys:
            QMessageBox.information(self, "Load All", "No available indexes.")
            return
        for key in keys:
            self._load_index_pair(key)
        self._refresh_available_indexes()

    # ────────────────────── fetch & index ────────────────────────────
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
            self._log(f"ERROR: {e}")
            return

        self._log(f"{len(pages)} pages fetched.")

        # titles-only diagnostic
        if self.titles_only.isChecked():
            for pid, title, *_ in pages:
                self._log(f"[{pid}] {title}")
            QMessageBox.information(self, "Done", "Titles listed in log pane.")
            return

        model = self._lazy_model()
        texts = [title if self.only_title.isChecked() else f"{title}\n{body}" for _, title, body, _ in pages]
        dim = model.get_sentence_embedding_dimension()

        vecs = []
        for i in range(0, len(texts), 128):
            self.prog.setValue(int(i / len(texts) * 100))
            QApplication.processEvents()
            vecs.extend(model.encode(texts[i : i + 128], convert_to_numpy=True))
        self.prog.setValue(100)
        vecs = np.asarray(vecs, dtype="float32")
        faiss.normalize_L2(vecs)

        self._log("Building FAISS index …")
        quantizer = faiss.IndexHNSWFlat(dim, 32)
        index = faiss.IndexIVFFlat(quantizer, dim, self.nlist.value(), faiss.METRIC_INNER_PRODUCT)
        index.train(vecs)
        index.add(vecs)
        index.nprobe = self.nprobe.value()

        safe_key = "".join(ch if ch.isalnum() else "_" for ch in space_key)
        INDEXES_DIR.mkdir(exist_ok=True)
        idx_path = INDEXES_DIR / f"confluence_{safe_key}.index"
        map_path = INDEXES_DIR / f"id_to_page_{safe_key}.pkl"

        faiss.write_index(index, str(idx_path))
        pickle.dump(
            {i: (pid, title, url) for i, (pid, title, _, url) in enumerate(pages)},
            open(map_path, "wb"),
        )

        self._log(f"Index built and saved ({idx_path}).")
        QMessageBox.information(self, "Done", f"Indexed {len(pages)} pages.")
        self._refresh_available_indexes()

    # ─────────────────────── search ────────────────────────────────
    def do_search(self):
        if not self.loaded_indices:
            QMessageBox.warning(self, "No index", "Load or build at least one index.")
            return
        query = self.query.text().strip()
        if not query:
            return

        self._log(f"Query: {query!r}")

        vec = self._lazy_model().encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(vec)

        k = self.top_k.value()
        all_hits = []
        for d in self.loaded_indices:
            dist, ids = d["index"].search(vec, k)  # higher dist → better
            for idx, sim in zip(ids[0], dist[0]):
                if idx < 0:
                    continue
                pid, title, url = d["map"][int(idx)]
                all_hits.append((sim, d["key"], pid, title, url))

        # best similarity first
        all_hits.sort(key=lambda x: x[0], reverse=True)
        all_hits = all_hits[:k]

        self.results.clear()
        for rank, (sim, key, pid, title, url) in enumerate(all_hits, 1):
            if url:
                self.results.append(f'{rank}. [{key}:{pid}] <a href="{url}">{title}</a> ' f"(sim={sim:.3f})")
            else:
                self.results.append(f"{rank}. [{key}:{pid}] {title} (sim={sim:.3f})")


# ────────────────────── entry point ───────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ConfluenceSearch()
    win.show()
    sys.exit(app.exec_())
