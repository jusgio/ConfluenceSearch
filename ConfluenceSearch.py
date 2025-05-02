#!/usr/bin/env python3
"""
Confluence semantic‑search GUI (FAISS backend)

Changes in this revision
─────────────────────────
• “Fetch & Index” now produces **two** indexes:
    1. Titles‑only      → file suffix “__titles”
    2. Body‑text‑only   → file suffix “__text”
  Both share the same page‑mapping *.pkl* structure.
• Available‑index list shows “[titles]” or “[text]” so you can load either.
• Filenames remain backward‑compatible for previously created (single) indexes.
Everything else remains the same.
"""

import json
import os
import sys
import pickle
from pathlib import Path

import requests
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
    QMessageBox,
    QProgressBar,
    QFormLayout,
    QCheckBox,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
)
from PyQt5.QtCore import Qt

# ───────────────────────────── constants ──────────────────────────────
CACHE_FILE = Path("confluence_spaces.json")
INDEXES_DIR = Path("indexes")
IDX_PREFIX = "confluence_"
MAP_PREFIX = "id_to_page_"
SEPARATOR = "__"  # separates key and (sanitised) name in filenames
SUFFIX_TITLES = "titles"  # extra suffix for title‑only   index files
SUFFIX_TEXT = "text"  # extra suffix for body‑text    index files


# ───────────────────────────── helpers ────────────────────────────────
def fetch_spaces(base_url: str, auth) -> list[dict]:
    """
    Return list[{key,name,status}] for all *current* spaces visible to the user.

    For Confluence Cloud we **must NOT** pass `status=all` (Cloud returns 400).
    The default behaviour already lists only “current” (a.k.a. live) spaces,
    so archived / trashed spaces never reach the caller.
    """
    out, start, limit = [], 0, 50
    url = f"{base_url.rstrip('/')}/rest/api/space"
    while True:
        params = {"limit": limit, "start": start}
        r = requests.get(url, params=params, auth=auth, timeout=30)
        r.raise_for_status()
        data = r.json()
        out.extend(
            {
                "key": s["key"],
                "name": s["name"],
                "status": s.get("status", "current"),
            }
            for s in data.get("results", [])
        )
        if not data.get("_links", {}).get("next"):
            break
        start += limit
    return out


def space_has_pages(base_url: str, space_key: str, auth) -> bool:
    """
    Quick check to see whether the given space contains at least one page.
    Uses limit=1 so the payload is minimal.
    """
    url = f"{base_url.rstrip('/')}/rest/api/content"
    params = {"spaceKey": space_key, "limit": 1, "start": 0}
    try:
        r = requests.get(url, params=params, auth=auth, timeout=30)
        r.raise_for_status()
        return bool(r.json().get("results"))
    except Exception:
        # Treat any error as “no pages” to avoid breaking UX
        return False


def fetch_pages(base_url: str, space_key: str, auth) -> list[tuple]:
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
        items = r.json().get("results", [])
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


def _safe(s: str) -> str:
    """Return a filesystem‑safe version of *s* (letters+digits, else '_')."""
    return "".join(c if c.isalnum() else "_" for c in s)


def _index_rest(key: str, name: str) -> str:
    """The part of the filename after the prefix (sans extension)."""
    return f"{key}{SEPARATOR}{_safe(name)}"


def _parse_index_filename(path: Path) -> tuple[str, str]:
    """
    Extract (key, display_name) from an index or map filename.
    Handles optional “__titles” / “__text” suffixes for the new dual‑index mode.
    """
    stem = path.stem
    if stem.startswith(IDX_PREFIX):
        rest = stem[len(IDX_PREFIX) :]
    elif stem.startswith(MAP_PREFIX):
        rest = stem[len(MAP_PREFIX) :]
    else:
        rest = stem

    suffix = ""
    if rest.endswith(f"{SEPARATOR}{SUFFIX_TITLES}"):
        suffix = SUFFIX_TITLES
        rest_main = rest[: -len(f"{SEPARATOR}{SUFFIX_TITLES}")]
    elif rest.endswith(f"{SEPARATOR}{SUFFIX_TEXT}"):
        suffix = SUFFIX_TEXT
        rest_main = rest[: -len(f"{SEPARATOR}{SUFFIX_TEXT}")]
    else:
        rest_main = rest

    if SEPARATOR in rest_main:
        key, safe_name = rest_main.split(SEPARATOR, 1)
        disp_name = safe_name.replace("_", " ").strip()
    else:  # legacy filename containing only the key
        key = rest_main
        disp_name = key

    if suffix:
        key = f"{key}{SEPARATOR}{suffix}"  # ensures uniqueness in loaded list
        disp_name += f" [{suffix}]"
    return key, disp_name


# ──────────────────────── main GUI class ──────────────────────────────
class ConfluenceSearch(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Confluence Semantic Search (FAISS)")
        self.resize(1180, 650)

        # runtime
        self.model = None
        # each element: {"key": <unique_key>, "name": <display_name>,
        #                "index": faiss_index, "map": id_to_page}
        self.loaded_indices: list[dict] = []

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

        # spaces selector ‑‑ multi‑select list
        self.space_list = QListWidget()
        self.space_list.setSelectionMode(QAbstractItemView.MultiSelection)
        btn_spaces = QPushButton("Load Spaces")
        btn_spaces.setFixedWidth(120)
        btn_spaces.clicked.connect(self.load_spaces)
        h_space = QHBoxLayout()
        h_space.addWidget(self.space_list, stretch=1)
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

        self.titles_only = QCheckBox("List titles only (skip embed/index)")
        self.titles_only.setChecked(False)

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
        form.addRow(QLabel("Spaces:"), h_space)
        form.addRow(self.include_personal)
        form.addRow("Embedding model:", self.model_box)
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
        self.top_k.setRange(1, 300)
        self.top_k.setValue(20)
        btn_search = QPushButton("Search")
        btn_search.clicked.connect(self.do_search)
        q_line.addWidget(QLabel("Query:"))
        q_line.addWidget(self.query, stretch=1)
        q_line.addWidget(QLabel("Top‑k:"))
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
                self._fill_space_list(spaces)
                self._log(f"Loaded {len(spaces)} cached spaces.")
            except Exception as e:
                self._log(f"Cache read error: {e}")

    def _write_spaces_cache(self, spaces):
        try:
            CACHE_FILE.write_text(json.dumps(spaces, indent=2))
            self._log(f"Saved {len(spaces)} spaces to cache.")
        except Exception as e:
            self._log(f"Cache write error: {e}")

    def _fill_space_list(self, spaces):
        """Populate the multi‑select list with the space *names* only."""
        include_pers = self.include_personal.isChecked()
        self.space_list.clear()
        for s in spaces:
            if not include_pers and s["key"].startswith("~"):
                continue
            item = QListWidgetItem(s["name"])
            item.setData(Qt.UserRole, s["key"])
            self.space_list.addItem(item)
        if self.space_list.count() == 0:
            self.space_list.addItem("— no spaces —")

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

        # Filter out empty spaces
        filtered_spaces = []
        self._log("Checking spaces for content …")
        for s in spaces:
            if space_has_pages(self.base_url.text(), s["key"], auth):
                filtered_spaces.append(s)
            else:
                self._log(f"Skipping empty space: {s['name']} ({s['key']})")

        self._fill_space_list(filtered_spaces)
        self._write_spaces_cache(filtered_spaces)
        QMessageBox.information(self, "Spaces", f"{self.space_list.count()} spaces loaded.")

    # ────────── index‑loading helpers ──────────
    def _refresh_available_indexes(self):
        self.available_list.clear()
        if not INDEXES_DIR.exists():
            return
        loaded_keys = {d["key"] for d in self.loaded_indices}
        for idx_file in INDEXES_DIR.glob(f"{IDX_PREFIX}*.index"):
            key, disp_name = _parse_index_filename(idx_file)
            # Require matching map file
            map_file = INDEXES_DIR / f"{MAP_PREFIX}{idx_file.stem[len(IDX_PREFIX):]}.pkl"
            if not map_file.exists():
                continue
            if key in loaded_keys:
                continue
            item = QListWidgetItem(disp_name)
            item.setData(Qt.UserRole, idx_file)  # store full path for later
            self.available_list.addItem(item)

    def _load_index_pair(self, idx_file: Path):
        key, disp_name = _parse_index_filename(idx_file)
        rest = idx_file.stem[len(IDX_PREFIX) :]
        map_file = INDEXES_DIR / f"{MAP_PREFIX}{rest}.pkl"
        try:
            index = faiss.read_index(str(idx_file))
            id_map = pickle.load(open(map_file, "rb"))
        except Exception as e:
            self._log(f"ERROR loading {idx_file.name}: {e}")
            return
        self.loaded_indices.append({"key": key, "name": disp_name, "index": index, "map": id_map})
        self._log(f"Loaded index '{disp_name}' ({len(id_map)} vectors).")

    def load_selected_indexes(self):
        idx_paths = [
            self.available_list.item(i).data(Qt.UserRole)
            for i in range(self.available_list.count())
            if self.available_list.item(i).isSelected()
        ]
        if not idx_paths:
            QMessageBox.information(self, "Load Selected", "No indexes selected.")
            return
        for idx_path in idx_paths:
            self._load_index_pair(idx_path)
        self._refresh_available_indexes()

    def load_all_available_indexes(self):
        idx_paths = [self.available_list.item(i).data(Qt.UserRole) for i in range(self.available_list.count())]
        if not idx_paths:
            QMessageBox.information(self, "Load All", "No available indexes.")
            return
        for idx_path in idx_paths:
            self._load_index_pair(idx_path)
        self._refresh_available_indexes()

    # ────────────────────── fetch & index ────────────────────────────
    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        """Helper: encode texts in chunks, normalise to L2."""
        model = self._lazy_model()
        vecs = []
        for i in range(0, len(texts), 128):
            self.prog.setValue(int(i / len(texts) * 100))
            QApplication.processEvents()
            vecs.extend(model.encode(texts[i : i + 128], convert_to_numpy=True))
        self.prog.setValue(100)
        vecs = np.asarray(vecs, dtype="float32")
        faiss.normalize_L2(vecs)
        return vecs

    def _build_faiss_ivf(self, dim: int, vecs: np.ndarray) -> faiss.IndexIVFFlat:
        quantizer = faiss.IndexHNSWFlat(dim, 32)
        index = faiss.IndexIVFFlat(quantizer, dim, self.nlist.value(), faiss.METRIC_INNER_PRODUCT)
        index.train(vecs)
        index.add(vecs)
        index.nprobe = self.nprobe.value()
        return index

    def fetch_and_index(self):
        self._clear_log()
        self.prog.setValue(0)

        # collect selected spaces
        sel_items = [item for item in self.space_list.selectedItems() if item.data(Qt.UserRole)]
        if not sel_items:
            QMessageBox.warning(self, "No space selected", "Select at least one space to index.")
            return
        space_keys = [it.data(Qt.UserRole) for it in sel_items]
        space_names = [it.text().strip() for it in sel_items]

        self._log(f"Fetching pages from {len(space_keys)} space(s): {', '.join(space_names)} …")
        auth = HTTPBasicAuth(self.username.text(), self.api_token.text())

        pages = []
        for skey, sname in zip(space_keys, space_names):
            try:
                sp_pages = fetch_pages(self.base_url.text(), skey, auth)
                self._log(f"  • {sname}: {len(sp_pages)} pages")
                pages.extend(sp_pages)
            except Exception as e:
                self._log(f"ERROR fetching {sname} ({skey}): {e}")

        if not pages:
            QMessageBox.warning(self, "No content", "No pages were fetched from the selected spaces.")
            return

        self._log(f"TOTAL pages fetched: {len(pages)}.")

        # titles‑only diagnostic
        if self.titles_only.isChecked():
            for pid, title, *_ in pages:
                self._log(f"[{pid}] {title}")
            QMessageBox.information(self, "Done", "Titles listed in log pane.")
            return

        # Prepare lists for dual indexing
        titles_list = [title for _, title, _, _ in pages]
        body_list = [body for _, _, body, _ in pages]

        # Encode titles & build titles index
        self._log("Encoding titles …")
        titles_vecs = self._encode_texts(titles_list)
        dim = titles_vecs.shape[1]
        self._log("Building FAISS index (titles) …")
        idx_titles = self._build_faiss_ivf(dim, titles_vecs)

        # Encode body text & build text index
        self._log("Encoding body text …")
        text_vecs = self._encode_texts(body_list)
        self._log("Building FAISS index (text) …")
        idx_text = self._build_faiss_ivf(dim, text_vecs)

        # create combined filename parts
        combined_key = "+".join(space_keys)
        combined_name = f"{len(space_keys)}spaces"
        rest = _index_rest(combined_key, combined_name)

        INDEXES_DIR.mkdir(exist_ok=True)

        # ───── store titles index ─────
        idx_path_t = INDEXES_DIR / f"{IDX_PREFIX}{rest}{SEPARATOR}{SUFFIX_TITLES}.index"
        map_path_t = INDEXES_DIR / f"{MAP_PREFIX}{rest}{SEPARATOR}{SUFFIX_TITLES}.pkl"
        faiss.write_index(idx_titles, str(idx_path_t))
        pickle.dump(
            {i: (pid, title, url) for i, (pid, title, _, url) in enumerate(pages)},
            open(map_path_t, "wb"),
        )

        # ───── store text   index ─────
        idx_path_x = INDEXES_DIR / f"{IDX_PREFIX}{rest}{SEPARATOR}{SUFFIX_TEXT}.index"
        map_path_x = INDEXES_DIR / f"{MAP_PREFIX}{rest}{SEPARATOR}{SUFFIX_TEXT}.pkl"
        faiss.write_index(idx_text, str(idx_path_x))
        pickle.dump(
            {i: (pid, title, url) for i, (pid, title, _, url) in enumerate(pages)},
            open(map_path_x, "wb"),
        )

        self._log(f"Indexes built and saved:\n  • {idx_path_t.name}\n  • {idx_path_x.name}")
        QMessageBox.information(
            self,
            "Done",
            f"Indexed {len(pages)} pages across {len(space_keys)} spaces " f"→ 2 files written (titles & text).",
        )
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
                all_hits.append((sim, d["name"], pid, title, url))

        all_hits.sort(key=lambda x: x[0], reverse=True)
        all_hits = all_hits[:k]

        self.results.clear()
        for rank, (sim, space_name, pid, title, url) in enumerate(all_hits, 1):
            tag = f"[{space_name}:{pid}]"
            link = f'<a href="{url}">{title}</a>' if url else title
            self.results.append(f"{rank}. {tag} {link} (sim={sim:.3f})")


# ────────────────────── entry point ───────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ConfluenceSearch()
    win.show()
    sys.exit(app.exec_())
