"""Audit artifact generation for visual chunk-quality inspection."""

from __future__ import annotations

import json
import logging
from html import escape
from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image as PILImage

from src.core.models import ProcessedChunk, RawChunk
from src.utils.serialization import serialize_chunk

log = logging.getLogger(__name__)


def save_chunk_audit(
    pdf_path: str | Path,
    extracted: list[tuple[RawChunk, ProcessedChunk]],
    accepted: list[ProcessedChunk],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Persist audit artifacts for visual review of extracted chunks."""
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audit_root = output_dir / f"{pdf_path.stem}_audit"
    pages_dir = audit_root / "pages"
    figures_dir = audit_root / "figures"
    pages_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    page_images = _render_page_previews(pdf_path, pages_dir)
    accepted_ids = {chunk.chunk_id for chunk in accepted}

    audit_entries: list[dict[str, Any]] = []
    for raw, processed in extracted:
        figure_artifact = _export_figure_artifact(raw, processed, figures_dir)
        status = _resolve_status(processed, accepted_ids)
        audit_entries.append(
            {
                "status": status,
                "raw": _serialize_raw_chunk(raw, figure_artifact),
                "processed": serialize_chunk(processed),
                "corrected": serialize_chunk(processed.validation.corrected)
                if processed.validation and processed.validation.corrected
                else None,
            }
        )

    audit_data = {
        "pdf_file": pdf_path.name,
        "pages": [
            {
                "page_num": page_num,
                "image_path": path.as_posix(),
            }
            for page_num, path in sorted(page_images.items())
        ],
        "chunks": audit_entries,
    }

    json_path = output_dir / f"{pdf_path.stem}_audit.json"
    html_path = output_dir / f"{pdf_path.stem}_audit.html"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(audit_data, handle, ensure_ascii=False, indent=2)

    with html_path.open("w", encoding="utf-8") as handle:
        handle.write(_build_html(audit_data))

    log.info("✓ Saved audit report to %s", html_path)
    log.info("✓ Saved audit data to %s", json_path)
    return {"json": json_path, "html": html_path, "root": audit_root}


def _render_page_previews(pdf_path: Path, pages_dir: Path) -> dict[int, Path]:
    """Render each PDF page to a preview image for overlay review."""
    rendered: dict[int, Path] = {}
    document = pymupdf.open(str(pdf_path))
    try:
        for page_index in range(len(document)):
            page = document[page_index]
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
            image_path = pages_dir / f"page_{page_index + 1:03d}.png"
            pixmap.save(str(image_path))
            rendered[page_index + 1] = image_path.relative_to(pages_dir.parent.parent)
    finally:
        document.close()
    return rendered


def _export_figure_artifact(raw: RawChunk, processed: ProcessedChunk, figures_dir: Path) -> str | None:
    """Export raw figure images to disk and return relative path."""
    if not isinstance(raw.raw_content, PILImage.Image):
        return processed.artifact_path or None

    image_path = figures_dir / f"{processed.chunk_id}.png"
    if not image_path.exists():
        raw.raw_content.save(image_path, format="PNG")
    return image_path.relative_to(figures_dir.parent.parent).as_posix()


def _resolve_status(processed: ProcessedChunk, accepted_ids: set[str]) -> str:
    """Determine audit status for a processed chunk."""
    if processed.validation and processed.validation.corrected:
        return "corrected" if processed.validation.corrected.chunk_id in accepted_ids else "discarded"
    if processed.chunk_id in accepted_ids:
        return "accepted"
    if processed.validation and not processed.validation.is_valid:
        return "discarded"
    return "discarded"


def _serialize_raw_chunk(raw: RawChunk, artifact_path: str | None) -> dict[str, Any]:
    """Serialize raw chunk metadata for audit output."""
    bbox = raw.bbox
    return {
        "chunk_type": raw.chunk_type.value,
        "page_num": raw.page_num,
        "source_file": raw.source_file,
        "bbox": {
            "x0": bbox[0],
            "y0": bbox[1],
            "x1": bbox[2],
            "y1": bbox[3],
        }
        if bbox is not None
        else None,
        "page_size": {
            "width": raw.page_width,
            "height": raw.page_height,
        },
        "source_preview": raw.source_preview,
        "artifact_path": artifact_path or raw.artifact_path or None,
    }


def _build_html(audit_data: dict[str, Any]) -> str:
    """Generate a standalone HTML viewer for the audit JSON."""
    payload = json.dumps(audit_data, ensure_ascii=False)
    # Escape closing script tags and other HTML-sensitive sequences within JavaScript.
    payload = payload.replace("</", "<\\/")
    title = escape(f"Chunk Audit - {audit_data['pdf_file']}")
    template = """<!DOCTYPE html>
<html lang=\"ja\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>__TITLE__</title>
  <style>
    :root {{
      --bg: #f4efe7;
      --panel: #fffdf8;
      --ink: #1f2933;
      --muted: #6b7280;
      --text: #2f855a;
      --table: #b45309;
      --figure: #1d4ed8;
      --discarded: #b91c1c;
      --border: #d8d1c3;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: 'Iowan Old Style', 'Palatino Linotype', serif; background: linear-gradient(180deg, #efe6d7 0%, var(--bg) 60%, #e8ded0 100%); color: var(--ink); }}
    .layout {{ display: grid; grid-template-columns: 360px 1fr; min-height: 100vh; }}
    .sidebar {{ padding: 24px; border-right: 1px solid var(--border); background: rgba(255,255,255,0.7); backdrop-filter: blur(10px); overflow: auto; }}
    .content {{ padding: 24px; overflow: auto; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin: 0; }}
    p {{ color: var(--muted); margin: 0 0 16px; }}
    .filters {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }}
    button {{ border: 1px solid var(--border); background: var(--panel); padding: 8px 12px; border-radius: 999px; cursor: pointer; }}
    button.active {{ background: #1f2933; color: white; }}
    .chunk-list {{ display: grid; gap: 10px; }}
    .chunk-card {{ border: 1px solid var(--border); background: var(--panel); border-radius: 16px; padding: 14px; cursor: pointer; box-shadow: 0 8px 24px rgba(31, 41, 51, 0.06); }}
    .chunk-card small {{ color: var(--muted); display: block; margin-bottom: 6px; }}
    .page {{ margin-bottom: 32px; border: 1px solid var(--border); border-radius: 24px; padding: 16px; background: rgba(255,255,255,0.75); box-shadow: 0 16px 48px rgba(31, 41, 51, 0.08); }}
    .page-header {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 12px; }}
    .canvas {{ position: relative; border-radius: 18px; overflow: hidden; border: 1px solid var(--border); background: white; }}
    .canvas img {{ width: 100%; display: block; }}
    .overlay {{ position: absolute; inset: 0; }}
    .bbox {{ position: absolute; border: 2px solid; border-radius: 10px; background: rgba(255,255,255,0.08); cursor: pointer; }}
    .bbox.text {{ border-color: var(--text); }}
    .bbox.table {{ border-color: var(--table); }}
    .bbox.figure {{ border-color: var(--figure); }}
    .bbox.discarded {{ border-style: dashed; border-color: var(--discarded); }}
    .detail {{ position: sticky; top: 24px; margin-bottom: 24px; border: 1px solid var(--border); border-radius: 20px; padding: 18px; background: var(--panel); }}
    .detail pre {{ white-space: pre-wrap; word-break: break-word; background: #f8f5ef; padding: 12px; border-radius: 14px; overflow: auto; }}
    .detail img {{ max-width: 100%; border-radius: 14px; border: 1px solid var(--border); }}
    .diff {{ font-family: monospace; font-size:12px; background:#f0ede8; border-radius:14px; overflow:auto; padding:12px; line-height:1.55; white-space:pre-wrap; word-break:break-word; }}
    .diff-add {{ background:#d1fae5; color:#065f46; display:block; }}
    .diff-del {{ background:#fee2e2; color:#991b1b; text-decoration:line-through; display:block; }}
    .diff-ctx {{ color:#6b7280; display:block; }}
    @media (max-width: 980px) {{ .layout {{ grid-template-columns: 1fr; }} .detail {{ position: static; }} }}
  </style>
</head>
<body>
  <div class=\"layout\">
    <aside class=\"sidebar\">
      <h1>__TITLE__</h1>
      <p>ページ画像に対する bbox overlay と raw / structured / corrected 差分を確認できます。</p>
      <div class=\"filters\" id=\"filters\"></div>
      <div class=\"chunk-list\" id=\"chunk-list\"></div>
    </aside>
    <main class=\"content\">
      <section class=\"detail\" id=\"detail\">
        <strong>Chunk を選択すると詳細を表示します。</strong>
      </section>
      <section id=\"pages\"></section>
    </main>
  </div>
  <script type="application/json" id="audit-data">
__PAYLOAD__
  </script>
  <script>
    const audit = JSON.parse(document.getElementById('audit-data').textContent);
    const filters = ["all", "text", "table", "figure", "corrected", "discarded"];
    let currentFilter = "all";

    const chunks = audit.chunks.map((entry, index) => ({
      id: entry.processed.chunk_id,
      index,
      page_num: entry.processed.page_num,
      type: entry.processed.chunk_type,
      status: entry.status,
      raw: entry.raw,
      processed: entry.processed,
      corrected: entry.corrected,
    }));

    function diffLines(oldText, newText) {
        const oldLines = (oldText || "").split("\\n");
        const newLines = (newText || "").split("\\n");
      const m = oldLines.length;
      const n = newLines.length;
      const dp = Array.from({length: m + 1}, () => new Array(n + 1).fill(0));
      for (let i = m - 1; i >= 0; i--) {
        for (let j = n - 1; j >= 0; j--) {
          if (oldLines[i] === newLines[j]) {
            dp[i][j] = dp[i + 1][j + 1] + 1;
          } else {
            dp[i][j] = Math.max(dp[i + 1][j], dp[i][j + 1]);
          }
        }
      }
      const result = [];
      let i = 0, j = 0;
      while (i < m || j < n) {
        if (i < m && j < n && oldLines[i] === newLines[j]) {
          result.push({type: "ctx", text: oldLines[i]});
          i++; j++;
        } else if (j < n && (i >= m || dp[i + 1][j] >= dp[i][j + 1])) {
          result.push({type: "add", text: newLines[j]});
          j++;
        } else {
          result.push({type: "del", text: oldLines[i]});
          i++;
        }
      }
      return result;
    }

    function renderDiff(diff) {
      return diff.map(({type, text}) =>
        `<span class="diff-${type}">${escapeHtml(text)}</span>`
      ).join("");
    }

    function renderFilters() {
      const host = document.getElementById("filters");
      host.innerHTML = "";
      filters.forEach((filter) => {
        const button = document.createElement("button");
        button.textContent = filter;
        button.className = filter === currentFilter ? "active" : "";
        button.onclick = () => {
          currentFilter = filter;
          renderFilters();
          renderChunkList();
          renderPages();
        };
        host.appendChild(button);
      });
    }

    function visibleChunks() {
      return chunks.filter((chunk) => {
        if (currentFilter === "all") return true;
        if (currentFilter === "corrected") return chunk.status === "corrected";
        if (currentFilter === "discarded") return chunk.status === "discarded";
        return chunk.type === currentFilter;
      });
    }

    function renderChunkList() {
      const host = document.getElementById("chunk-list");
      host.innerHTML = "";
      visibleChunks().forEach((chunk) => {
        const card = document.createElement("div");
        card.className = "chunk-card";
        const title = chunk.processed.intuition_summary || chunk.raw.source_preview || "(no summary)";
        card.innerHTML = `<small>p.${chunk.page_num} | ${chunk.type} | ${chunk.status}</small><strong>${escapeHtml(title)}</strong>`;
        card.onclick = () => selectChunk(chunk.id);
        host.appendChild(card);
      });
    }

    function renderPages() {
      const host = document.getElementById("pages");
      host.innerHTML = "";
      audit.pages.forEach((page) => {
        const pageChunks = visibleChunks().filter((chunk) => chunk.page_num === page.page_num);
        const section = document.createElement("section");
        section.className = "page";
        section.innerHTML = `<div class=\"page-header\"><h2>Page ${page.page_num}</h2><small>${pageChunks.length} chunks</small></div>`;

        const canvas = document.createElement("div");
        canvas.className = "canvas";
        const image = document.createElement("img");
        image.src = page.image_path;
        image.alt = `Page ${page.page_num}`;
        canvas.appendChild(image);

        const overlay = document.createElement("div");
        overlay.className = "overlay";
        pageChunks.forEach((chunk) => {
          if (!chunk.raw.bbox || !chunk.raw.page_size.width || !chunk.raw.page_size.height) return;
          const box = document.createElement("div");
          box.className = `bbox ${chunk.type} ${chunk.status === "discarded" ? "discarded" : ""}`;
          const bbox = chunk.raw.bbox;
          box.style.left = `${(bbox.x0 / chunk.raw.page_size.width) * 100}%`;
          box.style.top = `${(bbox.y0 / chunk.raw.page_size.height) * 100}%`;
          box.style.width = `${((bbox.x1 - bbox.x0) / chunk.raw.page_size.width) * 100}%`;
          box.style.height = `${((bbox.y1 - bbox.y0) / chunk.raw.page_size.height) * 100}%`;
          box.title = `${chunk.type} | ${chunk.status}`;
          box.onclick = () => selectChunk(chunk.id);
          overlay.appendChild(box);
        });
        canvas.appendChild(overlay);
        section.appendChild(canvas);
        host.appendChild(section);
      });
    }

    function selectChunk(chunkId) {
      const chunk = chunks.find((item) => item.id === chunkId);
      if (!chunk) return;

      const detail = document.getElementById("detail");
      const issues = chunk.processed.validation?.issues?.length
        ? `- ${chunk.processed.validation.issues.join("\\n- ")}`
        : "No validation issues";
      const figure = chunk.raw.artifact_path
        ? `<p><strong>Figure artifact</strong></p><img src=\"${chunk.raw.artifact_path}\" alt=\"figure\">`
        : "";
      const corrected = chunk.corrected
        ? `<p><strong>Corrected diff</strong></p><div class="diff">${renderDiff(diffLines(chunk.processed.structured_text, chunk.corrected.structured_text))}</div>`
        : "";
      detail.innerHTML = `
        <small>p.${chunk.page_num} | ${chunk.type} | ${chunk.status}</small>
        <h2>${escapeHtml(chunk.processed.intuition_summary || "Chunk detail")}</h2>
        <p><strong>Raw preview</strong></p>
        <pre>${escapeHtml(chunk.raw.source_preview || "")}</pre>
        <p><strong>Structured text</strong></p>
        <pre>${escapeHtml(chunk.processed.structured_text || "")}</pre>
        <p><strong>Validation</strong></p>
        <pre>${escapeHtml(issues)}</pre>
        ${corrected}
        ${figure}
      `;
    }

    function escapeHtml(text) {
      return String(text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    renderFilters();
    renderChunkList();
    renderPages();
    if (chunks.length) {
      selectChunk(chunks[0].id);
    }
  </script>
</body>
</html>
"""
    template = template.replace("{{", "{").replace("}}", "}")
    return template.replace("__TITLE__", title).replace("__PAYLOAD__", payload)
