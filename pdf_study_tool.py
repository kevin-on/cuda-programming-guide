#!/usr/bin/env python3
"""Study helper for querying, reading, and annotating a PDF with PyMuPDF."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import fitz


def build_section_ranges(toc: list[list[Any]], page_count: int) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    if not toc:
        return sections

    cleaned = []
    for row in toc:
        if len(row) < 3:
            continue
        level, title, start_page = row[0], str(row[1]).strip(), int(row[2])
        if start_page < 1:
            start_page = 1
        cleaned.append((int(level), title, start_page))

    stack: list[tuple[int, str]] = []
    for i, (level, title, start_page) in enumerate(cleaned):
        end_page = page_count
        for next_level, _, next_start in cleaned[i + 1 :]:
            if next_level <= level:
                end_page = max(start_page, next_start - 1)
                break

        section_id = f"s{i:04d}"
        while stack and stack[-1][0] >= level:
            stack.pop()
        parent_id = stack[-1][1] if stack else None
        stack.append((level, section_id))

        sections.append(
            {
                "id": section_id,
                "ordinal": i,
                "level": level,
                "title": title,
                "start_page": start_page,
                "end_page": end_page,
                "parent_id": parent_id,
            }
        )

    for i, section in enumerate(sections):
        section["previous_id"] = sections[i - 1]["id"] if i > 0 else None
        section["next_id"] = sections[i + 1]["id"] if i < len(sections) - 1 else None
    return sections


def find_section_for_page(sections: list[dict[str, Any]], page_num: int) -> dict[str, Any] | None:
    match = None
    for section in sections:
        if section["start_page"] <= page_num <= section["end_page"]:
            match = section
    return match


def section_by_id(sections: list[dict[str, Any]], section_id: str) -> dict[str, Any] | None:
    for section in sections:
        if section["id"] == section_id:
            return section
    return None


def build_index(pdf_path: Path) -> dict[str, Any]:
    doc = fitz.open(pdf_path)
    try:
        page_count = doc.page_count
        toc = doc.get_toc()
        sections = build_section_ranges(toc, page_count)
        page_text = [doc.load_page(i).get_text("text") for i in range(page_count)]
    finally:
        doc.close()

    stat = pdf_path.stat()
    return {
        "pdf_path": str(pdf_path.resolve()),
        "pdf_size": stat.st_size,
        "pdf_mtime": stat.st_mtime,
        "page_count": page_count,
        "sections": sections,
        "page_text": page_text,
    }


def load_or_build_index(pdf_path: Path, cache_path: Path, rebuild: bool) -> tuple[dict[str, Any], bool]:
    pdf_path = pdf_path.resolve()
    if not rebuild and cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            stat = pdf_path.stat()
            if (
                data.get("pdf_path") == str(pdf_path)
                and data.get("pdf_size") == stat.st_size
                and abs(float(data.get("pdf_mtime", 0.0)) - stat.st_mtime) < 1e-6
            ):
                return data, True
        except Exception:
            pass

    data = build_index(pdf_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data, ensure_ascii=True), encoding="utf-8")
    return data, False


def format_snippet(text: str, phrase: str, start_idx: int, context_chars: int) -> str:
    left = max(0, start_idx - context_chars)
    right = min(len(text), start_idx + len(phrase) + context_chars)
    snippet = " ".join(text[left:right].replace("\n", " ").split())
    phrase_pos = snippet.lower().find(phrase.lower())
    if phrase_pos >= 0:
        end = phrase_pos + len(phrase)
        snippet = f"{snippet[:phrase_pos]}[[{snippet[phrase_pos:end]}]]{snippet[end:]}"
    if left > 0:
        snippet = "..." + snippet
    if right < len(text):
        snippet = snippet + "..."
    return snippet


def maybe_trim(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars] + "...", True
    return text, False


def find_phrase_matches(
    pdf_path: Path,
    index_data: dict[str, Any],
    phrase: str,
    context_chars: int,
    section_id: str | None = None,
) -> list[dict[str, Any]]:
    phrase_lower = phrase.lower()
    sections = index_data.get("sections", [])
    page_text = index_data.get("page_text", [])
    page_count = int(index_data.get("page_count", 0))
    matches: list[dict[str, Any]] = []

    target_section = None
    if section_id is not None:
        target_section = section_by_id(sections, section_id)
        if target_section is None:
            raise ValueError(f"Unknown section_id: {section_id}")
        start_page = target_section["start_page"]
        end_page = target_section["end_page"]
    else:
        start_page = 1
        end_page = page_count

    doc = fitz.open(pdf_path)
    try:
        for page_num in range(start_page, end_page + 1):
            page_idx = page_num - 1
            text = page_text[page_idx]
            text_lower = text.lower()
            if phrase_lower not in text_lower:
                continue

            section = find_section_for_page(sections, page_num)
            rects = doc.load_page(page_idx).search_for(phrase)

            starts = []
            start = text_lower.find(phrase_lower)
            while start != -1:
                starts.append(start)
                start = text_lower.find(phrase_lower, start + 1)

            for i, start_idx in enumerate(starts):
                rect = rects[i] if i < len(rects) else None
                matches.append(
                    {
                        "page": page_num,
                        "on_page_index": i,
                        "section_id": section["id"] if section else None,
                        "section_level": section["level"] if section else None,
                        "section_title": section["title"] if section else None,
                        "snippet": format_snippet(text, phrase, start_idx, context_chars),
                        "rect": [rect.x0, rect.y0, rect.x1, rect.y1] if rect else None,
                    }
                )
    finally:
        doc.close()

    if sections:
        first_section_page = min(section["start_page"] for section in sections)
    else:
        first_section_page = 1

    matches.sort(
        key=lambda m: (
            m["page"] < first_section_page,
            m["section_title"] is None,
            m["page"],
            m["on_page_index"],
        )
    )
    for i, match in enumerate(matches):
        match["match_id"] = f"m{i:05d}"
    return matches


def print_json(data: dict[str, Any]) -> None:
    print(json.dumps(data, ensure_ascii=True))


def cmd_list_sections(args: argparse.Namespace) -> int:
    index_data, cache_hit = load_or_build_index(args.pdf, args.cache, args.rebuild_cache)
    sections = index_data["sections"]
    if args.title_contains:
        term = args.title_contains.lower()
        sections = [s for s in sections if term in s["title"].lower()]
    total = len(sections)
    shown = sections if args.max_results == 0 else sections[: args.max_results]

    payload = {
        "cache_hit": cache_hit,
        "source_pdf": str(args.pdf.resolve()),
        "total_sections": total,
        "returned_sections": len(shown),
        "sections": shown,
    }
    if args.json:
        print_json(payload)
    else:
        print(f"cache {'hit' if cache_hit else 'rebuilt'}: {args.cache}")
        print(f"sections: showing {len(shown)} of {total}")
        for section in shown:
            print(
                f"{section['id']} L{section['level']} "
                f"p{section['start_page']}-{section['end_page']} "
                f"parent={section['parent_id']} {section['title']}"
            )
    return 0


def cmd_read_pages(args: argparse.Namespace) -> int:
    index_data, cache_hit = load_or_build_index(args.pdf, args.cache, args.rebuild_cache)
    page_count = index_data["page_count"]
    if args.page_from < 1 or args.page_to < 1 or args.page_from > page_count or args.page_to > page_count:
        print(f"Invalid page range. Valid pages: 1..{page_count}")
        return 2
    if args.page_from > args.page_to:
        print("Invalid page range: --page-from must be <= --page-to")
        return 2

    sections = index_data["sections"]
    page_text = index_data["page_text"]
    pages = []
    for page_num in range(args.page_from, args.page_to + 1):
        text = page_text[page_num - 1]
        shown_text, truncated = maybe_trim(text, args.max_chars_per_page)
        section = find_section_for_page(sections, page_num)
        pages.append(
            {
                "page": page_num,
                "section_id": section["id"] if section else None,
                "section_title": section["title"] if section else None,
                "char_count": len(text),
                "truncated": truncated,
                "text": shown_text,
            }
        )

    payload = {
        "cache_hit": cache_hit,
        "source_pdf": str(args.pdf.resolve()),
        "page_from": args.page_from,
        "page_to": args.page_to,
        "pages": pages,
    }
    if args.json:
        print_json(payload)
    else:
        print(f"cache {'hit' if cache_hit else 'rebuilt'}: {args.cache}")
        for page in pages:
            print(f"page={page['page']} section={page['section_id']} {page['section_title']}")
            print(page["text"])
            print()
    return 0


def cmd_read_section(args: argparse.Namespace) -> int:
    index_data, cache_hit = load_or_build_index(args.pdf, args.cache, args.rebuild_cache)
    sections = index_data["sections"]
    page_text = index_data["page_text"]

    target = section_by_id(sections, args.section_id)
    if target is None:
        print(f"Unknown section_id: {args.section_id}")
        return 2

    target_idx = target["ordinal"]
    left = max(0, target_idx - args.before)
    right = min(len(sections) - 1, target_idx + args.after)
    selected = sections[left : right + 1]

    out_sections = []
    for section in selected:
        pages = page_text[section["start_page"] - 1 : section["end_page"]]
        full_text = "\n".join(pages)
        shown_text, truncated = maybe_trim(full_text, args.max_chars)
        out_sections.append(
            {
                "section_id": section["id"],
                "title": section["title"],
                "level": section["level"],
                "start_page": section["start_page"],
                "end_page": section["end_page"],
                "parent_id": section["parent_id"],
                "previous_id": section["previous_id"],
                "next_id": section["next_id"],
                "char_count": len(full_text),
                "truncated": truncated,
                "text": shown_text,
            }
        )

    payload = {
        "cache_hit": cache_hit,
        "source_pdf": str(args.pdf.resolve()),
        "target_section_id": args.section_id,
        "before": args.before,
        "after": args.after,
        "sections": out_sections,
    }
    if args.json:
        print_json(payload)
    else:
        print(f"cache {'hit' if cache_hit else 'rebuilt'}: {args.cache}")
        for section in out_sections:
            print(
                f"{section['section_id']} L{section['level']} "
                f"p{section['start_page']}-{section['end_page']} {section['title']}"
            )
            print(section["text"])
            print()
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    index_data, cache_hit = load_or_build_index(args.pdf, args.cache, args.rebuild_cache)
    try:
        matches = find_phrase_matches(
            args.pdf,
            index_data,
            args.phrase,
            args.context_chars,
            section_id=args.section_id,
        )
    except ValueError as error:
        print(str(error))
        return 2

    shown = matches if args.max_results == 0 else matches[: args.max_results]
    payload = {
        "cache_hit": cache_hit,
        "source_pdf": str(args.pdf.resolve()),
        "phrase": args.phrase,
        "section_id": args.section_id,
        "total_matches": len(matches),
        "returned_matches": len(shown),
        "matches": shown,
    }
    if args.json:
        print_json(payload)
    else:
        print(f"cache {'hit' if cache_hit else 'rebuilt'}: {args.cache}")
        print(f"found {len(matches)} matches for phrase: {args.phrase!r}")
        for match in shown:
            section = "Unknown section"
            if match["section_title"]:
                section = f"{match['section_id']} L{match['section_level']} {match['section_title']}"
            print(f"{match['match_id']} page={match['page']} section={section}")
            print(f"    {match['snippet']}")
        if len(matches) > len(shown):
            print(f"... {len(matches) - len(shown)} more matches not shown")
    return 0


def cmd_annotate(args: argparse.Namespace) -> int:
    index_data, cache_hit = load_or_build_index(args.pdf, args.cache, args.rebuild_cache)
    try:
        matches = find_phrase_matches(
            args.pdf,
            index_data,
            args.phrase,
            args.context_chars,
            section_id=args.section_id,
        )
    except ValueError as error:
        print(str(error))
        return 2

    target = None
    for match in matches:
        if match["match_id"] == args.match_id:
            target = match
            break

    if target is None:
        print(f"Unknown match_id: {args.match_id}")
        return 2
    if target["rect"] is None:
        print("Matched text did not yield a geometric location for annotation.")
        return 2

    if not args.dry_run:
        doc = fitz.open(args.pdf)
        try:
            page = doc.load_page(target["page"] - 1)
            rect = fitz.Rect(target["rect"])

            highlight = page.add_highlight_annot(rect)
            highlight.update()

            note_pos = fitz.Point(rect.x0, max(0, rect.y0 - 14))
            note = page.add_text_annot(note_pos, args.note)
            note.set_info(title="Study Note")
            note.update()

            if doc.can_save_incrementally():
                doc.saveIncr()
            else:
                doc.save(args.pdf)
        finally:
            doc.close()

    payload = {
        "cache_hit": cache_hit,
        "source_pdf": str(args.pdf.resolve()),
        "phrase": args.phrase,
        "match_id": args.match_id,
        "note": args.note,
        "dry_run": args.dry_run,
        "annotated": not args.dry_run,
        "target": target,
    }
    if args.json:
        print_json(payload)
    else:
        section = "Unknown section"
        if target["section_title"]:
            section = f"{target['section_id']} L{target['section_level']} {target['section_title']}"
        print(f"cache {'hit' if cache_hit else 'rebuilt'}: {args.cache}")
        print(f"target={args.match_id} page={target['page']} section={section}")
        if args.dry_run:
            print("dry-run only: no annotation written")
        else:
            print(f"annotation added in-place: {args.pdf}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query, read, and annotate a PDF for study workflows.")
    parser.add_argument("--pdf", type=Path, required=True, help="Path to the target PDF.")
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(".cache/pdf_study_index.json"),
        help="Path to small JSON cache index.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Force cache rebuild even if the PDF metadata matches.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_sections = subparsers.add_parser("list-sections", help="List section metadata with stable IDs.")
    list_sections.add_argument("--title-contains", help="Filter sections where title includes this text.")
    list_sections.add_argument("--max-results", type=int, default=0, help="0 means return all sections.")
    list_sections.add_argument("--json", action="store_true", help="Return machine-readable JSON.")
    list_sections.set_defaults(func=cmd_list_sections)

    read_pages = subparsers.add_parser("read-pages", help="Read page text for an explicit page range.")
    read_pages.add_argument("--page-from", type=int, required=True, help="Start page (1-based, inclusive).")
    read_pages.add_argument("--page-to", type=int, required=True, help="End page (1-based, inclusive).")
    read_pages.add_argument(
        "--max-chars-per-page",
        type=int,
        default=0,
        help="Trim page text to this length; 0 means no trimming.",
    )
    read_pages.add_argument("--json", action="store_true", help="Return machine-readable JSON.")
    read_pages.set_defaults(func=cmd_read_pages)

    read_section = subparsers.add_parser("read-section", help="Read a section with optional neighbors.")
    read_section.add_argument("--section-id", required=True, help="Section ID from list-sections.")
    read_section.add_argument("--before", type=int, default=0, help="How many previous sections to include.")
    read_section.add_argument("--after", type=int, default=0, help="How many next sections to include.")
    read_section.add_argument(
        "--max-chars",
        type=int,
        default=0,
        help="Trim each section text to this length; 0 means no trimming.",
    )
    read_section.add_argument("--json", action="store_true", help="Return machine-readable JSON.")
    read_section.set_defaults(func=cmd_read_section)

    query = subparsers.add_parser("query", help="Find phrase matches with section-aware context.")
    query.add_argument("--phrase", required=True, help="Phrase to search for.")
    query.add_argument("--section-id", help="Optional section boundary for search.")
    query.add_argument("--max-results", type=int, default=10, help="0 means return all matches.")
    query.add_argument(
        "--context-chars",
        type=int,
        default=220,
        help="Number of context characters around each phrase hit.",
    )
    query.add_argument("--json", action="store_true", help="Return machine-readable JSON.")
    query.set_defaults(func=cmd_query)

    annotate = subparsers.add_parser("annotate", help="Add highlight + note annotation in-place.")
    annotate.add_argument("--phrase", required=True, help="Phrase used to generate match IDs.")
    annotate.add_argument("--match-id", required=True, help="Target match ID from query output.")
    annotate.add_argument("--note", required=True, help="Annotation note content.")
    annotate.add_argument("--section-id", help="Optional section boundary used during query.")
    annotate.add_argument(
        "--context-chars",
        type=int,
        default=220,
        help="Context chars for deterministic match generation.",
    )
    annotate.add_argument("--dry-run", action="store_true", help="Validate target without writing.")
    annotate.add_argument("--json", action="store_true", help="Return machine-readable JSON.")
    annotate.set_defaults(func=cmd_annotate)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.pdf.exists():
        print(f"PDF not found: {args.pdf}")
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
