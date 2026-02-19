# CUDA Programming Guide Study Workflow

## Scope

This repository is used to study `cuda-programming-guide.pdf` with assistant support.

## Environment Requirement

- Always run commands with conda `base`.
- Use `conda run -n base ...` (do not rely on `conda activate`).

Example:

```bash
conda run -n base python --version
```

## Primary Tooling

- Single script: `pdf_study_tool.py`
- Backend: `PyMuPDF` (`fitz`)
- Cache file: `.cache/pdf_study_index.json`

The cache is a custom JSON index built from PDF metadata, TOC-derived section ranges, and page text.

## Agent-Oriented Interface

All commands support deterministic CLI behavior. For LLM agents, prefer `--json`.

### 1) List sections

```bash
conda run -n base python pdf_study_tool.py --pdf cuda-programming-guide.pdf list-sections --json
```

Returns stable `section_id` values, hierarchy metadata, and page ranges.

### 2) Read section (with neighbors)

```bash
conda run -n base python pdf_study_tool.py --pdf cuda-programming-guide.pdf read-section --section-id s0007 --before 1 --after 1 --json
```

Use this to read previous/next sections around a target section.

### 3) Read explicit page range

```bash
conda run -n base python pdf_study_tool.py --pdf cuda-programming-guide.pdf read-pages --page-from 21 --page-to 23 --json
```

Use this when section boundaries are not enough.

### 4) Query phrase

```bash
conda run -n base python pdf_study_tool.py --pdf cuda-programming-guide.pdf query --phrase "warp" --json
```

Optional section-bounded search:

```bash
conda run -n base python pdf_study_tool.py --pdf cuda-programming-guide.pdf query --phrase "warp" --section-id s0007 --json
```

Returns stable `match_id` values to support deterministic follow-up actions.

### 5) Annotate in place

```bash
conda run -n base python pdf_study_tool.py --pdf cuda-programming-guide.pdf annotate --phrase "warp" --match-id m00000 --note "my note"
```

This modifies the original PDF in place by adding:

- highlight annotation on the matched phrase
- text note annotation near the same location

For safe validation before writing:

```bash
conda run -n base python pdf_study_tool.py --pdf cuda-programming-guide.pdf annotate --phrase "warp" --match-id m00000 --note "my note" --dry-run --json
```

## Study Notes Workflow

- Store notes in `notes/`.
