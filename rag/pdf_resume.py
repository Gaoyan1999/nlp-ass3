"""Convert uploaded PDF resumes into simple Markdown."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path


def pdf_bytes_to_markdown(pdf_bytes: bytes, filename: str = "resume.pdf") -> str:
    """Extract text from PDF bytes and format it as simple Markdown."""
    if not pdf_bytes:
        raise ValueError("Uploaded PDF is empty.")
    if not filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF resumes are supported.")

    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = Path(temp_dir) / "resume.pdf"
        text_path = Path(temp_dir) / "resume.txt"
        pdf_path.write_bytes(pdf_bytes)

        subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), str(text_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        text = text_path.read_text(encoding="utf-8", errors="replace")

    markdown = text_to_markdown(text)
    if len(markdown) < 30:
        raise ValueError("Could not extract enough text from the PDF resume.")
    return markdown


def text_to_markdown(text: str) -> str:
    """Lightly structure extracted resume text as Markdown."""
    lines = [line.rstrip() for line in text.replace("\f", "\n").splitlines()]
    output: list[str] = []
    previous_blank = True

    for raw_line in lines:
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            if not previous_blank:
                output.append("")
            previous_blank = True
            continue

        if _looks_like_heading(line):
            if output and output[-1] != "":
                output.append("")
            output.append(f"## {line}")
        elif line.startswith(("•", "-", "*")):
            output.append(f"- {line.lstrip('•-* ').strip()}")
        else:
            output.append(line)
        previous_blank = False

    return "\n".join(output).strip()


def _looks_like_heading(line: str) -> bool:
    if len(line) > 48 or line.endswith((".", ",", ";", ":")):
        return False
    words = line.split()
    if len(words) > 5:
        return False
    known_headings = {
        "education",
        "skills",
        "work experience",
        "experience",
        "projects",
        "summary",
        "certifications",
        "awards",
    }
    return line.casefold() in known_headings
