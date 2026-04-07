"""Parse repository files into structures suitable for chunking."""

from pathlib import Path

from app.db.models import ParsedFile


LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c-header",
    ".hpp": "cpp-header",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".sql": "sql",
    ".md": "markdown",
    ".txt": "text",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".env": "env",
    "": "text",
}


def parse_file(root: Path, file_path: Path) -> ParsedFile:
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    relative_path = file_path.relative_to(root).as_posix()
    lines = content.splitlines()

    return ParsedFile(
        path=relative_path,
        language=LANGUAGE_MAP.get(file_path.suffix.lower(), "text"),
        content=content,
        line_count=len(lines),
    )
