"""Filter unsupported, generated, or irrelevant files before indexing."""

from pathlib import Path


IGNORED_DIR_NAMES = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    "local_docs",
    "node_modules",
    "dist",
    "build",
    "target",
    ".next",
    ".venv",
    "venv",
}

IGNORED_FILE_NAMES = {
    ".ds_store",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
}

IGNORED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".obj",
    ".class",
    ".pyc",
}

SUPPORTED_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".cs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".kts",
    ".scala",
    ".sql",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".env",
}


def should_skip_directory(directory: Path) -> bool:
    return directory.name.lower() in IGNORED_DIR_NAMES


def should_index_file(file_path: Path) -> bool:
    name = file_path.name.lower()
    suffix = file_path.suffix.lower()

    if name in IGNORED_FILE_NAMES:
        return False

    if suffix in IGNORED_EXTENSIONS:
        return False

    if suffix in SUPPORTED_EXTENSIONS:
        return True

    return suffix == ""
