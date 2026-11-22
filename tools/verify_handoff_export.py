#!/usr/bin/env python3
"""Verify the exported research snapshot without importing ML packages."""

import hashlib
import json
from pathlib import Path, PurePosixPath


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'research/EXPORT_MANIFEST.json').read_text())
    errors = []
    for row in manifest['files']:
        rel = PurePosixPath(row['path'])
        if rel.is_absolute() or '..' in rel.parts:
            errors.append(f"unsafe path: {row['path']}")
            continue
        path = root / str(rel)
        if row['kind'] == 'symlink':
            if not path.is_symlink() or str(path.readlink()) != row['target']:
                errors.append(f"symlink differs: {rel}")
            continue
        if not path.is_file() or path.is_symlink():
            errors.append(f"missing file: {rel}")
            continue
        raw = path.read_bytes()
        if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
            errors.append(f"content differs: {rel}")
        if bool(path.stat().st_mode & 0o111) != row['executable']:
            errors.append(f"executable bit differs: {rel}")
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f"Verified {len(manifest['files'])} entries from the research export manifest")


if __name__ == '__main__':
    main()
