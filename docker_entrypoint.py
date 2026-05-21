"""Container entrypoint: fix upload dir permissions, migrate, drop to appuser, start Uvicorn."""

from __future__ import annotations

import os
import pwd
import subprocess
import sys


def _ensure_upload_dir_writable() -> None:
    """Docker volumes are often root-owned; the API runs as appuser."""
    upload_dir = os.environ.get("UPLOAD_DIR", "/app/uploads")
    os.makedirs(upload_dir, exist_ok=True)
    try:
        pw = pwd.getpwnam("appuser")
        os.chown(upload_dir, pw.pw_uid, pw.pw_gid)
        for root, dirnames, filenames in os.walk(upload_dir):
            os.chown(root, pw.pw_uid, pw.pw_gid)
            for name in dirnames:
                path = os.path.join(root, name)
                os.chown(path, pw.pw_uid, pw.pw_gid)
            for name in filenames:
                path = os.path.join(root, name)
                os.chown(path, pw.pw_uid, pw.pw_gid)
    except (KeyError, OSError) as exc:
        print(f"Warning: could not chown {upload_dir}: {exc}", flush=True)


def _drop_to_appuser() -> None:
    pw = pwd.getpwnam("appuser")
    os.setgid(pw.pw_gid)
    os.setuid(pw.pw_uid)


def main() -> None:
    _ensure_upload_dir_writable()

    if os.environ.get("DATABASE_URL"):
        print("Running database migrations...", flush=True)
        subprocess.check_call([sys.executable, "-m", "alembic", "upgrade", "head"])
    else:
        print("DATABASE_URL is not set; skipping migrations.", flush=True)

    _drop_to_appuser()

    port = os.environ.get("PORT", "8000")
    os.execvp(
        "uvicorn",
        ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", port],
    )


if __name__ == "__main__":
    main()
