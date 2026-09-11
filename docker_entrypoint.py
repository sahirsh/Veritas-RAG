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
    except (KeyError, OSError, PermissionError) as exc:
        print(f"Warning: could not chown {upload_dir}: {exc}", flush=True)


def _drop_to_appuser() -> None:
    """Drop root privileges when possible (local Docker). Skip on Cloud Run non-root."""
    if os.geteuid() != 0:
        print("Already running as non-root; skipping privilege drop.", flush=True)
        return

    pw = pwd.getpwnam("appuser")
    os.setgid(pw.pw_gid)
    os.setuid(pw.pw_uid)

    os.environ["HOME"] = pw.pw_dir
    os.environ["USER"] = pw.pw_name
    os.environ["LOGNAME"] = pw.pw_name


def main() -> None:
    _ensure_upload_dir_writable()

    if os.environ.get("DATABASE_URL"):
        print("Running database migrations...", flush=True)
        try:
            subprocess.check_call([sys.executable, "-m", "alembic", "upgrade", "head"])
        except subprocess.CalledProcessError as exc:
            # Still start the API so Cloud Run can bind PORT; /health will show DB issues.
            print(f"Warning: migrations failed (exit {exc.returncode}); starting API anyway.", flush=True)
    else:
        print("DATABASE_URL is not set; skipping migrations.", flush=True)

    _drop_to_appuser()

    # Cloud Run injects PORT (typically 8080). Do not hardcode 8000 in Cloud Run env.
    port = os.environ.get("PORT", "8080")
    print(f"Starting uvicorn on 0.0.0.0:{port}", flush=True)
    os.execvp(
        "uvicorn",
        ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", port],
    )


if __name__ == "__main__":
    main()
