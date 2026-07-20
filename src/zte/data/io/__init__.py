"""Dataset acquisition and low-level parsing.

- `zte.data.io.mat_loader` - read a single ZuCo `.mat` file into a plain dict.
- `zte.data.io.sources` - resolve a data source (directory or archive) into extracted `.mat` files.
- `zte.data.io.remote` - Google Drive download/upload and mounting helpers.
- `zte.data.io.drive_download` - resumable Drive-folder download primitives.
"""

from __future__ import annotations

__all__: list[str] = []
