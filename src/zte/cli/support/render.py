from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def dataframe_to_markdown(frame: 'pd.DataFrame') -> str:
    """Renders a DataFrame as a Markdown table.

    Args:
        frame (pd.DataFrame): The table to render; column order is preserved.

    Returns:
        str: The table as Markdown (header row, separator row, one row per record).

    """
    cols = list(frame.columns)
    head = '| ' + ' | '.join(cols) + ' |\n| ' + ' | '.join(['---'] * len(cols)) + ' |\n'
    body = ''.join('| ' + ' | '.join(str(v) for v in row) + ' |\n' for row in frame.to_numpy())
    return head + body
