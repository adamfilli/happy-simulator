"""Filename sanitization utilities.

Provides functions to convert arbitrary strings into safe filenames
by replacing problematic characters and enforcing length limits.
"""

import re


def sanitize_filename(filename: str) -> str:
    """Convert a string to a safe filename.

    Replaces characters that are invalid or problematic on common
    filesystems (Windows, macOS, Linux). Strips leading/trailing dots
    and truncates to 255 characters.

    Args:
        filename: The string to sanitize.

    Returns:
        A filesystem-safe version of the input.
    """
    sanitized = re.sub(r'[\/<>|:&;`?*\^%$#@!=+[\]{}(),\"\s]', '_', filename)
    sanitized = re.sub(r'^\.*|\.*$', '', sanitized)
    sanitized = sanitized[:255]
    return sanitized