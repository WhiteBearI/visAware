# Copyright (C) 2026 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""User-visible entry point for automatic RapidOCR table recognition."""

from .rapidOCR import RapidOCRTableRecognizer as CustomContentRecognizer

__all__ = ["CustomContentRecognizer"]
