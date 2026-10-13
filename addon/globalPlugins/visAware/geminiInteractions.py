# Copyright (C) 2026 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Request construction for Google's Interactions protocol."""

from typing import Any

INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
INTERACTIONS_API_REVISION = "2026-05-20"


def buildInteractionsRequest(apiKey: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
	return {
		"method": "POST",
		"url": INTERACTIONS_URL,
		"headers": {
			"Content-Type": "application/json",
			"Api-Revision": INTERACTIONS_API_REVISION,
			"x-goog-api-key": apiKey,
		},
		"json": payload,
		"timeout": timeout,
	}
