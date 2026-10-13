# Copyright (C) 2026 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Request construction for the OpenAI-compatible Responses protocol."""

from typing import Any


def buildResponsesUrl(baseUrl: str) -> str:
	baseUrl = baseUrl.rstrip("/")
	return baseUrl if baseUrl.endswith("/responses") else f"{baseUrl}/responses"


def buildResponsesRequest(
	apiKey: str,
	baseUrl: str,
	payload: dict[str, Any],
	*,
	timeout: float,
) -> dict[str, Any]:
	return {
		"method": "POST",
		"url": buildResponsesUrl(baseUrl),
		"headers": {"Authorization": f"Bearer {apiKey}", "Content-Type": "application/json"},
		"json": payload,
		"timeout": timeout,
	}
