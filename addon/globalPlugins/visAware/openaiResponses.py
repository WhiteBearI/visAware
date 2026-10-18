# Copyright (C) 2026 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Transport helpers for the OpenAI-compatible Responses protocol."""

from __future__ import annotations

from typing import Any

import addonHandler

from . import network
from .exceptions import ApiError

addonHandler.initTranslation()


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


class ResponsesClient:
	"""Small transport boundary shared by Responses API clients."""

	def __init__(self, apiKey: str, baseUrl: str, transport: Any = None) -> None:
		self.apiKey = apiKey.strip()
		self.baseUrl = (baseUrl or "").rstrip("/")
		self._transport = transport or network

	def requestParams(
		self,
		payload: dict[str, Any],
		*,
		timeout: float,
		cancelCheck: Any = None,
	) -> dict[str, Any]:
		params = buildResponsesRequest(self.apiKey, self.baseUrl, payload, timeout=timeout)
		if callable(cancelCheck):
			params["cancelCheck"] = cancelCheck
		return params

	def create(
		self,
		payload: dict[str, Any],
		*,
		timeout: float,
		cancelCheck: Any = None,
	) -> dict[str, Any]:
		response = self._transport.sendRequest(
			**self.requestParams(payload, timeout=timeout, cancelCheck=cancelCheck),
		)
		result = response.json()
		if not isinstance(result, dict):
			raise ApiError(_("Responses API returned an invalid response."))
		error = result.get("error")
		if error:
			message = error.get("message") if isinstance(error, dict) else error
			raise ApiError(_("Responses API error: {}").format(message or str(error)))
		return result

	def listModels(self, *, timeout: float = 30, cancelCheck: Any = None) -> list[str]:
		response = self._transport.sendRequest(
			"GET",
			f"{self.baseUrl}/models",
			headers={"Authorization": f"Bearer {self.apiKey}"},
			timeout=timeout,
			cancelCheck=cancelCheck,
		)
		result = response.json()
		if not isinstance(result, dict):
			return []
		models = result.get("data")
		if not isinstance(models, list):
			return []
		return sorted(
			{
				item["id"].strip()
				for item in models
				if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"].strip()
			},
		)
