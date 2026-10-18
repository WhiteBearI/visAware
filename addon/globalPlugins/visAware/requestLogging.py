# Copyright (C) 2025 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import config


SENSITIVE_LOG_KEYS = {
	"authorization",
	"app_id",
	"app_key",
	"appid",
	"apikey",
	"api_key",
	"api_secret",
	"apisecret",
	"client_id",
	"client_secret",
	"password",
	"src",
	"signature",
	"token",
	"x_ai_gateway_app_id",
	"x_ai_gateway_signature",
	"x_goog_api_key",
	"x_api_key",
}

SENSITIVE_QUERY_KEYS = {
	"access_token",
	"api_key",
	"apikey",
	"app_key",
	"client_secret",
	"key",
	"token",
}

LARGE_PAYLOAD_KEYS = {
	"audio",
	"content",
	"data",
	"file",
	"image",
	"imagebytes",
	"imagecontent",
	"inline_data",
	"payload",
	"src",
}


def redactForLog(value: Any, keyName: str = "") -> Any:
	"""Returns request parameters safe enough for verbose debug logging."""
	normalizedKeyName = keyName.lower().replace("-", "_")
	if normalizedKeyName in SENSITIVE_LOG_KEYS:
		return "<redacted>"
	if normalizedKeyName in LARGE_PAYLOAD_KEYS and isinstance(value, str):
		return f"<redacted payload: {len(value)} chars>"
	if normalizedKeyName == "image_url" and isinstance(value, str):
		return f"<redacted data URL: {len(value)} chars>"
	if isinstance(value, Mapping):
		return {
			key: redactForLog(
				childValue,
				"image_url" if normalizedKeyName == "image_url" and key == "url" else str(key),
			)
			for key, childValue in value.items()
		}
	if isinstance(value, memoryview):
		return f"<memoryview: {value.nbytes} bytes>"
	if isinstance(value, tuple):
		return tuple(redactForLog(item, keyName) for item in value)
	if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
		return [redactForLog(item, keyName) for item in value]
	if isinstance(value, (bytes, bytearray)):
		return f"<{type(value).__name__}: {len(value)} bytes>"
	if isinstance(value, str) and normalizedKeyName == "url":
		return redactUrlForLog(value)
	if isinstance(value, str) and len(value) > 500:
		return f"{value[:200]}...[{len(value)} chars]"
	return value


def redactUrlForLog(url: str) -> str:
	"""Redacts sensitive URL query parameters in verbose request logs."""
	try:
		parts = urlsplit(url)
	except ValueError:
		return url
	if parts.scheme.lower() == "data":
		return f"<redacted data URL: {len(url)} chars>"
	if not parts.query:
		return url
	query = urlencode(
		[
			(key, "<redacted>" if key.lower() in SENSITIVE_QUERY_KEYS else value)
			for key, value in parse_qsl(parts.query, keep_blank_values=True)
		],
	)
	return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


def isVerboseDebugLoggingEnabled() -> bool:
	try:
		return bool(config.conf["visAwareGeneral"]["verboseDebugLogging"])
	except (KeyError, AttributeError, TypeError):
		return False
