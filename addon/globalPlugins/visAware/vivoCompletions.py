# Copyright (C) 2025 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Vivo BlueLM completions requests and response format."""

import json
import uuid
from typing import Any

import addonHandler
from logHandler import log

from . import vivoAuth
from .exceptions import ApiError, StreamReplacementError

addonHandler.initTranslation()

MODEL = "BlueLM-Vision-Aid"


def buildCompletionsRequest(
	user: str,
	password: str,
	messages: list[dict[str, str]],
	extra: dict[str, Any],
	*,
	stream: bool = False,
) -> dict[str, Any]:
	uri = "/vivogpt/completions/stream" if stream else "/vivogpt/completions"
	requestId = str(uuid.uuid4())
	headers = vivoAuth.genSignHeaders(user, password, "POST", uri, {"requestId": requestId})
	headers["Content-Type"] = "application/json"
	payload = {
		"model": MODEL,
		"sessionId": str(uuid.uuid4()),
		"provider": "vivo",
		"messages": messages,
		"extra": extra,
	}
	url = f"https://api-ai.vivo.com.cn{uri}?requestId={requestId}"
	log.debug(
		f"Vivo completions request excluding message contents: url={url}, model={MODEL}, "
		f"sessionId={payload['sessionId']}, extraKeys={list(extra)}, messages={len(messages)}",
	)
	return {"method": "POST", "url": url, "headers": headers, "json": payload}


def getApiErrorMessage(apiResult: dict) -> str | None:
	if apiResult.get("code") == 0:
		return None
	# Translators: A VIVO API error message. {message} is the service message and {code} is the error code.
	return _("Vivo API error: {message} (Code: {code})").format(
		# Translators: A fallback error message when VIVO does not provide an error description.
		message=apiResult.get("msg", _("Unknown Vivo API error")),
		code=apiResult.get("code"),
	)


def processStreamChunk(chunk: bytes) -> str | None:
	"""
	Extracts text from one VIVO BlueLLM streaming response line.

	:param chunk: One raw line from the streaming HTTP response.
	:returns: New model text, or None when the line does not contain text.
	:raises ApiError: If the stream reports a service error.
	"""
	decodedLine = chunk.decode("utf-8", "ignore").strip()
	if not decodedLine.startswith("data:"):
		return None
	payload = decodedLine[len("data:") :].strip()
	if not payload:
		return None
	if payload == "[DONE]":
		return None
	try:
		data = json.loads(payload)
	except json.JSONDecodeError:
		log.debugWarning(f"VIVO BlueLLM ignored malformed streaming chunk: {decodedLine}")
		return None
	if data.get("code") not in (None, 0):
		errorMessage = getApiErrorMessage(data)
		log.warning(f"VIVO BlueLLM streaming chunk reported API error: {errorMessage}")
		raise ApiError(errorMessage, data.get("code"))
	if data.get("type") == "thinking":
		return None
	message = data.get("message", "")
	if message and data.get("type") not in (None, "", "text"):
		return None
	if not message:
		reply = data.get("reply")
		if reply:
			raise StreamReplacementError(str(reply), replacementText=str(reply))
		return None
	return str(message)


def extractText(apiResult: dict) -> str:
	"""
	Extracts the plain text result from the parsed API response.

	:param apiResult: The parsed JSON dictionary from the API.
	:returns: The recognized text as a single string.
	"""
	streamedText = apiResult.get("streamed_text")
	if streamedText:
		return str(streamedText)

	contentString = apiResult.get("data", {}).get("content", "")
	if not contentString:
		return ""

	try:
		contentList = json.loads(contentString)
		if isinstance(contentList, list):
			if not contentList:
				return ""
			firstItem = contentList[0]
			if isinstance(firstItem, dict):
				return str(firstItem.get("text") or "")
			return str(firstItem)
		if isinstance(contentList, str):
			return contentList
	except json.JSONDecodeError:
		log.warning(f"Could not parse 'data.content' as JSON: {contentString}")
		return contentString

	return contentString
