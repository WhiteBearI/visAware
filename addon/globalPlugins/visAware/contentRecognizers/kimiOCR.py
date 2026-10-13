# Copyright (C) 2026 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Kimi OCR through the OpenAI-compatible Chat Completions API."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any

import addonHandler
from logHandler import log
from PIL import Image

from .._kimiEngine import KimiEngineMixin
from ..kimiConnection import ensureKimiConnection
from ..exceptions import ApiError
from ..openaiCompatible import (
	OpenAICompatibleClient,
	OpenAICompatibleSettings,
	getMessageText,
	parseResponse,
	validateCompletion,
)
from ..kimiProvider import (
	applyKimiThinking,
	isKimiK3Model,
	isKimiK27Model,
)
from ..recogHandler import BaseRecognizer, RecognitionRequest

addonHandler.initTranslation()

KIMI_REQUEST_TIMEOUT = 180
KIMI_MAX_COMPLETION_TOKENS = 32_768
NORMALIZED_COORDINATE_SCALE = 1000

OCR_PROMPT = (
	"Perform OCR only. Return every visible text line in natural reading order. "
	"For each line, return localized text runs that can be clicked accurately. "
	"Use exactly one top-level lines array. Each line object must contain exactly one words array. "
	"Each word object must contain exactly a string text and a four-number box2d array. "
	"For scripts with visible word separators, use words. For scripts without visible word separators, "
	"use the smallest contiguous readable text run you can localize reliably. "
	"Do not describe the image and do not add text that is not visible. "
	"Use bounding boxes in the [ymin, xmin, ymax, xmax] format normalized to 0-1000 "
	"relative to this exact input image. If there is no visible text, return an empty lines array. "
	"Return only the JSON object described by the response format."
)

# Translators: An error message for malformed Kimi OCR structured data.
_MALFORMED_STRUCTURED_DATA_MESSAGE = _("Kimi OCR returned malformed structured data.")
# Translators: Reported when OCR coordinates cannot be mapped because the image size is unavailable.
_MISSING_IMAGE_SIZE_MESSAGE = _("Kimi OCR cannot convert coordinates without the uploaded image size.")

# K3 and K2.7 have stable structured-output support. K2.6 is intentionally
# sent JSON mode and is checked again locally because its strict-schema support
# is less predictable for nested arrays.
OCR_RESPONSE_SCHEMA: dict[str, Any] = {
	"type": "object",
	"properties": {
		"lines": {
			"type": "array",
			"items": {
				"type": "object",
				"properties": {
					"words": {
						"type": "array",
						"items": {
							"type": "object",
							"properties": {
								"text": {"type": "string"},
								"box2d": {
									"type": "array",
									"items": {"type": "number"},
								},
							},
							"required": ["text", "box2d"],
							"additionalProperties": False,
						},
					},
				},
				"required": ["words"],
				"additionalProperties": False,
			},
		},
	},
	"required": ["lines"],
	"additionalProperties": False,
}


class CustomContentRecognizer(KimiEngineMixin, BaseRecognizer):
	"""Recognizes text using Kimi's Chat Completions API."""

	name = "kimiOCR"
	supportsAutomaticRecognition = True
	# Translators: The description of the Kimi OCR engine. "Moonshot AI" is a proper noun
	# and should not be translated.
	description = _("Kimi (Moonshot AI) OCR")

	uploadBase64EncodeImage = True
	uploadImageFormat = "JPEG"
	isStreaming = False

	_uploadedImageSize: tuple[int, int] | None = None

	@property
	def supportedSettings(self) -> list[Any]:
		return [
			*self._kimiConnectionSettings(),
			self.autoRecognitionModelSetting(),
			self._kimiReasoningSetting(),
		]

	@classmethod
	def check(cls) -> bool:
		return True

	def _ensureConnection(
		self,
		forceDiscovery: bool = False,
		cancellationChecker: Callable[[], None] | None = None,
		discoveryTimeout: float | None = None,
	) -> None:
		ensureKimiConnection(
			self,
			forceDiscovery=forceDiscovery,
			cancellationChecker=cancellationChecker,
			discoveryTimeout=discoveryTimeout,
			missingKeyMessage=_("API key is missing. Please configure it in the Kimi OCR engine settings."),
		)

	def _serializeImage(self, pilImage: Image.Image) -> bytes:
		self._uploadedImageSize = pilImage.size
		return super()._serializeImage(pilImage)

	def _buildRequestParams(self, imageContent: bytes, request: RecognitionRequest) -> dict:
		self._ensureConnection(
			cancellationChecker=getattr(request, "cancellationChecker", None),
			discoveryTimeout=25 if getattr(request, "isAutomaticRecognition", False) else None,
		)
		payload: dict[str, Any] = {
			"model": self.model,
			"max_completion_tokens": KIMI_MAX_COMPLETION_TOKENS,
			"messages": [
				{
					"role": "system",
					"content": "You are an OCR engine. Return only the JSON object described by the user.",
				},
				{
					"role": "user",
					"content": [
						{
							"type": "image_url",
							"image_url": {
								"url": f"data:image/jpeg;base64,{imageContent.decode('ascii')}",
							},
						},
						{"type": "text", "text": OCR_PROMPT},
					],
				},
			],
		}
		if isKimiK3Model(self.model) or isKimiK27Model(self.model):
			payload["response_format"] = {
				"type": "json_schema",
				"json_schema": {
					"name": "kimi_ocr_result",
					"strict": True,
					"schema": OCR_RESPONSE_SCHEMA,
				},
			}
		else:
			payload["response_format"] = {"type": "json_object"}
		applyKimiThinking(payload, self.model, self.reasoningEffort)
		client = OpenAICompatibleClient(
			OpenAICompatibleSettings(
				apiKey=self.apiKey,
				baseUrl=self._resolvedBaseUrl or self.baseUrl,
			)
		)
		return client.requestParams(
			payload,
			timeout=KIMI_REQUEST_TIMEOUT,
			stream=request.streamResult,
			cancelCheck=getattr(request, "cancellationChecker", None),
		)

	def processApiResult(self, result: bytes) -> str | bool:
		try:
			self._extractStructuredResult(parseResponse(result))
		except ApiError as e:
			return str(e)
		return False

	def extractText(self, apiResult: dict) -> str:
		structuredResult = self._extractStructuredResult(apiResult)
		return "\n".join(
			" ".join(word["text"].strip() for word in line["words"] if word["text"].strip())
			for line in structuredResult["lines"]
			if any(word["text"].strip() for word in line["words"])
		)

	def _convertToLineResultFormat(self, apiResult: dict) -> list:
		structuredResult = self._extractStructuredResult(apiResult)
		lineResult: list[list[dict[str, Any]]] = []
		for line in structuredResult["lines"]:
			lineWords = [
				convertedWord
				for word in line["words"]
				if (convertedWord := self._convertStructuredWord(word)) is not None
			]
			if lineWords:
				lineResult.append(lineWords)
		return lineResult

	def _cleanup(self) -> None:
		self._uploadedImageSize = None
		super()._cleanup()

	def _extractStructuredResult(self, apiResult: dict[str, Any]) -> dict[str, Any]:
		_choice, message = validateCompletion(apiResult, {"stop"})
		text = getMessageText(message).strip()
		if not text:
			# Translators: An error message for an empty Kimi OCR response.
			raise ApiError(_("Kimi OCR returned an empty response."))
		try:
			structuredResult = json.loads(self._stripMarkdownJsonFence(text))
		except json.JSONDecodeError as e:
			# Translators: An error message for malformed Kimi OCR structured JSON.
			raise ApiError(_("Kimi OCR returned invalid JSON.")) from e
		self._validateStructuredResult(structuredResult)
		return structuredResult

	@staticmethod
	def _stripMarkdownJsonFence(text: str) -> str:
		text = text.strip()
		if not text.startswith("```"):
			return text
		lines = text.splitlines()
		if len(lines) >= 3 and lines[-1].strip() == "```":
			return "\n".join(lines[1:-1]).strip()
		return text

	def _validateStructuredResult(self, structuredResult: Any) -> None:
		if not isinstance(structuredResult, dict) or not isinstance(structuredResult.get("lines"), list):
			raise ApiError(_MALFORMED_STRUCTURED_DATA_MESSAGE)
		for line in structuredResult["lines"]:
			if not isinstance(line, dict) or not isinstance(line.get("words"), list):
				raise ApiError(_MALFORMED_STRUCTURED_DATA_MESSAGE)
			for word in line["words"]:
				if not isinstance(word, dict) or not isinstance(word.get("text"), str):
					raise ApiError(_MALFORMED_STRUCTURED_DATA_MESSAGE)
				box = word.get("box2d")
				if not isinstance(box, list) or len(box) != 4:
					raise ApiError(_MALFORMED_STRUCTURED_DATA_MESSAGE)
				if any(isinstance(coordinate, bool) for coordinate in box):
					raise ApiError(_MALFORMED_STRUCTURED_DATA_MESSAGE)
				try:
					coordinates = [float(coordinate) for coordinate in box]
				except (TypeError, ValueError) as e:
					raise ApiError(_MALFORMED_STRUCTURED_DATA_MESSAGE) from e
				if not all(math.isfinite(coordinate) for coordinate in coordinates):
					raise ApiError(_MALFORMED_STRUCTURED_DATA_MESSAGE)

	def _convertStructuredWord(self, word: dict[str, Any]) -> dict[str, Any] | None:
		text = word["text"].strip()
		if not text:
			return None
		if not self._uploadedImageSize:
			raise ApiError(_MISSING_IMAGE_SIZE_MESSAGE)
		imageWidth, imageHeight = self._uploadedImageSize
		if imageWidth <= 0 or imageHeight <= 0:
			raise ApiError(_MISSING_IMAGE_SIZE_MESSAGE)

		yMin, xMin, yMax, xMax = self._normalizeBox(word["box2d"])
		if xMax <= xMin or yMax <= yMin:
			log.debugWarning(f"Kimi OCR skipped a degenerate box: {word!r}")
			return None
		x = round(xMin / NORMALIZED_COORDINATE_SCALE * imageWidth)
		y = round(yMin / NORMALIZED_COORDINATE_SCALE * imageHeight)
		right = round(xMax / NORMALIZED_COORDINATE_SCALE * imageWidth)
		bottom = round(yMax / NORMALIZED_COORDINATE_SCALE * imageHeight)
		x = max(0, min(x, imageWidth - 1))
		y = max(0, min(y, imageHeight - 1))
		right = max(x + 1, min(right, imageWidth))
		bottom = max(y + 1, min(bottom, imageHeight))
		return {"text": text, "x": x, "y": y, "width": right - x, "height": bottom - y}

	@staticmethod
	def _normalizeBox(box: list[Any]) -> tuple[int, int, int, int]:
		values = [max(0, min(round(float(coordinate)), NORMALIZED_COORDINATE_SCALE)) for coordinate in box]
		yMin, xMin, yMax, xMax = values
		if yMax < yMin:
			yMin, yMax = yMax, yMin
		if xMax < xMin:
			xMin, xMax = xMax, xMin
		return yMin, xMin, yMax, xMax
