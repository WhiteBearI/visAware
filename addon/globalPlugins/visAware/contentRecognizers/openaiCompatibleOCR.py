# Copyright (C) 2026 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""OCR through an OpenAI-compatible Chat Completions endpoint."""

from __future__ import annotations

import json
import math
from typing import Any

import addonHandler
from logHandler import log
from PIL import Image

from .._modelSettings import KnownModelsMixin
from .._openaiEngine import OpenAICompatibleEngineMixin
from ..engineSettings import TextInputEngineSetting
from ..exceptions import ApiError, AuthenticationError
from ..openaiCompatible import (
	OpenAICompatibleClient,
	OpenAICompatibleSettings,
	getMessageText,
	parseResponse,
	setCompletionTokenLimit,
	validateCompletion,
)
from ..recogHandler import BaseRecognizer, RecognitionRequest

addonHandler.initTranslation()

REQUEST_TIMEOUT = 180
MAX_COMPLETION_TOKENS = 4_096
NORMALIZED_COORDINATE_SCALE = 1000

OCR_PROMPT = (
	"Perform OCR only. Return every visible text line in natural reading order. "
	"For each line, return localized text runs that can be clicked accurately. "
	"Use exactly one top-level lines array. Each line object must contain exactly one words array. "
	"Each word object must contain exactly a string text and a four-number box2d array. "
	"For scripts with visible word separators, use words. For scripts without visible word separators, "
	"use the smallest contiguous readable text run you can localize reliably. "
	"Do not describe the image and do not add text that is not visible. "
	"Return a JSON object with the requested lines array. "
	"Use bounding boxes in the [ymin, xmin, ymax, xmax] format normalized to 0-1000 "
	"relative to this exact input image. If there is no visible text, return an empty lines array."
)
PLAIN_OCR_PROMPT = (
	"Perform OCR only. Return every visible text line in natural reading order. "
	"Do not describe the image or add text that is not visible. Return only the recognized text."
)

_MALFORMED_STRUCTURED_DATA_MESSAGE = _("OpenAI-compatible OCR returned malformed structured data.")
_MISSING_IMAGE_SIZE_MESSAGE = _(
	"OpenAI-compatible OCR cannot convert coordinates without the uploaded image size."
)


class CustomContentRecognizer(OpenAICompatibleEngineMixin, KnownModelsMixin, BaseRecognizer):
	"""Recognizes text using OpenAI-compatible Chat Completions."""

	name = "openaiCompatibleOCR"
	supportsAutomaticRecognition = True
	# Translators: The description of the OpenAI-compatible OCR engine.
	description = _("OpenAI-compatible")
	uploadBase64EncodeImage = True
	uploadImageFormat = "JPEG"
	supportsStreaming = False
	isStreaming = False
	defaultBaseUrl = "https://api.openai.com/v1"
	defaultModel = ""
	_prompt = PLAIN_OCR_PROMPT
	_uploadedImageSize: tuple[int, int] | None = None

	@property
	def supportedSettings(self) -> list[Any]:
		return [
			*self._openAICompatibleSettings(),
			self.autoRecognitionModelSetting(),
			self.imageQualitySetting(),
			TextInputEngineSetting(
				"prompt",
				# Translators: The label for a setting to customize the OCR prompt.
				displayNameWithAccelerator=_("Custom &prompt"),
				multiline=True,
				configKey="promptV2",
			),
			self.autoRecognitionPromptSetting(),
		]

	@property
	def prompt(self) -> str:
		return self._prompt

	@prompt.setter
	def prompt(self, value: str) -> None:
		self._prompt = value.strip() or PLAIN_OCR_PROMPT

	@classmethod
	def check(cls) -> bool:
		return True

	def _serializeImage(self, pilImage: Image.Image) -> bytes:
		self._uploadedImageSize = pilImage.size
		return super()._serializeImage(pilImage)

	def _buildRequestParams(self, imageContent: bytes, request: RecognitionRequest) -> dict:
		self._ensureConfigured()
		structuredResult = not request.textResult
		prompt = getattr(request, "prompt", None) or self.prompt
		if structuredResult:
			if prompt == PLAIN_OCR_PROMPT:
				prompt = OCR_PROMPT
			else:
				prompt = f"{OCR_PROMPT}\nAdditional instruction: {prompt}"
		payload: dict[str, Any] = {
			"model": self.model,
			"messages": [
				{
					"role": "system",
					"content": "You are an OCR engine. Return only the requested OCR result.",
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
						{"type": "text", "text": prompt},
					],
				},
			],
		}
		setCompletionTokenLimit(payload, self.model, MAX_COMPLETION_TOKENS)
		if structuredResult:
			payload["response_format"] = {"type": "json_object"}
		client = OpenAICompatibleClient(
			OpenAICompatibleSettings(apiKey=self.apiKey, baseUrl=self.baseUrl),
		)
		return client.requestParams(
			payload,
			timeout=REQUEST_TIMEOUT,
			stream=False,
			cancelCheck=getattr(request, "cancellationChecker", None),
		)

	def processApiResult(self, result: bytes) -> str | bool:
		try:
			apiResult = parseResponse(result)
			_choice, message = validateCompletion(apiResult, {"stop"})
			if not getMessageText(message).strip():
				return _("Server returned a successful but empty response.")
		except ApiError as e:
			return str(e)
		return False

	def extractText(self, apiResult: dict) -> str:
		_choice, message = validateCompletion(apiResult, {"stop"})
		text = getMessageText(message).strip()
		if not text:
			raise ApiError(_("OpenAI-compatible OCR returned an empty response."))
		return text

	def _convertToLineResultFormat(self, apiResult: dict) -> list:
		_choice, message = validateCompletion(apiResult, {"stop"})
		text = getMessageText(message).strip()
		try:
			structuredResult = json.loads(self._stripMarkdownJsonFence(text))
		except json.JSONDecodeError as e:
			raise ApiError(_MALFORMED_STRUCTURED_DATA_MESSAGE) from e
		self._validateStructuredResult(structuredResult)
		lineResult: list[list[dict[str, Any]]] = []
		for line in structuredResult["lines"]:
			lineWords = [
				converted
				for word in line["words"]
				if (converted := self._convertStructuredWord(word)) is not None
			]
			if lineWords:
				lineResult.append(lineWords)
		return lineResult

	def _cleanup(self) -> None:
		self._uploadedImageSize = None
		super()._cleanup()

	def _ensureConfigured(self) -> None:
		if not self.apiKey:
			raise AuthenticationError(
				_("API key is missing. Please configure it in the OpenAI-compatible OCR settings."),
			)
		if not self.model:
			raise ApiError(
				_("Model is missing. Please configure it in the OpenAI-compatible OCR settings."),
			)

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
			log.debugWarning(f"OpenAI-compatible OCR skipped a degenerate box: {word!r}")
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
