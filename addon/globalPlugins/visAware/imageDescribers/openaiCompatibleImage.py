# Copyright (C) 2026 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Image description through an OpenAI-compatible Chat Completions endpoint."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from copy import deepcopy
from typing import Any

import addonHandler

from .. import network
from .._modelSettings import KnownModelsMixin
from .._openaiEngine import OpenAICompatibleEngineMixin
from ..conversation import QuestionStreamFinished, QuestionStreamText
from ..engineSettings import BooleanEngineSetting, TextInputEngineSetting
from ..exceptions import ApiError, AuthenticationError, StreamIncompleteError
from ..openaiCompatible import (
	ChatCompletionStreamState,
	OpenAICompatibleClient,
	OpenAICompatibleSettings,
	extractAssistantMessage,
	getMessageText,
	parseResponse,
	setCompletionTokenLimit,
	validateCompletion,
)
from ..recogHandler import BaseDescriber, RecognitionRequest
from ._prompts import DEFAULT_IMAGE_DESCRIPTION_PROMPT

addonHandler.initTranslation()

REQUEST_TIMEOUT = 180
MAX_COMPLETION_TOKENS = 4_096
DEFAULT_PROMPT = DEFAULT_IMAGE_DESCRIPTION_PROMPT


class CustomContentRecognizer(OpenAICompatibleEngineMixin, KnownModelsMixin, BaseDescriber):
	"""Describes images using OpenAI-compatible Chat Completions."""

	name = "openaiCompatibleImage"
	supportsAutomaticRecognition = True
	# Translators: The description of the OpenAI-compatible image description engine.
	description = _("OpenAI-compatible")
	uploadBase64EncodeImage = True
	uploadImageFormat = "JPEG"
	supportsStreaming = True
	supportsQuestions = True
	supportsQuestionStreaming = True
	isStreaming = False
	defaultBaseUrl = "https://api.openai.com/v1"
	defaultModel = ""
	_useStreaming = False
	_prompt = DEFAULT_PROMPT

	@property
	def supportedSettings(self) -> list[Any]:
		return [
			*self._openAICompatibleSettings(),
			self.autoRecognitionModelSetting(),
			BooleanEngineSetting(
				"useStreaming",
				# Translators: The label for an engine setting to enable streaming output.
				_("&Enable streaming output (if available)"),
			),
			self.imageQualitySetting(),
			TextInputEngineSetting(
				"prompt",
				# Translators: The label for a setting to customize the image description prompt.
				displayNameWithAccelerator=_("Custom &prompt"),
				multiline=True,
				configKey="promptV2",
			),
			self.autoRecognitionPromptSetting(),
		]

	@property
	def useStreaming(self) -> bool:
		return self._useStreaming

	@useStreaming.setter
	def useStreaming(self, value: bool) -> None:
		self._useStreaming = bool(value)
		self.isStreaming = self._useStreaming

	@property
	def prompt(self) -> str:
		return self._prompt

	@prompt.setter
	def prompt(self, value: str) -> None:
		self._prompt = value.strip() or DEFAULT_PROMPT

	@classmethod
	def check(cls) -> bool:
		return True

	def _buildRequestParams(self, imageContent: bytes, request: RecognitionRequest) -> dict:
		self._ensureConfigured()
		payload = {
			"model": self.model,
			"messages": [self._buildInitialUserMessage(imageContent, getattr(request, "prompt", None))],
		}
		setCompletionTokenLimit(payload, self.model, MAX_COMPLETION_TOKENS)
		client = OpenAICompatibleClient(
			OpenAICompatibleSettings(apiKey=self.apiKey, baseUrl=self.baseUrl),
		)
		return client.requestParams(
			payload,
			timeout=REQUEST_TIMEOUT,
			stream=request.streamResult,
			cancelCheck=getattr(request, "cancellationChecker", None),
		)

	def _buildInitialUserMessage(self, imageContent: bytes, prompt: str | None = None) -> dict[str, Any]:
		return {
			"role": "user",
			"content": [
				{
					"type": "image_url",
					"image_url": {
						"url": f"data:image/jpeg;base64,{imageContent.decode('ascii')}",
					},
				},
				{"type": "text", "text": prompt if prompt is not None else self.prompt},
			],
		}

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
		message = extractAssistantMessage(apiResult)
		return getMessageText(message).strip() if message else ""

	def _resetStreamingState(self) -> None:
		self._streamState = ChatCompletionStreamState()
		self._lastQuestionResponse = None

	def processStreamChunk(self, chunk: bytes, request: RecognitionRequest) -> str | None:
		return self._streamState.consume(chunk)

	def _getStreamingResponse(self, fullResponseText: str) -> dict[str, Any]:
		response = self._streamState.buildResponse()
		_choice, message = validateCompletion(response, {"stop"})
		if not getMessageText(message).strip():
			raise StreamIncompleteError(
				_("OpenAI-compatible service returned a successful but empty response."),
				partialText=fullResponseText,
			)
		self._lastQuestionResponse = deepcopy(response)
		return response

	def askQuestion(
		self,
		context: Any,
		question: str,
		cancellationChecker: Callable[[], None] | None = None,
	) -> str:
		self._checkQuestionCancelled(cancellationChecker)
		response = network.sendRequest(
			**self._buildQuestionRequestParams(context, question, False, cancellationChecker),
		)
		self._checkQuestionCancelled(cancellationChecker)
		apiResult = parseResponse(response.content)
		_choice, message = validateCompletion(apiResult, {"stop"})
		answer = self._validateQuestionAnswer(getMessageText(message))
		self._lastQuestionResponse = deepcopy(apiResult)
		return answer

	def askQuestionStream(
		self,
		context: Any,
		question: str,
		cancellationChecker: Callable[[], None] | None = None,
	) -> Iterator[QuestionStreamText | QuestionStreamFinished]:
		self._checkQuestionCancelled(cancellationChecker)
		requestParams = self._buildQuestionRequestParams(context, question, True, cancellationChecker)
		yield from self._iterQuestionStreamingResponse(
			requestParams,
			RecognitionRequest(textResult=True, streamResult=True),
			cancellationChecker,
		)

	def _buildQuestionRequestParams(
		self,
		context: Any,
		question: str,
		stream: bool,
		cancellationChecker: Callable[[], None] | None = None,
	) -> dict:
		self._ensureConfigured()
		imageContent = self.prepareImageContentFromImage(self._getConversationImage(context))
		messages = [self._buildInitialUserMessage(imageContent)]
		messages.append(
			self._assistantMessage(getattr(context, "response", None), getattr(context, "initialText", ""))
		)
		for turn in getattr(context, "turns", []) or []:
			role = getattr(turn, "role", "")
			text = str(getattr(turn, "text", "")).strip()
			if role == "assistant":
				messages.append(self._assistantMessage(getattr(turn, "response", None), text))
			elif role == "user" and text:
				messages.append({"role": "user", "content": text})
		messages.append({"role": "user", "content": question})
		payload = {"model": self.model, "messages": messages}
		setCompletionTokenLimit(payload, self.model, MAX_COMPLETION_TOKENS)
		client = OpenAICompatibleClient(
			OpenAICompatibleSettings(apiKey=self.apiKey, baseUrl=self.baseUrl),
		)
		return client.requestParams(
			payload,
			timeout=REQUEST_TIMEOUT,
			stream=stream,
			cancelCheck=cancellationChecker,
		)

	@staticmethod
	def _assistantMessage(response: Any, text: str) -> dict[str, Any]:
		message = extractAssistantMessage(response)
		if message is not None:
			return {"role": "assistant", "content": getMessageText(message)}
		return {"role": "assistant", "content": str(text)}

	def _consumeQuestionResponse(self) -> dict[str, Any] | None:
		response = getattr(self, "_lastQuestionResponse", None)
		self._lastQuestionResponse = None
		return deepcopy(response) if isinstance(response, dict) else None

	def _ensureConfigured(self) -> None:
		if not self.apiKey:
			raise AuthenticationError(
				_("API key is missing. Please configure it in the OpenAI-compatible image settings."),
			)
		if not self.model:
			raise ApiError(
				_("Model is missing. Please configure it in the OpenAI-compatible image settings."),
			)
