# Copyright (C) 2025 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""An image description engine that uses the VIVO BlueLLM Vision model."""

import addonHandler
import json
from collections import OrderedDict
from collections.abc import Callable, Iterator
from typing import List, Any
from logHandler import log

from .. import network
from ..conversation import QuestionStreamEvent
from ..recogHandler import BaseDescriber, RecognitionRequest
from ..exceptions import ApiError
from .. import vivoAuth, vivoCompletions
from ..engineSettings import (
	BooleanEngineSetting,
	ChoiceEngineSetting,
	TextInputEngineSetting,
	NumericEngineSetting,
)
from ._prompts import DEFAULT_IMAGE_DESCRIPTION_PROMPT

addonHandler.initTranslation()

VIVO_BLUE_LLM_MAX_TOKENS = 1024


class CustomContentRecognizer(BaseDescriber):
	"""An image description engine that uses the VIVO BlueLLM Vision model."""

	name = "vivoBlueLLM"
	supportsAutomaticRecognition = True
	# Translators: The description of the VIVO BlueLLM engine.
	description = _("VIVO BlueLLM Vision (NVDACN)")

	# --- Engine State ---
	supportsStreaming = True
	supportsQuestions = True
	supportsQuestionStreaming = True
	isStreaming = False
	uploadBase64EncodeImage = True
	uploadImageFormat = "JPEG"

	# --- Engine Default Settings ---
	_useStreaming: bool = False
	# Translators: This is the default prompt sent to the VIVO BlueLLM model.
	# It guides the model to provide an objective description.
	_prompt: str = DEFAULT_IMAGE_DESCRIPTION_PROMPT
	_temperature: int = 70  # Using an integer range 0-100 for slider control
	_thinkingLevel: str = "off"

	@property
	def supportedSettings(self) -> List[Any]:
		"""
		Define the user-configurable settings for this engine.

		:returns: A list of `EngineSetting` objects.
		"""
		temperatureSetting = NumericEngineSetting(
			name="temperature",
			# Translators: Label for the AI model temperature setting.
			displayNameWithAccelerator=_("&Temperature"),
		)
		temperatureSetting.minVal = 0
		temperatureSetting.maxVal = 200  # Corresponds to API's 0.0 to 2.0
		temperatureSetting.configSpec = "integer(default=70,min=0,max=200)"

		return [
			BooleanEngineSetting(
				"useStreaming",
				# Translators: The label for an engine setting to enable streaming output.
				_("&Enable streaming output (if available)"),
			),
			ChoiceEngineSetting(
				name="thinkingLevel",
				# Translators: The label for a setting to control model thinking mode.
				displayNameWithAccelerator=_("Thinking &level"),
				optionsPropertyName="availableThinkingLevels",
			),
			TextInputEngineSetting(
				"prompt",
				# Translators: The label for an engine setting to customize the prompt for the vision model.
				_("Custom &prompt"),
				multiline=True,
				configKey="promptV2",
			),
			self.autoRecognitionPromptSetting(),
			temperatureSetting,
		]

	@property
	def useStreaming(self) -> bool:
		return self._useStreaming

	@useStreaming.setter
	def useStreaming(self, value: bool) -> None:
		"""
		Sets whether streaming is enabled for simple text output.

		:param value: The new boolean value.
		"""
		self._useStreaming = value
		self.isStreaming = value

	@property
	def prompt(self) -> str:
		return self._prompt

	@prompt.setter
	def prompt(self, value: str) -> None:
		self._prompt = value

	@property
	def temperature(self) -> int:
		return self._temperature

	@temperature.setter
	def temperature(self, value: int) -> None:
		self._temperature = value

	@property
	def thinkingLevel(self) -> str:
		return self._thinkingLevel

	@thinkingLevel.setter
	def thinkingLevel(self, value: str) -> None:
		if value in self.availableThinkingLevels:
			self._thinkingLevel = value

	@property
	def availableThinkingLevels(self) -> dict:
		"""
		Provides the items for the thinking level choice control in the settings UI.

		:returns: A dictionary of thinking level IDs to display names.
		"""
		levels = OrderedDict(
			{
				"off": _("Off (faster)"),
				"high": _("High (better reasoning, slower)"),
			},
		)
		return self.generateStringSettings(levels)

	@classmethod
	def check(cls) -> bool:
		"""
		Checks if the engine is available.

		:returns: Always True as it's a cloud service.
		"""
		return True

	def _buildExtraParams(self) -> dict[str, float | bool | int]:
		return {
			"temperature": self.temperature / 100.0,
			"enable_thinking": self.thinkingLevel != "off",
			"max_tokens": VIVO_BLUE_LLM_MAX_TOKENS,
		}

	def _buildRequestParamsFromMessages(
		self,
		messages: list[dict[str, str]],
		request: RecognitionRequest,
	) -> dict:
		"""
		Builds VIVO request parameters from a message list.

		:param messages: VIVO-compatible message objects.
		:param request: Request-local recognition options.
		:returns: A dictionary of parameters for `requests`.
		"""
		user, password = vivoAuth.getNvdacnCredentials()
		return vivoCompletions.buildCompletionsRequest(
			user,
			password,
			messages,
			self._buildExtraParams(),
			stream=request.streamResult,
		)

	def _buildRequestParams(self, imageContent: bytes, request: RecognitionRequest) -> dict:
		"""
		Builds the parameters dictionary for the network request.

		:param imageContent: The raw byte content of the image, base64 encoded.
		:returns: A dictionary of parameters for `requests`.
		:raises AuthenticationError: If NVDACN credentials are not configured.
		"""
		imageBase64 = imageContent.decode("utf-8")
		return self._buildRequestParamsFromMessages(
			[
				{
					"role": "user",
					"content": f"data:image/{self.uploadImageFormat};base64,{imageBase64}",
					"contentType": "image",
				},
				{
					"role": "user",
					"content": getattr(request, "prompt", None) or self.prompt,
					"contentType": "text",
				},
			],
			request,
		)

	def processStreamChunk(self, chunk: bytes, request: RecognitionRequest) -> str | None:
		return vivoCompletions.processStreamChunk(chunk)

	def askQuestion(
		self,
		context: Any,
		question: str,
		cancellationChecker: Callable[[], None] | None = None,
	) -> str:
		"""
		Asks VIVO BlueLLM a follow-up question about the previous image description.

		:param context: The conversation context created from recognition history.
		:param question: The user's follow-up question.
		:param cancellationChecker: Optional callback that raises when cancelled.
		:returns: VIVO BlueLLM's answer.
		"""
		self._checkQuestionCancelled(cancellationChecker)
		messages = self._buildQuestionMessages(context, question)
		self._checkQuestionCancelled(cancellationChecker)
		requestParams = self._buildRequestParamsFromMessages(
			messages,
			RecognitionRequest(textResult=True, streamResult=False),
		)
		self._checkQuestionCancelled(cancellationChecker)
		response = network.sendRequest(**requestParams)
		self._checkQuestionCancelled(cancellationChecker)
		apiErrorMessage = self.processApiResult(response.content)
		if apiErrorMessage:
			raise ApiError(str(apiErrorMessage))
		apiResult = self._convertToJson(response.content)
		return self._validateQuestionAnswer(self.extractText(apiResult))

	def _buildQuestionMessages(self, context: Any, question: str) -> list[dict[str, str]]:
		"""
		Builds a VIVO-compatible message list for a follow-up question.

		:param context: The conversation context created from recognition history.
		:param question: The user's follow-up question.
		:returns: Messages accepted by the VIVO completions API.
		"""
		image = self._getConversationImage(context)
		imageBase64 = self.prepareImageContentFromImage(image).decode("utf-8")
		messages = [
			{
				"role": "user",
				"content": f"data:image/{self.uploadImageFormat};base64,{imageBase64}",
				"contentType": "image",
			},
			{"role": "user", "content": self.prompt, "contentType": "text"},
			{"role": "assistant", "content": context.initialText, "contentType": "text"},
		]
		for turn in context.turns:
			role = "assistant" if getattr(turn, "role", "") == "assistant" else "user"
			text = str(getattr(turn, "text", "")).strip()
			if text:
				messages.append({"role": role, "content": text, "contentType": "text"})
		messages.append({"role": "user", "content": question, "contentType": "text"})
		return messages

	def askQuestionStream(
		self,
		context: Any,
		question: str,
		cancellationChecker: Callable[[], None] | None = None,
	) -> Iterator[QuestionStreamEvent]:
		self._checkQuestionCancelled(cancellationChecker)
		messages = self._buildQuestionMessages(context, question)
		self._checkQuestionCancelled(cancellationChecker)
		request = RecognitionRequest(textResult=True, streamResult=True)
		requestParams = self._buildRequestParamsFromMessages(messages, request)
		self._checkQuestionCancelled(cancellationChecker)
		yield from self._iterQuestionStreamingResponse(requestParams, request, cancellationChecker)

	def processApiResult(self, result: bytes) -> str | bool:
		"""
		Handles error checking from the API response.

		:param result: The raw byte response from the server.
		:returns: An error message string on failure, or False on success.
		:raises ApiError: If the server response is malformed.
		"""
		try:
			responseText = result.decode("utf-8", "ignore")
			responseJson = json.loads(responseText)
		except (json.JSONDecodeError, UnicodeDecodeError) as e:
			log.error("Failed to decode or parse VIVO API response.", exc_info=True)
			# Translators: An error message for an invalid server response.
			raise ApiError(_("Invalid response from server.")) from e

		errorMessage = vivoCompletions.getApiErrorMessage(responseJson)
		if errorMessage:
			log.error(f"VIVO API returned an error: {errorMessage}")
			return errorMessage

		responseData = responseJson.get("data")
		if not isinstance(responseData, dict):
			responseData = {}
		content = responseData.get("content")
		if not content:
			log.warning("VIVO API response successful, but 'data.content' is missing or empty.")
			# Translators: An error message for an empty but successful server response.
			return _("Server returned a successful but empty response.")
		log.debug(
			"VIVO BlueLLM response received: "
			f"model={responseData.get('model')!r}, contentChars={len(str(content))}, "
			f"hasReasoning={bool(responseData.get('reasoningContent'))}",
		)

		return False  # Indicates success

	def extractText(self, apiResult: dict) -> str:
		return vivoCompletions.extractText(apiResult)
