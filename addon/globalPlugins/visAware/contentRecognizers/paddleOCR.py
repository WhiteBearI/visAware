# Copyright (C) 2026 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""An OCR engine that uses PaddleOCR and PaddleOCR-VL services."""

from __future__ import annotations

import addonHandler
from collections import OrderedDict
import config
from contentRecog import LinesWordsResult, RecogImageInfo, SimpleTextResult
import json
from PIL import Image
from typing import Any

from ..engineSettings import BooleanEngineSetting, ChoiceEngineSetting, TextInputEngineSetting
from ..exceptions import ApiError, AuthenticationError
from .. import recogHistory
from ..recogHandler import AUTO_RECOGNITION_REQUEST_TIMEOUT, BaseRecognizer, RecognitionRequest
from ._paddleOCRResults import IMAGE_FIELD_NAMES, PaddleOCRResultParser
from ._paddleOCRClient import (
	DEFAULT_AISTUDIO_ASYNC_URL,
	MODEL_PADDLEOCR_VL,
	MODEL_PADDLEOCR_VL_1_5,
	MODEL_PADDLEOCR_VL_1_6,
	MODEL_PP_OCR_V5,
	MODEL_PP_OCR_V6,
	MODEL_PP_STRUCTURE_V3,
	PaddleOCRClient,
	PaddleOCRClientOptions,
	SERVICE_TYPE_AISTUDIO_ASYNC,
	SERVICE_TYPE_AISTUDIO_SYNC,
	SERVICE_TYPE_SELF_HOSTED,
)

addonHandler.initTranslation()

MAX_HISTORY_STRING_LENGTH = 20000


class CustomContentRecognizer(BaseRecognizer):
	"""Recognizes text and document layout using PaddleOCR services."""

	name = "paddleOCR"
	supportsAutomaticRecognition = True
	# Translators: The description of the PaddleOCR engine.
	description = _("PaddleOCR")

	uploadBase64EncodeImage = True
	uploadImageFormat = "PNG"
	isStreaming = False
	maxSize = 10 * 1024 * 1024

	_serviceType: str = SERVICE_TYPE_AISTUDIO_ASYNC
	_aistudioAsyncApiUrl: str = DEFAULT_AISTUDIO_ASYNC_URL
	_aistudioAsyncToken: str = ""
	_aistudioAsyncModel: str = MODEL_PADDLEOCR_VL_1_5
	_aistudioSyncApiUrl: str = ""
	_aistudioSyncToken: str = ""
	_aistudioSyncModel: str = MODEL_PADDLEOCR_VL_1_5
	_selfHostedApiUrl: str = ""
	_selfHostedToken: str = ""
	_selfHostedModel: str = MODEL_PADDLEOCR_VL_1_6
	_useDocOrientationClassify: bool = False
	_useDocUnwarping: bool = False
	_useTextlineOrientation: bool = False
	_useChartRecognition: bool = False
	_uploadedImageSize: tuple[int, int] | None = None
	_PROFILE_SETTINGS = {
		SERVICE_TYPE_AISTUDIO_ASYNC: {
			"apiUrl": "_aistudioAsyncApiUrl",
			"token": "_aistudioAsyncToken",
			"model": "_aistudioAsyncModel",
		},
		SERVICE_TYPE_AISTUDIO_SYNC: {
			"apiUrl": "_aistudioSyncApiUrl",
			"token": "_aistudioSyncToken",
			"model": "_aistudioSyncModel",
		},
		SERVICE_TYPE_SELF_HOSTED: {
			"apiUrl": "_selfHostedApiUrl",
			"token": "_selfHostedToken",
			"model": "_selfHostedModel",
		},
	}

	@property
	def supportedSettings(self) -> list[Any]:
		"""
		Defines the user-configurable settings for this engine.

		:returns: A list of engine setting objects.
		"""
		settings = [
			ChoiceEngineSetting(
				name="serviceType",
				# Translators: The label for selecting the PaddleOCR service type.
				displayNameWithAccelerator=_("Service &type"),
				optionsPropertyName="availableServiceTypes",
			),
			ChoiceEngineSetting(
				name="model",
				# Translators: The label for selecting a PaddleOCR model.
				displayNameWithAccelerator=_("&Model"),
				optionsPropertyName="availableModels",
			),
			self.autoRecognitionModelSetting(),
			TextInputEngineSetting(
				name="apiUrl",
				# Translators: The label for the PaddleOCR API URL field.
				displayNameWithAccelerator=_("API &URL"),
			),
			TextInputEngineSetting(
				name="token",
				# Translators: The label for the PaddleOCR API token field.
				displayNameWithAccelerator=_("API &token"),
			),
		]
		settings.extend(
			[
				BooleanEngineSetting(
					name="useDocOrientationClassify",
					# Translators: The label for enabling PaddleOCR document orientation detection.
					displayNameWithAccelerator=_("&Detect document orientation"),
				),
				BooleanEngineSetting(
					name="useDocUnwarping",
					# Translators: The label for enabling PaddleOCR document unwarping.
					displayNameWithAccelerator=_("Enable document &unwarping"),
				),
			],
		)
		if self._isCurrentOCRModel():
			settings.append(
				BooleanEngineSetting(
					name="useTextlineOrientation",
					# Translators: The label for enabling PaddleOCR text line orientation detection.
					displayNameWithAccelerator=_("Detect text line &orientation"),
				),
			)
		else:
			settings.append(
				BooleanEngineSetting(
					name="useChartRecognition",
					# Translators: The label for enabling PaddleOCR chart recognition.
					displayNameWithAccelerator=_("Enable &chart recognition"),
				),
			)
		return settings

	def getConfigSpec(self) -> dict:
		"""
		Returns the full configuration spec, including inactive service profiles.

		Inactive profile fields are not shown in the UI, but still need a config
		spec so official and self-hosted credentials can be saved side by side.
		"""
		spec = super().getConfigSpec()
		for name, _attr in self._iterProfileConfigItems():
			spec[name] = "string(default=None)"
		return spec

	def saveSettings(self) -> None:
		"""Saves all PaddleOCR service profiles without letting them overwrite each other."""
		if not self.configSectionName:
			return
		conf = config.conf[self.configSectionName][self.name]
		conf["serviceType"] = self.serviceType
		conf["autoRecognitionModel"] = self.autoRecognitionModel
		for name, attr in self._iterProfileConfigItems():
			conf[name] = getattr(self, attr)
		for name in (
			"useDocOrientationClassify",
			"useDocUnwarping",
			"useTextlineOrientation",
			"useChartRecognition",
		):
			conf[name] = getattr(self, name)

	def loadSettings(self, onlyChanged: bool = False) -> None:
		"""Loads all PaddleOCR service profiles, then selects the active one."""
		if not self.configSectionName:
			return
		conf = config.conf[self.configSectionName][self.name]
		serviceType = conf.get("serviceType")
		if serviceType in self.availableServiceTypes:
			self._setLoadedSetting("_serviceType", serviceType, onlyChanged)
		if "autoRecognitionModel" in conf:
			value = str(conf["autoRecognitionModel"] or "").strip()
			self._setLoadedSetting("_autoRecognitionModel", value, onlyChanged)
		for name, attr in self._iterProfileConfigItems():
			if name in conf:
				value = str(conf[name] or "")
				if attr.endswith("ApiUrl"):
					value = value.strip()
				self._setLoadedSetting(attr, value, onlyChanged)
		for name in (
			"useDocOrientationClassify",
			"useDocUnwarping",
			"useTextlineOrientation",
			"useChartRecognition",
		):
			if name not in conf:
				continue
			value = self._coerceBoolean(conf[name])
			if value is not None:
				self._setLoadedSetting(f"_{name}", value, onlyChanged)
		self._ensureValidModel()

	@classmethod
	def _iterProfileConfigItems(cls):
		for profile in cls._PROFILE_SETTINGS.values():
			for attr in profile.values():
				yield attr[1:], attr

	def _setLoadedSetting(self, attr: str, value: Any, onlyChanged: bool) -> None:
		if onlyChanged and getattr(self, attr, None) == value:
			return
		setattr(self, attr, value)

	@staticmethod
	def _coerceBoolean(value: Any) -> bool | None:
		if isinstance(value, bool):
			return value
		normalized = str(value).strip().lower()
		if normalized in ("1", "true", "yes", "on"):
			return True
		if normalized in ("0", "false", "no", "off"):
			return False
		return None

	def _currentProfileAttr(self, name: str) -> str:
		return self._PROFILE_SETTINGS[self.serviceType][name]

	@property
	def apiUrl(self) -> str:
		return getattr(self, self._currentProfileAttr("apiUrl"))

	@apiUrl.setter
	def apiUrl(self, value: str) -> None:
		setattr(self, self._currentProfileAttr("apiUrl"), value.strip())

	@property
	def token(self) -> str:
		return getattr(self, self._currentProfileAttr("token"))

	@token.setter
	def token(self, value: str) -> None:
		setattr(self, self._currentProfileAttr("token"), value)

	@property
	def model(self) -> str:
		return getattr(self, self._currentProfileAttr("model"))

	@model.setter
	def model(self, value: str) -> None:
		if value in self.availableModels:
			setattr(self, self._currentProfileAttr("model"), value)

	@property
	def serviceType(self) -> str:
		return self._serviceType

	@serviceType.setter
	def serviceType(self, value: str) -> None:
		if value in self.availableServiceTypes:
			self._serviceType = value
			self._ensureValidModel()

	def _ensureValidModel(self) -> None:
		if self.model in self.availableModels:
			return
		self.model = next(iter(self.availableModels))

	def _isCurrentOCRModel(self) -> bool:
		return self.model in (MODEL_PP_OCR_V5, MODEL_PP_OCR_V6)

	@property
	def useDocOrientationClassify(self) -> bool:
		return self._useDocOrientationClassify

	@useDocOrientationClassify.setter
	def useDocOrientationClassify(self, value: bool) -> None:
		self._useDocOrientationClassify = value

	@property
	def useDocUnwarping(self) -> bool:
		return self._useDocUnwarping

	@useDocUnwarping.setter
	def useDocUnwarping(self, value: bool) -> None:
		self._useDocUnwarping = value

	@property
	def useTextlineOrientation(self) -> bool:
		return self._useTextlineOrientation

	@useTextlineOrientation.setter
	def useTextlineOrientation(self, value: bool) -> None:
		self._useTextlineOrientation = value

	@property
	def useChartRecognition(self) -> bool:
		return self._useChartRecognition

	@useChartRecognition.setter
	def useChartRecognition(self, value: bool) -> None:
		self._useChartRecognition = value

	@property
	def availableServiceTypes(self) -> dict:
		"""
		Provides the supported PaddleOCR service types for the settings UI.

		:returns: A dictionary of service type IDs to display names.
		"""
		serviceTypes = OrderedDict(
			{
				SERVICE_TYPE_AISTUDIO_ASYNC: _("AI Studio hosted task API (recommended)"),
				SERVICE_TYPE_AISTUDIO_SYNC: _("AI Studio deployed service"),
				SERVICE_TYPE_SELF_HOSTED: _("Self-hosted PaddleOCR service"),
			},
		)
		return self.generateStringSettings(serviceTypes)

	@property
	def availableModels(self) -> dict:
		"""
		Provides the supported PaddleOCR model names for the settings UI.

		:returns: A dictionary of model IDs to display names.
		"""
		models = OrderedDict(
			(
				(MODEL_PADDLEOCR_VL_1_6, _("PaddleOCR-VL 1.6 (document understanding)")),
				(MODEL_PP_OCR_V6, _("PP-OCRv6 (fast text OCR)")),
			)
			if self.serviceType == SERVICE_TYPE_SELF_HOSTED
			else (
				(MODEL_PADDLEOCR_VL_1_5, _("PaddleOCR-VL 1.5 (recommended)")),
				(MODEL_PADDLEOCR_VL, _("PaddleOCR-VL")),
				(MODEL_PP_OCR_V5, _("PP-OCRv5")),
				(MODEL_PP_STRUCTURE_V3, _("PP-StructureV3")),
			),
		)
		return self.generateStringSettings(models)

	@classmethod
	def check(cls) -> bool:
		"""
		Checks if the engine is available.

		:returns: Always True because this is a configurable HTTP engine.
		"""
		return True

	def _serializeImage(self, pilImage: Image.Image) -> bytes:
		"""
		Serializes the upload image and records its dimensions for diagnostics.

		:param pilImage: The image being serialized.
		:returns: The serialized image bytes.
		"""
		self._uploadedImageSize = pilImage.size
		return super()._serializeImage(pilImage)

	def _buildRequestParams(self, imageContent: bytes, request: RecognitionRequest) -> dict:
		"""
		Builds request state consumed by the PaddleOCR client.

		:param imageContent: The base64 encoded serialized image bytes.
		:param request: The request-local recognition options.
		:returns: A small request dictionary for `_handleStandardResponse`.
		"""
		self._validateSettings()
		return {"imageContent": imageContent}

	def _handleStandardResponse(
		self,
		requestParams: dict,
		imageInfo: RecogImageInfo,
		cancellationEvent: Any,
		request: RecognitionRequest,
	) -> Any:
		"""Handles PaddleOCR's sync and async service variants through the private client."""
		client = PaddleOCRClient(
			self._buildClientOptions(),
			cancellationChecker=lambda: self._checkCancelled(cancellationEvent),
			requestTimeout=(AUTO_RECOGNITION_REQUEST_TIMEOUT if request.isAutomaticRecognition else None),
		)
		apiResult = client.recognizeImage(requestParams["imageContent"])
		self._checkCancelled(cancellationEvent)
		historyEntry = (
			recogHistory.createEntry(self, self.originalImage, self._pruneResultForHistory(apiResult))
			if self.originalImage
			else None
		)
		if request.textResult:
			ocrResult = self.extractText(apiResult)
			if not ocrResult or ocrResult.isspace():
				# Translators: An error message for a blank recognition result.
				raise ApiError(_("Recognition result is blank."))
			return recogHistory.attachEntry(SimpleTextResult(ocrResult), historyEntry)
		lineResult = self._convertToLineResultFormat(apiResult)
		if lineResult:
			return recogHistory.attachEntry(LinesWordsResult(lineResult, imageInfo), historyEntry)
		ocrResult = self.extractText(apiResult)
		if not ocrResult or ocrResult.isspace():
			# Translators: An error message for a blank recognition result.
			raise ApiError(_("Recognition result is blank."))
		textOnlyResult = SimpleTextResult(ocrResult)
		textOnlyResult.forceVirtualDocument = True
		return recogHistory.attachEntry(textOnlyResult, historyEntry)

	def _validateSettings(self) -> None:
		if self.serviceType in (SERVICE_TYPE_AISTUDIO_ASYNC, SERVICE_TYPE_AISTUDIO_SYNC) and not self.token:
			# Translators: An error message if the PaddleOCR token is missing.
			raise AuthenticationError(_("API token is missing. Please configure it in PaddleOCR settings."))
		if self.serviceType != SERVICE_TYPE_AISTUDIO_ASYNC and (
			not self.apiUrl or self.apiUrl == DEFAULT_AISTUDIO_ASYNC_URL
		):
			# Translators: An error message if the PaddleOCR API URL is missing.
			raise AuthenticationError(_("API URL is missing. Please configure it in PaddleOCR settings."))

	def _buildClientOptions(self) -> PaddleOCRClientOptions:
		return PaddleOCRClientOptions(
			serviceType=self.serviceType,
			apiUrl=self.apiUrl,
			token=self.token,
			model=self.model,
			useDocOrientationClassify=self.useDocOrientationClassify,
			useDocUnwarping=self.useDocUnwarping,
			useTextlineOrientation=self.useTextlineOrientation,
			useChartRecognition=self.useChartRecognition,
		)

	def processApiResult(self, result: bytes) -> str | bool:
		"""
		Handles error checking for direct PaddleOCR responses.

		:param result: The raw byte response from the server.
		:returns: False on success, otherwise a user-visible error message.
		"""
		try:
			responseJson = self._convertToJson(result)
			error = responseJson.get("error")
			if error:
				return str(error)
			errorCode = responseJson.get("errorCode") or responseJson.get("error_code")
			if errorCode not in (None, 0, "0"):
				return str(responseJson.get("errorMsg") or responseJson.get("error_msg") or errorCode)
		except (json.JSONDecodeError, AttributeError) as e:
			return str(e)
		return False

	def extractText(self, apiResult: dict) -> str:
		"""
		Extracts the plain text result from PaddleOCR variants.

		:param apiResult: The parsed PaddleOCR response.
		:returns: Recognized text as a single string.
		"""
		return PaddleOCRResultParser.extractText(apiResult)

	def _convertToLineResultFormat(self, apiResult: dict) -> list:
		"""
		Converts PaddleOCR response data into NVDA's rich OCR result format.

		:param apiResult: The parsed PaddleOCR response.
		:returns: A list of lines where each line contains word dictionaries.
		"""
		return PaddleOCRResultParser.toLineResult(apiResult)

	def _cleanup(self) -> None:
		"""Cleans up engine state after recognition."""
		self._uploadedImageSize = None
		super()._cleanup()

	@classmethod
	def _pruneResultForHistory(cls, value: Any) -> Any:
		if isinstance(value, dict):
			pruned: dict[str, Any] = {}
			for key, child in value.items():
				if key in IMAGE_FIELD_NAMES:
					continue
				pruned[key] = cls._pruneResultForHistory(child)
			return pruned
		if isinstance(value, list):
			return [cls._pruneResultForHistory(child) for child in value]
		if isinstance(value, str) and len(value) > MAX_HISTORY_STRING_LENGTH:
			return value[:MAX_HISTORY_STRING_LENGTH] + "...[omitted]"
		return value
