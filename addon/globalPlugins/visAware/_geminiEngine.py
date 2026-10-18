# Copyright (C) 2026 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Shared Gemini API-key and model discovery settings."""

from collections import OrderedDict
from typing import Any

import addonHandler
import config

from . import network
from ._modelSettings import KnownModelsMixin
from .engineSettings import ButtonEngineSetting, ChoiceEngineSetting, TextInputEngineSetting
from .exceptions import ApiError, AuthenticationError
from .geminiModels import getGeminiModelChoices

addonHandler.initTranslation()


MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000"
_DEPRECATED_GEMINI_MODELS = {"gemini-3.1-flash-lite-preview"}
_UNSUPPORTED_GEMINI_MARKERS = (
	"-computer-use",
	"-audio",
	"-image",
	"-live",
	"-robotics",
	"-transcribe",
	"-tts",
	"nano-banana",
	"omni-",
	"customtools",
)


def _modelId(model: dict[str, Any]) -> str:
	name = model.get("name")
	if not isinstance(name, str):
		return ""
	return name.removeprefix("models/").strip()


def _isSelectableModel(model: dict[str, Any], family: str) -> bool:
	modelId = _modelId(model)
	methods = model.get("supportedGenerationMethods")
	if not modelId or not isinstance(methods, list) or "generateContent" not in methods:
		return False
	if not modelId.startswith(f"{family}-"):
		return False
	if family == "gemma":
		# shortcut: model metadata has no modality flag; expand this verified family when a new vision Gemma generation is released.
		return modelId.startswith("gemma-4-") and modelId.endswith("-it")
	if modelId in _DEPRECATED_GEMINI_MODELS:
		return False
	return not any(marker in modelId for marker in _UNSUPPORTED_GEMINI_MARKERS)


def listGeminiModelNames(apiKey: str, family: str = "gemini") -> list[str]:
	"""Returns selectable GenerateContent model IDs for a Gemini API key."""
	if not apiKey.strip():
		raise AuthenticationError(_("API key is missing."))
	response = network.sendRequest(
		"GET",
		MODELS_URL,
		headers={"x-goog-api-key": apiKey.strip()},
		timeout=30,
		retry=True,
	)
	payload = response.json()
	models = payload.get("models") if isinstance(payload, dict) else None
	if not isinstance(models, list):
		raise ApiError(_("Gemini returned an invalid model list."))
	return list(
		dict.fromkeys(
			modelId
			for model in models
			if isinstance(model, dict) and _isSelectableModel(model, family)
			for modelId in [_modelId(model)]
		),
	)


class GeminiEngineMixin(KnownModelsMixin):
	"""Shared model choices and explicit discovery for Gemini API engines."""

	modelFamily = "gemini"
	_apiKey: str = ""

	def loadSettings(self, onlyChanged: bool = False) -> None:
		super().loadSettings(onlyChanged=onlyChanged)
		if not self.configSectionName:
			return
		try:
			configuredModel = config.conf[self.configSectionName][self.name]["model"]
		except (KeyError, AttributeError):
			return
		if configuredModel != self.model:
			self.model = configuredModel

	@property
	def apiKey(self) -> str:
		return self._apiKey

	@apiKey.setter
	def apiKey(self, value: str) -> None:
		value = value.strip()
		if value != self._apiKey:
			self._setKnownModels([])
		self._apiKey = value

	@property
	def availableModels(self) -> dict:
		knownModels = self._getKnownModels()
		if knownModels:
			models = OrderedDict((model, model) for model in knownModels)
		else:
			models = self._getStaticModelChoices()
		if self.model and self.model not in models:
			models[self.model] = self.model
		return self.generateStringSettings(models)

	def _getStaticModelChoices(self) -> OrderedDict[str, str]:
		return getGeminiModelChoices()

	def _fetchModelNames(self, apiKey: str | None = None) -> list[str]:
		return listGeminiModelNames(apiKey if apiKey is not None else self.apiKey, self.modelFamily)

	def fetchModelsChanger(self, evt: Any) -> None:
		import ui
		from .engineSettingsUI import runSettingsAction

		evt.Skip()
		key = self.apiKey
		ui.message(_("Fetching {engine} model list").format(engine=self.description))

		def fetch() -> list[str]:
			return self._fetchModelNames(key)

		def onSuccess(models: list[str]) -> None:
			if not models:
				ui.message(_("No models were found for {engine}.").format(engine=self.description))
				return
			self._setKnownModels(models)
			ui.message(
				_("{engine} models loaded: {count}").format(engine=self.description, count=len(models)),
			)

		runSettingsAction(
			evt.GetEventObject(),
			self,
			fetch,
			onSuccess,
			lambda message: ui.message(
				_("Could not fetch {engine} models: {message}").format(
					engine=self.description,
					message=message,
				),
			),
			lambda: key == self.apiKey,
		)

	def _geminiModelSettings(self) -> list[Any]:
		return [
			TextInputEngineSetting(
				name="apiKey",
				# Translators: The label for the text field to enter a Gemini API key.
				displayNameWithAccelerator=_("API &key"),
				refreshSettingsOnChange=True,
			),
			ButtonEngineSetting(
				name="fetchModels",
				# Translators: The label for a button that fetches Gemini model names.
				displayNameWithAccelerator=_("&Fetch models"),
			),
			ChoiceEngineSetting(
				name="model",
				# Translators: The label for a setting to select a Gemini model.
				displayNameWithAccelerator=_("&Model"),
				optionsPropertyName="availableModels",
			),
		]
