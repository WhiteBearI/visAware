# Copyright (C) 2026 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

from typing import Any

import addonHandler
from .exceptions import AuthenticationError
from .openaiCompatible import OpenAICompatibleClient, OpenAICompatibleSettings

addonHandler.initTranslation()


class OpenAICompatibleEngineMixin:
	defaultBaseUrl: str
	defaultModel: str
	_apiKey: str = ""

	@property
	def apiKey(self) -> str:
		return self._apiKey

	@apiKey.setter
	def apiKey(self, value: str) -> None:
		value = value.strip()
		if value != self._apiKey:
			self._clearKnownModels()
		self._apiKey = value

	@property
	def baseUrl(self) -> str:
		return getattr(self, "_baseUrl", self.defaultBaseUrl)

	@baseUrl.setter
	def baseUrl(self, value: str) -> None:
		value = value.strip().rstrip("/") or self.defaultBaseUrl
		if value != self.baseUrl:
			self._clearKnownModels()
		self._baseUrl = value

	@property
	def model(self) -> str:
		return getattr(self, "_model", self.defaultModel)

	@model.setter
	def model(self, value: str) -> None:
		self._model = value.strip() or self.defaultModel

	@property
	def availableModels(self) -> dict:
		models = {name: name for name in self._getKnownModels()}
		if self.model:
			models.setdefault(self.model, self.model)
		return self.generateStringSettings(models)

	def _getKnownModels(self) -> list[str]:
		return list(getattr(self, "_knownModels", []))

	def _clearKnownModels(self) -> None:
		setKnownModels = getattr(self, "_setKnownModels", None)
		if callable(setKnownModels):
			setKnownModels([])

	def _openAICompatibleSettings(
		self,
		*,
		includeFetch: bool = True,
		editableModel: bool = True,
	) -> list[Any]:
		from .engineSettings import (
			ButtonEngineSetting,
			ChoiceEngineSetting,
			EditableChoiceEngineSetting,
			TextInputEngineSetting,
		)

		settings = [
			TextInputEngineSetting(
				name="apiKey",
				# Translators: The label for an OpenAI-compatible API key.
				displayNameWithAccelerator=_("API &key"),
			),
			TextInputEngineSetting(
				name="baseUrl",
				# Translators: The label for an OpenAI-compatible base URL.
				displayNameWithAccelerator=_("Base &URL"),
				refreshSettingsOnChange=True,
			),
		]
		if includeFetch:
			settings.append(
				ButtonEngineSetting(
					name="fetchModels",
					# Translators: The label for an OpenAI-compatible model discovery button.
					displayNameWithAccelerator=_("&Fetch models"),
				),
			)
		modelSetting = EditableChoiceEngineSetting if editableModel else ChoiceEngineSetting
		settings.append(
			modelSetting(
				name="model",
				# Translators: The label for an OpenAI-compatible model.
				displayNameWithAccelerator=_("&Model"),
				optionsPropertyName="availableModels",
			),
		)
		return settings

	def _fetchModelNames(self, apiKey: str | None = None, baseUrl: str | None = None) -> list[str]:
		apiKey = self.apiKey if apiKey is None else apiKey
		baseUrl = self.baseUrl if baseUrl is None else baseUrl
		if not apiKey:
			raise AuthenticationError(_("API key is missing."))
		return OpenAICompatibleClient(
			OpenAICompatibleSettings(apiKey=apiKey, baseUrl=baseUrl),
		).listModels()

	def fetchModelsChanger(self, evt: Any) -> None:
		from .engineSettingsUI import runSettingsAction
		import ui

		evt.Skip()
		key = (self.apiKey, self.baseUrl)
		ui.message(_("Fetching {engine} model list").format(engine=self.description))

		def fetch() -> list[str]:
			if not key[0]:
				raise AuthenticationError(_("API key is missing."))
			return self._fetchModelNames(*key)

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
			lambda: key == (self.apiKey, self.baseUrl),
		)
