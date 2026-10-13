# Copyright (C) 2026 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""OpenAI Responses API computer-use agent engine."""

from __future__ import annotations

from typing import Any

import addonHandler
import ui

from ..._modelSettings import KnownModelsMixin
from ...engineSettings import (
	ButtonEngineSetting,
	EditableChoiceEngineSetting,
	TextInputEngineSetting,
)
from ..openai import (
	DEFAULT_OPENAI_BASE_URL,
	DEFAULT_OPENAI_MODEL,
	OpenAIAgentClient,
	OpenAIAgentSettings,
)
from ..engine import BaseAgentEngine

addonHandler.initTranslation()


class AgentEngine(KnownModelsMixin, BaseAgentEngine):
	"""OpenAI-backed computer-use agent engine."""

	name = "openai"
	# Translators: The description of the OpenAI Agent engine.
	description = _("OpenAI")

	_apiKey: str = ""
	_baseUrl: str = DEFAULT_OPENAI_BASE_URL
	_model: str = DEFAULT_OPENAI_MODEL

	@property
	def supportedSettings(self) -> list[Any]:
		return [
			TextInputEngineSetting(
				name="apiKey",
				# Translators: The label for the OpenAI API key used by the Agent.
				displayNameWithAccelerator=_("API &key"),
			),
			TextInputEngineSetting(
				name="baseUrl",
				# Translators: The label for the OpenAI-compatible base URL used by the Agent.
				displayNameWithAccelerator=_("Base &URL"),
			),
			ButtonEngineSetting(
				name="fetchModels",
				# Translators: The label for the button that fetches OpenAI model names.
				displayNameWithAccelerator=_("&Fetch models"),
			),
			EditableChoiceEngineSetting(
				name="model",
				# Translators: The label for the OpenAI model used by the Agent.
				displayNameWithAccelerator=_("&Model"),
				optionsPropertyName="availableModels",
			),
			self._imageQualitySetting(),
		]

	@property
	def apiKey(self) -> str:
		return self._apiKey

	@apiKey.setter
	def apiKey(self, value: str) -> None:
		self._apiKey = value.strip()

	@property
	def baseUrl(self) -> str:
		return self._baseUrl

	@baseUrl.setter
	def baseUrl(self, value: str) -> None:
		self._baseUrl = value.strip() or DEFAULT_OPENAI_BASE_URL

	@property
	def model(self) -> str:
		return self._model

	@model.setter
	def model(self, value: str) -> None:
		self._model = value.strip() or DEFAULT_OPENAI_MODEL

	@property
	def availableModels(self) -> dict:
		models = {modelName: modelName for modelName in self._getKnownModels()}
		if self.model and self.model not in models:
			models[self.model] = self.model
		return self.generateStringSettings(models)

	@classmethod
	def check(cls) -> bool:
		return True

	def createClient(self) -> OpenAIAgentClient:
		return OpenAIAgentClient(self._clientSettings())

	def _clientSettings(self) -> OpenAIAgentSettings:
		return OpenAIAgentSettings(
			apiKey=self.apiKey,
			baseUrl=self.baseUrl,
			model=self.model,
			imageQuality=self.imageQuality,
			source="Agent OpenAI engine settings",
		)

	def fetchModelsChanger(self, evt: Any) -> None:
		from ...engineSettingsUI import runSettingsAction

		evt.Skip()
		settings = self._clientSettings()
		key = (self.apiKey, self.baseUrl)
		# Translators: Reported while fetching the OpenAI model list.
		ui.message(_("Fetching OpenAI model list"))

		def onSuccess(modelNames: list[str]) -> None:
			if not modelNames:
				# Translators: Reported when OpenAI returns no models.
				ui.message(_("No OpenAI models were found."))
				return
			self._setKnownModels(modelNames)
			if not self.model:
				self.model = modelNames[0]
			# Translators: Reported after fetching OpenAI models. {count} is the number of models.
			ui.message(_("OpenAI models loaded: {count}").format(count=len(modelNames)))

		runSettingsAction(
			evt.GetEventObject(),
			self,
			lambda: OpenAIAgentClient(settings).listModels(),
			onSuccess,
			# Translators: Reported when fetching OpenAI models fails. {message} is the error message.
			lambda message: ui.message(_("Could not fetch OpenAI models: {message}").format(message=message)),
			lambda: key == (self.apiKey, self.baseUrl),
		)
