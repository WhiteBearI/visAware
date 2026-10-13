# Copyright (C) 2026 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Shared Kimi engine settings and model discovery UI."""

from typing import Any

import addonHandler

from .engineSettings import (
	ButtonEngineSetting,
	ChoiceEngineSetting,
	EditableChoiceEngineSetting,
	TextInputEngineSetting,
)
from .kimiConnection import applyKimiConnection, invalidateKimiConnection, resolveKimiConnection
from .kimiProvider import (
	DEFAULT_KIMI_BASE_URL,
	KIMI_AUTO_MODEL,
	KIMI_AUTO_SERVICE,
	KIMI_CUSTOM_SERVICE,
	KIMI_REASONING_EFFORTS,
	getKimiModelChoices,
	getKimiReasoningEffortChoices,
	getKimiServiceBaseUrl,
	getKimiServiceChoices,
	isKimiK26Model,
	isKimiK3Model,
	normalizeKimiReasoningEffort,
	normalizeKimiReasoningEffortForModel,
)

addonHandler.initTranslation()


class KimiEngineMixin:
	_apiKey: str = ""
	_baseUrl: str = DEFAULT_KIMI_BASE_URL
	_service: str = KIMI_AUTO_SERVICE
	_resolvedBaseUrl: str | None = None
	_resolvedModels: list[str] = []
	_connectionKey: tuple[str, str, str] | None = None
	_model: str = KIMI_AUTO_MODEL
	_autoModelSelected: bool = True
	_reasoningEffort: str = "high"

	@property
	def apiKey(self) -> str:
		return self._apiKey

	@apiKey.setter
	def apiKey(self, value: str) -> None:
		self._apiKey = value.strip()
		self._invalidateConnection()

	@property
	def service(self) -> str:
		return self._service

	@service.setter
	def service(self, value: str) -> None:
		previousService = self._service
		self._service = value if value in getKimiServiceChoices() else KIMI_AUTO_SERVICE
		if self._service not in {KIMI_AUTO_SERVICE, KIMI_CUSTOM_SERVICE}:
			self._baseUrl = getKimiServiceBaseUrl(self._service)
			if self.model not in getKimiModelChoices(self._baseUrl):
				self._autoModelSelected = True
				self._model = KIMI_AUTO_MODEL
		elif self._service == KIMI_AUTO_SERVICE and previousService == KIMI_CUSTOM_SERVICE:
			self._baseUrl = DEFAULT_KIMI_BASE_URL
			self._autoModelSelected = True
			self._model = KIMI_AUTO_MODEL
		self._invalidateConnection()

	@property
	def serviceChoices(self) -> dict:
		return self.generateStringSettings(getKimiServiceChoices())

	@property
	def baseUrl(self) -> str:
		return self._baseUrl

	@baseUrl.setter
	def baseUrl(self, value: str) -> None:
		self._baseUrl = value.strip().rstrip("/") or DEFAULT_KIMI_BASE_URL
		self._invalidateConnection()

	@property
	def model(self) -> str:
		return self._model

	@model.setter
	def model(self, value: str) -> None:
		value = value.strip()
		if value == _("Auto-select (on first use)"):
			value = KIMI_AUTO_MODEL
		self._autoModelSelected = not value or value == KIMI_AUTO_MODEL
		self._model = value or KIMI_AUTO_MODEL
		if self._autoModelSelected:
			self._resolvedBaseUrl = None
			self._resolvedModels = []
			self._connectionKey = None
		self._normalizeReasoningEffort()

	def saveSettings(self) -> None:
		runtimeModel = self._model
		if self._autoModelSelected:
			self._model = KIMI_AUTO_MODEL
		try:
			super().saveSettings()
		finally:
			self._model = runtimeModel

	@property
	def reasoningEffort(self) -> str:
		return self._reasoningEffort

	@reasoningEffort.setter
	def reasoningEffort(self, value: str) -> None:
		self._reasoningEffort = normalizeKimiReasoningEffort(value)
		self._normalizeReasoningEffort()

	@property
	def availableModels(self) -> dict:
		models = list(self._resolvedModels) or [KIMI_AUTO_MODEL]
		if self.model and self.model != KIMI_AUTO_MODEL and self.model not in models:
			models.append(self.model)
		return self.generateStringSettings(
			{
				KIMI_AUTO_MODEL: _("Auto-select (on first use)"),
				**{model: model for model in models if model != KIMI_AUTO_MODEL},
			}
		)

	@property
	def reasoningEffortChoices(self) -> dict:
		choices = getKimiReasoningEffortChoices(self.model)
		return self.generateStringSettings(choices or KIMI_REASONING_EFFORTS)

	def isSupported(self, settingName: str) -> bool:
		if settingName == "baseUrl":
			return self.service == KIMI_CUSTOM_SERVICE
		if settingName == "reasoningEffort":
			return isKimiK3Model(self.model) or isKimiK26Model(self.model)
		return super().isSupported(settingName)

	def _normalizeReasoningEffort(self) -> None:
		self._reasoningEffort = normalizeKimiReasoningEffortForModel(self.model, self._reasoningEffort)

	def _invalidateConnection(self) -> None:
		invalidateKimiConnection(self)

	def detectModelsChanger(self, evt: Any) -> None:
		import ui
		from .engineSettingsUI import runSettingsAction

		evt.Skip()
		key = (self.apiKey, self.service, self._baseUrl)
		model = self.model
		ui.message(_("Detecting Kimi service and models"))

		def onSuccess(connection: tuple[str, list[str]]) -> None:
			baseUrl, models = connection
			applyKimiConnection(self, key, baseUrl, models)
			ui.message(_("Kimi models loaded: {count}").format(count=len(models)))

		runSettingsAction(
			evt.GetEventObject(),
			self,
			lambda: resolveKimiConnection(*key, model, forceDiscovery=True),
			onSuccess,
			lambda message: ui.message(_("Could not detect Kimi service: {message}").format(message=message)),
			lambda: key == (self.apiKey, self.service, self._baseUrl),
		)

	fetchModelsChanger = detectModelsChanger

	def _kimiConnectionSettings(self) -> list[Any]:
		return [
			TextInputEngineSetting(
				name="apiKey",
				# Translators: The label for the text field to enter the Kimi API key.
				displayNameWithAccelerator=_("API &key"),
			),
			ChoiceEngineSetting(
				name="service",
				# Translators: The label for the Kimi service endpoint.
				displayNameWithAccelerator=_("Kimi &service"),
				optionsPropertyName="serviceChoices",
			),
			TextInputEngineSetting(
				name="baseUrl",
				# Translators: The label for the text field to enter the Kimi API base URL.
				displayNameWithAccelerator=_("Base &URL"),
				refreshSettingsOnChange=True,
			),
			ButtonEngineSetting(
				name="detectModels",
				# Translators: The label for the button that detects Kimi services and models.
				displayNameWithAccelerator=_("Detect service and &models"),
			),
			EditableChoiceEngineSetting(
				name="model",
				# Translators: The label for a setting to select the Kimi model.
				displayNameWithAccelerator=_("&Model"),
				optionsPropertyName="availableModels",
			),
		]

	def _kimiReasoningSetting(self) -> ChoiceEngineSetting:
		return ChoiceEngineSetting(
			name="reasoningEffort",
			# Translators: The label for Kimi thinking effort.
			displayNameWithAccelerator=_("&Thinking effort"),
			optionsPropertyName="reasoningEffortChoices",
		)
