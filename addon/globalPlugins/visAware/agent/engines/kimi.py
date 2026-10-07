# Copyright (C) 2026 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Kimi (Moonshot AI) computer-use agent engine."""

from __future__ import annotations

import config
from typing import Any

import addonHandler

from ...engineGUIHelper import (
	ButtonEngineSetting,
	ChoiceEngineSetting,
	EditableChoiceEngineSetting,
	TextInputEngineSetting,
)
from ...kimiModels import (
	DEFAULT_KIMI_BASE_URL,
	KIMI_AUTO_MODEL,
	KIMI_AUTO_SERVICE,
	KIMI_CUSTOM_SERVICE,
	KIMI_REASONING_EFFORTS,
	getPreferredKimiModel,
	getKimiModelChoices,
	getKimiReasoningEffortChoices,
	getKimiServiceBaseUrl,
	getKimiServiceChoices,
	ensureKimiConnection,
	invalidateKimiConnection,
	isKimiK26Model,
	isKimiK3Model,
	normalizeKimiReasoningEffort,
	normalizeKimiReasoningEffortForModel,
	startKimiModelDetection,
)
from ..kimi import KimiAgentClient, KimiAgentSettings
from ..settings import BaseAgentEngine

addonHandler.initTranslation()

_KNOWN_MODELS_CONFIG_KEY = "knownModels"


class AgentEngine(BaseAgentEngine):
	"""Kimi-backed computer-use agent engine."""

	name = "kimi"
	# Translators: The description of the Kimi Agent engine. "Moonshot AI" is a proper noun
	# and should not be translated.
	description = _("Kimi (Moonshot AI)")
	engineConfigSpec = {
		_KNOWN_MODELS_CONFIG_KEY: "list(default=list())",
	}

	_apiKey: str = ""
	_baseUrl: str = DEFAULT_KIMI_BASE_URL
	_service: str = KIMI_AUTO_SERVICE
	_resolvedBaseUrl: str | None = None
	_resolvedModels: list[str] = []
	_connectionKey: tuple[str, str, str] | None = None
	_model: str = KIMI_AUTO_MODEL
	_autoModelSelected: bool = True
	_reasoningEffort: str = "high"
	_knownModels: list[str]

	@property
	def supportedSettings(self) -> list[Any]:
		return [
			TextInputEngineSetting(
				name="apiKey",
				# Translators: The label for the Kimi API key used by the Agent.
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
				# Translators: The label for the Kimi-compatible base URL used by the Agent.
				displayNameWithAccelerator=_("Base &URL"),
				refreshSettingsOnChange=True,
			),
			ButtonEngineSetting(
				name="fetchModels",
				# Translators: The label for the button that detects Kimi services and models.
				displayNameWithAccelerator=_("Detect service and &models"),
			),
			EditableChoiceEngineSetting(
				name="model",
				# Translators: The label for the Kimi model used by the Agent.
				displayNameWithAccelerator=_("&Model"),
				optionsPropertyName="availableModels",
			),
			ChoiceEngineSetting(
				name="reasoningEffort",
				# Translators: The label for Kimi K3 reasoning effort.
				displayNameWithAccelerator=_("&Thinking effort"),
				optionsPropertyName="reasoningEffortChoices",
			),
			self._imageQualitySetting(),
		]

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
		baseUrl = value.strip().rstrip("/") or DEFAULT_KIMI_BASE_URL
		if baseUrl != self._baseUrl:
			self._knownModels = []
		self._baseUrl = baseUrl
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

	@property
	def reasoningEffort(self) -> str:
		return self._reasoningEffort

	@reasoningEffort.setter
	def reasoningEffort(self, value: str) -> None:
		self._reasoningEffort = normalizeKimiReasoningEffort(value)
		self._normalizeReasoningEffort()

	@property
	def availableModels(self) -> dict:
		models = {KIMI_AUTO_MODEL: _("Auto-select (on first use)")}
		models.update({modelName: modelName for modelName in self._getKnownModels()})
		if self.model and self.model != KIMI_AUTO_MODEL and self.model not in models:
			models[self.model] = self.model
		return self.generateStringSettings(models)

	@property
	def reasoningEffortChoices(self) -> dict:
		choices = getKimiReasoningEffortChoices(self.model)
		return self.generateStringSettings(choices or KIMI_REASONING_EFFORTS)

	def _normalizeReasoningEffort(self) -> None:
		self._reasoningEffort = normalizeKimiReasoningEffortForModel(self.model, self._reasoningEffort)

	def loadSettings(self, onlyChanged: bool = False) -> None:
		super().loadSettings(onlyChanged=onlyChanged)
		self._loadKnownModelsFromConfig()

	def saveSettings(self) -> None:
		runtimeModel = self._model
		if self._autoModelSelected:
			self._model = KIMI_AUTO_MODEL
		try:
			super().saveSettings()
		finally:
			self._model = runtimeModel
		if not self.configSectionName:
			return
		config.conf[self.configSectionName][self.name][_KNOWN_MODELS_CONFIG_KEY] = self._getKnownModels()

	@classmethod
	def check(cls) -> bool:
		return True

	def isSupported(self, settingName: str) -> bool:
		if settingName == "baseUrl":
			return self.service == KIMI_CUSTOM_SERVICE
		if settingName == "reasoningEffort":
			return isKimiK3Model(self.model) or isKimiK26Model(self.model)
		return super().isSupported(settingName)

	def createClient(self) -> KimiAgentClient:
		connectionIsCurrent = self._connectionKey == (self.apiKey, self.service, self._baseUrl) and bool(
			self._resolvedBaseUrl
		)
		return KimiAgentClient(
			KimiAgentSettings(
				apiKey=self.apiKey,
				baseUrl=self._resolvedBaseUrl if connectionIsCurrent else self.baseUrl,
				model=self.model,
				imageQuality=self.imageQuality,
				reasoningEffort=self.reasoningEffort,
				source="Agent Kimi engine settings",
				service=self.service,
				resolveConnection=not connectionIsCurrent,
				onConnectionResolved=self._onConnectionResolved,
			),
		)

	def _onConnectionResolved(self, baseUrl: str, models: list[str]) -> None:
		self._resolvedBaseUrl = baseUrl
		self._resolvedModels = list(models)
		self._connectionKey = (self.apiKey, self.service, self._baseUrl)
		self._setKnownModels(models)
		if self._autoModelSelected:
			self._model = getPreferredKimiModel(baseUrl, models)
			self._normalizeReasoningEffort()

	def fetchModelsChanger(self, evt: Any) -> None:
		startKimiModelDetection(self, evt)

	def _loadKnownModelsFromConfig(self) -> None:
		if not self.configSectionName:
			return
		try:
			knownModels = config.conf[self.configSectionName][self.name][_KNOWN_MODELS_CONFIG_KEY]
		except (KeyError, AttributeError):
			return
		if isinstance(knownModels, str):
			knownModels = [knownModels]
		self._setKnownModels([str(modelName) for modelName in knownModels])

	def _setKnownModels(self, modelNames: list[str]) -> None:
		knownModels: list[str] = []
		seenModelNames: set[str] = set()
		for modelName in modelNames:
			modelName = modelName.strip()
			if not modelName or modelName in seenModelNames:
				continue
			seenModelNames.add(modelName)
			knownModels.append(modelName)
		self._knownModels = knownModels

	def _getKnownModels(self) -> list[str]:
		return list(getattr(self, "_knownModels", []))

	def _invalidateConnection(self) -> None:
		invalidateKimiConnection(self)

	def _ensureConnection(self, forceDiscovery: bool = False) -> None:
		ensureKimiConnection(
			self,
			forceDiscovery=forceDiscovery,
			missingKeyMessage=_("API key is missing. Please configure it in the AI Agent settings."),
		)
