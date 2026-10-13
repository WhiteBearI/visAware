# Copyright (C) 2026 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Shared settings behavior for Ollama engines."""

from __future__ import annotations

import addonHandler
from typing import Any
import ui

from ._modelSettings import KnownModelsMixin
from ._ollamaClient import DEFAULT_OLLAMA_API_BASE_URL, OllamaClient
from .exceptions import ApiError

addonHandler.initTranslation()


class OllamaEngineMixin(KnownModelsMixin):
	"""Common configurable settings and model discovery for Ollama engines."""

	_apiBaseUrl: str = DEFAULT_OLLAMA_API_BASE_URL
	_apiKey: str = ""
	_model: str = ""
	_useStreaming: bool = False

	@property
	def apiBaseUrl(self) -> str:
		return self._apiBaseUrl

	@apiBaseUrl.setter
	def apiBaseUrl(self, value: str) -> None:
		self._apiBaseUrl = value.strip() if value else DEFAULT_OLLAMA_API_BASE_URL

	@property
	def apiKey(self) -> str:
		return self._apiKey

	@apiKey.setter
	def apiKey(self, value: str) -> None:
		self._apiKey = value.strip() if value else ""

	@property
	def model(self) -> str:
		return self._model

	@model.setter
	def model(self, value: str) -> None:
		self._model = value.strip() if value else ""

	@property
	def useStreaming(self) -> bool:
		return self._useStreaming

	@useStreaming.setter
	def useStreaming(self, value: bool) -> None:
		self._useStreaming = bool(value)
		self.isStreaming = bool(value)

	@property
	def availableModels(self) -> dict:
		"""
		Provides available Ollama models for the settings UI.

		:returns: A dictionary of model IDs to display names.
		"""
		models = {modelName: modelName for modelName in self._getKnownModels()}
		if self.model and self.model not in models:
			models[self.model] = self.model
		if not models:
			# Translators: A placeholder shown in the Ollama model list before models are fetched.
			models[""] = _("Fetch model list first")
		return self.generateStringSettings(models)

	def fetchModelsChanger(self, evt: Any) -> None:
		from .engineSettingsUI import runSettingsAction

		evt.Skip()
		key = (self.apiBaseUrl, self.apiKey)
		model = self.model
		# Translators: Reported while fetching the Ollama model list.
		ui.message(_("Fetching Ollama model list"))

		def onSuccess(modelNames: list[str]) -> None:
			if not modelNames:
				# Translators: Reported when Ollama has no local models.
				ui.message(_("No Ollama models were found."))
				return
			self._setKnownModels(modelNames)
			if self.model == model and (not self.model or self.model not in modelNames):
				self.model = modelNames[0]
			# Translators: Reported after fetching Ollama models. {count} is the number of models.
			ui.message(_("Ollama models loaded: {count}").format(count=len(modelNames)))

		runSettingsAction(
			evt.GetEventObject(),
			self,
			lambda: OllamaClient(*key).getModelNames(),
			onSuccess,
			# Translators: Reported when fetching Ollama models fails. {message} is the error message.
			lambda message: ui.message(_("Could not fetch Ollama models: {message}").format(message=message)),
			lambda: key == (self.apiBaseUrl, self.apiKey),
		)

	def _makeOllamaClient(self) -> OllamaClient:
		return OllamaClient(self.apiBaseUrl, self.apiKey)

	def _fetchModelNames(self) -> list[str]:
		return self._makeOllamaClient().getModelNames()

	def _resolveModel(self) -> str:
		if self.model:
			return self.model
		modelNames = self._fetchModelNames()
		if not modelNames:
			# Translators: An error message when Ollama has no local models to use.
			raise ApiError(_("No Ollama models are available. Pull a vision model in Ollama first."))
		self._setKnownModels(modelNames)
		self.model = modelNames[0]
		return self.model

	def _buildOllamaRequestParams(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
		return self._makeOllamaClient().buildRequestParams(endpoint, payload)
