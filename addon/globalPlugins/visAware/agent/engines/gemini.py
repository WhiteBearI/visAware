# Copyright (C) 2026 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Gemini computer-use agent engine."""

from __future__ import annotations

from typing import Any

import addonHandler

from ...engineSettings import ChoiceEngineSetting
from ...geminiModels import (
	DEFAULT_GEMINI_AGENT_MODEL,
	DEFAULT_GEMINI_MEDIA_RESOLUTION,
	GEMINI_COMPUTER_USE_MODELS,
	getGeminiAgentModelChoices,
	getGeminiMediaResolutionChoices,
)
from ..._geminiEngine import GeminiEngineMixin
from ..gemini import GeminiAgentClient, GeminiAgentSettings
from ..engine import BaseAgentEngine

addonHandler.initTranslation()


class AgentEngine(GeminiEngineMixin, BaseAgentEngine):
	"""Gemini-backed computer-use agent engine."""

	name = "gemini"
	# Translators: The description of the Google Gemini Agent engine.
	description = _("Google Gemini")

	_model: str = DEFAULT_GEMINI_AGENT_MODEL
	_mediaResolution: str = DEFAULT_GEMINI_MEDIA_RESOLUTION

	@property
	def supportedSettings(self) -> list[Any]:
		"""
		Defines the user-configurable settings for this Agent engine.

		:returns: A list of engine setting objects.
		"""
		return [
			*self._geminiModelSettings(),
			ChoiceEngineSetting(
				name="mediaResolution",
				# Translators: The label for the Gemini media resolution used by the Agent.
				displayNameWithAccelerator=_("Media &resolution"),
				optionsPropertyName="availableMediaResolutions",
			),
			self._imageQualitySetting(),
		]

	@property
	def model(self) -> str:
		return self._model

	@model.setter
	def model(self, value: str) -> None:
		if value in self.availableModels:
			self._model = value
		else:
			self._model = DEFAULT_GEMINI_AGENT_MODEL

	@property
	def mediaResolution(self) -> str:
		return self._mediaResolution

	@mediaResolution.setter
	def mediaResolution(self, value: str) -> None:
		if value in self.availableMediaResolutions:
			self._mediaResolution = value

	@property
	def availableMediaResolutions(self) -> dict:
		"""
		Provides media-resolution choices for the Agent engine settings.

		:returns: A dictionary of media-resolution IDs to display names.
		"""
		return self.generateStringSettings(getGeminiMediaResolutionChoices())

	@classmethod
	def check(cls) -> bool:
		"""
		Checks whether this Agent engine can be used.

		:returns: True because Gemini is configured entirely through settings.
		"""
		return True

	def createClient(self) -> GeminiAgentClient:
		"""Creates a Gemini Agent runtime client."""
		return GeminiAgentClient(
			GeminiAgentSettings(
				apiKey=self.apiKey,
				model=self.model,
				imageQuality=self.imageQuality,
				mediaResolution=self.mediaResolution,
				source="Agent Gemini engine settings",
			),
		)

	def _getStaticModelChoices(self):
		return getGeminiAgentModelChoices()

	def _fetchModelNames(self, apiKey: str | None = None) -> list[str]:
		return [model for model in super()._fetchModelNames(apiKey) if model in GEMINI_COMPUTER_USE_MODELS]
