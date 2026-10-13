# Copyright (C) 2026 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""DeepSeek-backed computer-use agent engine."""

from __future__ import annotations

import addonHandler
from typing import Any

from ...deepseekModels import (
	DEFAULT_DEEPSEEK_BASE_URL,
	DEFAULT_DEEPSEEK_VISION_MODEL,
	getDeepSeekVisionModelChoices,
)
from ...engineSettings import ChoiceEngineSetting, TextInputEngineSetting
from ..deepseek import DeepSeekAgentClient, DeepSeekAgentSettings
from ..engine import BaseAgentEngine

addonHandler.initTranslation()


class AgentEngine(BaseAgentEngine):
	"""DeepSeek-backed computer-use agent engine."""

	name = "deepseek"
	# Translators: The description of the DeepSeek Agent engine.
	description = _("DeepSeek")

	_apiKey: str = ""
	_baseUrl: str = DEFAULT_DEEPSEEK_BASE_URL
	_model: str = DEFAULT_DEEPSEEK_VISION_MODEL

	@property
	def supportedSettings(self) -> list[Any]:
		return [
			TextInputEngineSetting(
				name="apiKey",
				# Translators: The label for the DeepSeek API key used by the Agent.
				displayNameWithAccelerator=_("API &key"),
			),
			TextInputEngineSetting(
				name="baseUrl",
				# Translators: The label for the DeepSeek base URL used by the Agent.
				displayNameWithAccelerator=_("Base &URL"),
				refreshSettingsOnChange=True,
			),
			ChoiceEngineSetting(
				name="model",
				# Translators: The label for the DeepSeek model used by the Agent.
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
		self._baseUrl = value.strip().rstrip("/") or DEFAULT_DEEPSEEK_BASE_URL

	@property
	def model(self) -> str:
		return self._model

	@model.setter
	def model(self, value: str) -> None:
		value = value.strip()
		self._model = value if value in self.availableModels else DEFAULT_DEEPSEEK_VISION_MODEL

	@property
	def availableModels(self) -> dict:
		return self.generateStringSettings(getDeepSeekVisionModelChoices())

	@classmethod
	def check(cls) -> bool:
		return True

	def createClient(self) -> "DeepSeekAgentClient":
		return DeepSeekAgentClient(
			DeepSeekAgentSettings(
				apiKey=self.apiKey,
				baseUrl=self.baseUrl,
				model=self.model,
				imageQuality=self.imageQuality,
				source="Agent DeepSeek engine settings",
			),
		)
