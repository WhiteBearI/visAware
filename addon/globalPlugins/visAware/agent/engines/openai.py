# Copyright (C) 2026 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""OpenAI Responses API computer-use agent engine."""

from __future__ import annotations

from typing import Any

import addonHandler
from ..._modelSettings import KnownModelsMixin
from ..._openaiEngine import OpenAICompatibleEngineMixin
from ..openai import (
	DEFAULT_OPENAI_BASE_URL,
	DEFAULT_OPENAI_MODEL,
	OpenAIAgentClient,
	OpenAIAgentSettings,
)
from ..engine import BaseAgentEngine

addonHandler.initTranslation()


class AgentEngine(OpenAICompatibleEngineMixin, KnownModelsMixin, BaseAgentEngine):
	"""OpenAI-backed computer-use agent engine."""

	name = "openai"
	# Translators: The description of the OpenAI Agent engine.
	description = _("OpenAI")

	defaultBaseUrl = DEFAULT_OPENAI_BASE_URL
	defaultModel = DEFAULT_OPENAI_MODEL

	@property
	def supportedSettings(self) -> list[Any]:
		return [
			*self._openAICompatibleSettings(),
			self._imageQualitySetting(),
		]

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

	def _fetchModelNames(self, apiKey: str | None = None, baseUrl: str | None = None) -> list[str]:
		settings = self._clientSettings()
		if apiKey is not None or baseUrl is not None:
			settings = OpenAIAgentSettings(
				apiKey=apiKey if apiKey is not None else settings.apiKey,
				baseUrl=baseUrl if baseUrl is not None else settings.baseUrl,
				model=settings.model,
				imageQuality=settings.imageQuality,
				source=settings.source,
			)
		return OpenAIAgentClient(settings).listModels()
