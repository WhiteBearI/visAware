# Copyright (C) 2026 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""OpenAI-compatible Chat Completions Agent engine."""

from __future__ import annotations

from typing import Any

import addonHandler

from ..._modelSettings import KnownModelsMixin
from ..._openaiEngine import OpenAICompatibleEngineMixin
from ..engine import BaseAgentEngine
from ..openaiCompatible import OpenAICompatibleAgentClient, OpenAICompatibleAgentSettings

addonHandler.initTranslation()


class AgentEngine(OpenAICompatibleEngineMixin, KnownModelsMixin, BaseAgentEngine):
	"""Agent engine using an OpenAI-compatible Chat Completions endpoint."""

	name = "openaiCompatibleAgent"
	# Translators: The description of the OpenAI-compatible Agent engine.
	description = _("OpenAI-compatible")
	defaultBaseUrl = "https://api.openai.com/v1"
	defaultModel = ""

	@property
	def supportedSettings(self) -> list[Any]:
		return [*self._openAICompatibleSettings(), self._imageQualitySetting()]

	@classmethod
	def check(cls) -> bool:
		return True

	def createClient(self) -> OpenAICompatibleAgentClient:
		return OpenAICompatibleAgentClient(
			OpenAICompatibleAgentSettings(
				apiKey=self.apiKey,
				baseUrl=self.baseUrl,
				model=self.model,
				imageQuality=self.imageQuality,
				source="Agent OpenAI-compatible engine settings",
			),
		)
