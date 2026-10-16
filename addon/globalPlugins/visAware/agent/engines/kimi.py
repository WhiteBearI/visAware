# Copyright (C) 2026 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Kimi (Moonshot AI) computer-use agent engine."""

from __future__ import annotations

from typing import Any

import addonHandler

from ..._modelSettings import KnownModelsMixin
from ..._kimiEngine import KimiEngineMixin
from ...kimiConnection import applyKimiConnection
from ...kimiProvider import (
	KIMI_AUTO_MODEL,
)
from ..kimi import KimiAgentClient, KimiAgentSettings
from ..engine import BaseAgentEngine

addonHandler.initTranslation()


class AgentEngine(KimiEngineMixin, KnownModelsMixin, BaseAgentEngine):
	"""Kimi-backed computer-use agent engine."""

	name = "kimi"
	# Translators: The description of the Kimi Agent engine.
	description = _("Kimi")

	@property
	def supportedSettings(self) -> list[Any]:
		return [
			*self._kimiConnectionSettings(),
			self._kimiReasoningSetting(),
			self._imageQualitySetting(),
		]

	@property
	def availableModels(self) -> dict:
		models = {KIMI_AUTO_MODEL: _("Auto-select (on first use)")}
		models.update({modelName: modelName for modelName in self._getKnownModels()})
		if self.model and self.model != KIMI_AUTO_MODEL and self.model not in models:
			models[self.model] = self.model
		return self.generateStringSettings(models)

	@classmethod
	def check(cls) -> bool:
		return True

	def createClient(self) -> KimiAgentClient:
		key = (self.apiKey, self.service, self._baseUrl)
		connectionIsCurrent = self._connectionKey == key and bool(self._resolvedBaseUrl)

		def onConnectionResolved(baseUrl: str, models: list[str]) -> None:
			import wx

			wx.CallAfter(self._onConnectionResolved, key, baseUrl, models)

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
				onConnectionResolved=onConnectionResolved,
			),
		)

	def _onConnectionResolved(self, key: tuple[str, str, str], baseUrl: str, models: list[str]) -> None:
		if key == (self.apiKey, self.service, self._baseUrl):
			applyKimiConnection(self, key, baseUrl, models)
