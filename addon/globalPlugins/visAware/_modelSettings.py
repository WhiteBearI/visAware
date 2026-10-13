# Copyright (C) 2026 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Remembered model lists shared by engine settings."""

import config


class KnownModelsMixin:
	engineConfigSpec = {"knownModels": "list(default=list())"}
	_knownModels: list[str]

	def loadSettings(self, onlyChanged: bool = False) -> None:
		super().loadSettings(onlyChanged=onlyChanged)
		self._loadKnownModelsFromConfig()

	def saveSettings(self) -> None:
		super().saveSettings()
		if self.configSectionName:
			config.conf[self.configSectionName][self.name]["knownModels"] = self._getKnownModels()

	def _loadKnownModelsFromConfig(self) -> None:
		if not self.configSectionName:
			return
		try:
			knownModels = config.conf[self.configSectionName][self.name]["knownModels"]
		except (KeyError, AttributeError):
			return
		if isinstance(knownModels, str):
			knownModels = [knownModels]
		self._setKnownModels([str(modelName) for modelName in knownModels])

	def _setKnownModels(self, modelNames: list[str]) -> None:
		self._knownModels = list(dict.fromkeys(name.strip() for name in modelNames if name.strip()))

	def _getKnownModels(self) -> list[str]:
		return list(getattr(self, "_knownModels", []))
