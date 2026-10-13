# Copyright (C) 2025 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Helper classes for creating GUI settings panels for recognition engines."""

from __future__ import annotations

import wx
from gui.nvdaControls import CustomCheckListBox
from typing import TYPE_CHECKING

from .engineSettings import (
	CheckListEngineSetting,
	ChoiceEngineSetting,
	EditableChoiceEngineSetting,
	EngineSetting,
)

if TYPE_CHECKING:
	from .abstractEngine import AbstractEngine


class EngineSettingChanger:
	"""Functor which acts as a callback for GUI events to change numeric settings."""

	def __init__(self, setting: "EngineSetting", engine: "AbstractEngine"):
		"""
		Initializes the setting changer.

		:param setting: The engine setting this changer is for.
		:param engine: The engine instance to modify.
		"""
		self.engine = engine
		self.setting = setting

	def __call__(self, evt: wx.CommandEvent) -> None:
		"""
		Handles the GUI event to update the setting.

		:param evt: The wx event.
		"""
		evt.Skip()
		val = evt.GetSelection()
		setattr(self.engine, self.setting.name, val)


class TextInputEngineSettingChanger(EngineSettingChanger):
	"""Functor for GUI events to change text input settings."""

	def __init__(
		self,
		setting: "EngineSetting",
		engine: "AbstractEngine",
		panel: wx.Panel | None = None,
	):
		self.panel = panel
		super().__init__(setting, engine)

	def __call__(self, evt: wx.CommandEvent) -> None:
		"""
		Handles the GUI event to update the setting.

		:param evt: The wx event.
		"""
		evt.Skip()
		value = evt.GetString()
		if isinstance(self.setting, EditableChoiceEngineSetting):
			try:
				value = next(
					(
						option.id
						for option in getattr(self.engine, self.setting.optionsPropertyName).values()
						if option.displayName == value
					),
					value,
				)
			except AttributeError:
				pass
		setattr(self.engine, self.setting.name, value)
		if self.panel and hasattr(self.panel, "updateDriverSettings"):
			self.panel.updateDriverSettings(self.setting.name)


class StringEngineSettingChanger(EngineSettingChanger):
	"""Functor for GUI events to change string (choice) settings."""

	def __init__(self, setting: "ChoiceEngineSetting", engine: "AbstractEngine", panel: wx.Panel):
		"""
		Initializes the setting changer.

		:param setting: The engine setting this changer is for.
		:param engine: The engine instance to modify.
		:param panel: The parent panel containing the control.
		"""
		self.panel = panel
		super().__init__(setting, engine)

	def __call__(self, evt: wx.CommandEvent) -> None:
		"""
		Handles the GUI event to update the setting.

		:param evt: The wx event.
		"""
		evt.Skip()
		try:
			# Use the explicitly provided property name.
			availableValues = getattr(self.engine, self.setting.optionsPropertyName).values()
			newValue = list(availableValues)[evt.GetSelection()].id
			setattr(self.engine, self.setting.name, newValue)
			if hasattr(self.panel, "updateDriverSettings"):
				self.panel.updateDriverSettings(self.setting.name)
		except AttributeError:
			pass


class CheckListEngineSettingChanger(EngineSettingChanger):
	"""Functor for GUI events to change checklist settings."""

	def __init__(
		self,
		setting: "CheckListEngineSetting",
		engine: "AbstractEngine",
		checkListBox: CustomCheckListBox,
	):
		"""
		Initializes the setting changer.

		:param setting: The engine setting this changer is for.
		:param engine: The engine instance to modify.
		:param checkListBox: The checklist box control.
		"""
		super().__init__(setting, engine)
		self.checkList = checkListBox

	def __call__(self, evt: wx.CommandEvent) -> None:
		"""
		Handles the GUI event to update the setting.

		:param evt: The wx event.
		"""
		evt.Skip()
		checkedStrings = self.checkList.GetCheckedStrings()
		# Use the explicitly provided property name.
		availableSettings = getattr(self.engine, self.setting.optionsPropertyName).values()
		descToId = {opt.displayName: opt.id for opt in availableSettings}
		result = [descToId[s] for s in checkedStrings]
		setattr(self.engine, self.setting.name, result)
