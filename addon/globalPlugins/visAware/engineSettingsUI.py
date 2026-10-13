# Copyright (C) 2026 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

from __future__ import annotations

from abc import ABC
from threading import Thread
from typing import Any, Callable, List, Optional, Type

import addonHandler
import config
import gui
from gui import guiHelper
from gui.guiHelper import BoxSizerHelper
from gui.nvdaControls import CustomCheckListBox, EnhancedInputSlider
from gui.settingsDialogs import SettingsPanel, SettingsDialog
from logHandler import log
import wx
from wx.lib.expando import ExpandoTextCtrl

from .abstractEngine import AbstractEngine, AbstractEngineHandler, _getSettingConfigKey
from .engineGUIHelper import (
	EngineSettingChanger,
	CheckListEngineSettingChanger,
	StringEngineSettingChanger,
	TextInputEngineSettingChanger,
)
from .engineSettings import (
	EngineSetting,
	BooleanEngineSetting,
	ButtonEngineSetting,
	CheckListEngineSetting,
	ChoiceEngineSetting,
	EditableChoiceEngineSetting,
	NumericEngineSetting,
	ReadOnlyEngineSetting,
	TextInputEngineSetting,
)

addonHandler.initTranslation()


def runSettingsAction(
	button: wx.Button,
	engine: AbstractEngine,
	action: Callable[[], Any],
	onSuccess: Callable[[Any], None],
	onError: Callable[[str], None],
	isCurrent: Callable[[], bool],
) -> None:
	if getattr(button, "_visAwareSettingsAction", False):
		return
	parent: Any = button.GetParent()
	generation = getattr(parent, "_settingsActionGeneration", None)
	setattr(button, "_visAwareSettingsAction", True)
	button.Disable()

	def finish(result: Any, error: str | None) -> None:
		if not button or not parent:
			return
		setattr(button, "_visAwareSettingsAction", False)
		button.Enable()
		if (
			generation is not getattr(parent, "_settingsActionGeneration", None)
			or not parent.isEditingEngine(engine)
			or not isCurrent()
		):
			parent.updateDriverSettings()
			return
		if error is not None:
			onError(error)
		else:
			onSuccess(result)
		parent.updateDriverSettings()

	def work() -> None:
		try:
			result = action()
		except Exception as e:
			wx.CallAfter(finish, None, str(e))
		else:
			wx.CallAfter(finish, result, None)

	try:
		Thread(target=work, daemon=True).start()
	except Exception:
		setattr(button, "_visAwareSettingsAction", False)
		button.Enable()
		raise


class SpecificEnginePanel(SettingsPanel):
	"""A settings panel for a specific engine's settings."""

	# Translators: The title of the engine-specific settings panel.
	title = _("Engine settings")
	handler: Type[AbstractEngineHandler]
	_getEngine: Callable[[], Optional[AbstractEngine]]
	_engine: Optional[AbstractEngine] = None
	_loadedEngines: List[AbstractEngine]
	_sizerDict: dict
	_lastControl: Optional[wx.Window]

	def __init__(
		self,
		parent: wx.Window,
		handler: Type[AbstractEngineHandler],
		getEngine: Callable[[], Optional[AbstractEngine]],
	):
		"""
		Initializes the panel for a specific engine handler.

		:param parent: The parent window.
		:param handler: The engine handler this panel is for.
		:param getEngine: A callable returning the engine currently selected in the settings panel.
		"""
		self.handler = handler
		self._getEngine = getEngine
		self._loadedEngines = []
		self._sizerDict = {}
		self._lastControl = None
		self._controlBuilders = {
			NumericEngineSetting: self._makeNumericSettingControl,
			BooleanEngineSetting: self._makeBooleanSettingControl,
			TextInputEngineSetting: self._makeTextInputSettingControl,
			ReadOnlyEngineSetting: self._makeReadOnlySettingControl,
			CheckListEngineSetting: self._makeCheckListSettingControl,
			ButtonEngineSetting: self._makeButtonSettingControl,
			ChoiceEngineSetting: self._makeStringSettingControl,
			EditableChoiceEngineSetting: self._makeEditableChoiceSettingControl,
		}
		self._controlUpdaters = {
			NumericEngineSetting: lambda s, e: getattr(self, f"{s.name}Slider").SetValue(getattr(e, s.name)),
			BooleanEngineSetting: lambda s, e: getattr(self, f"{s.name}Checkbox").SetValue(
				getattr(e, s.name),
			),
			TextInputEngineSetting: lambda s, e: getattr(self, f"{s.name}TextCtrl").ChangeValue(
				str(getattr(e, s.name)),
			),
			ReadOnlyEngineSetting: lambda s, e: getattr(self, f"{s.name}ExpandoTextCtrl").SetValue(
				str(getattr(e, s.name)),
			),
			CheckListEngineSetting: self._updateCheckListBox,
			ChoiceEngineSetting: self._updateChoiceControl,
			EditableChoiceEngineSetting: self._updateEditableChoiceControl,
			ButtonEngineSetting: lambda s, e: getattr(self, f"{s.name}Button").SetLabel(
				s.displayNameWithAccelerator,
			),
		}
		self._eventUnbinders = {
			CheckListEngineSetting: lambda s: getattr(self, f"{s.name}CheckListBox").Unbind(
				wx.EVT_CHECKLISTBOX,
			),
			ChoiceEngineSetting: lambda s: getattr(self, f"{s.name}List").Unbind(wx.EVT_CHOICE),
			EditableChoiceEngineSetting: self._unbindEditableChoiceControl,
		}
		super().__init__(parent)

	def makeSettings(self, settingsSizer: wx.BoxSizer) -> None:
		"""
		Creates the settings controls for the current engine.

		:param settingsSizer: The sizer to which controls will be added.
		"""
		self._sizerDict.clear()
		self._lastControl = None
		self.updateDriverSettings()

	def isEditingEngine(self, engine: AbstractEngine) -> bool:
		return self._getEngine() is engine

	def onPanelActivated(self) -> None:
		"""Refreshes the panel when it becomes active."""
		engine = self._getEngine()
		if engine and engine is not self._engine:
			log.debug("Engine changed, refreshing specific engine settings panel.")
			self.settingsSizer.Clear(delete_windows=True)
			self.makeSettings(self.settingsSizer)
		super().onPanelActivated()

	def updateDriverSettings(self, changedSetting: Optional[str] = None) -> None:
		"""Updates the GUI controls to reflect the current engine's settings."""
		engine = self._engine = self._getEngine()
		if not engine:
			return
		self._rememberLoadedEngine(engine)

		for name, sizer in self._sizerDict.items():
			if name != changedSetting and not engine.isSupported(name):
				self.settingsSizer.Hide(sizer)

		for setting in engine.supportedSettings:
			if setting.name.startswith("autoRecognition"):
				continue
			isSupported = engine.isSupported(setting.name)
			createBeforeSupported = (
				isinstance(setting, TextInputEngineSetting) and setting.refreshSettingsOnChange
			)
			if not isSupported and not createBeforeSupported:
				continue
			if setting.name == changedSetting:
				continue
			if setting.name in self._sizerDict:
				if isSupported:
					self._updateControlValue(setting, engine)
			else:
				self._createControl(setting, engine)
				if not isSupported:
					self.settingsSizer.Hide(self._sizerDict[setting.name])

		self.settingsSizer.Layout()

	def _createControl(self, setting: EngineSetting, engine: AbstractEngine) -> None:
		"""
		Creates and adds a new control for a given setting.

		:param setting: The `EngineSetting` to create a control for.
		:param engine: The current engine instance.
		"""
		sHelper = BoxSizerHelper(self, sizer=self.settingsSizer)
		builder = self._controlBuilders.get(type(setting))
		if not builder:
			builder = self._makeStringSettingControl

		controlSizer = builder(setting, engine)
		self._sizerDict[setting.name] = controlSizer
		sHelper.addItem(
			controlSizer,
			flag=wx.EXPAND if isinstance(setting, TextInputEngineSetting) and setting.multiline else 0,
		)

	def _updateControlValue(self, setting: EngineSetting, engine: AbstractEngine) -> None:
		"""
		Updates the value of an existing control.

		:param setting: The `EngineSetting` for the control.
		:param engine: The current engine instance.
		"""
		self.settingsSizer.Show(self._sizerDict[setting.name])
		updater = self._controlUpdaters.get(type(setting))
		if updater:
			updater(setting, engine)

	def _updateCheckListBox(self, setting: CheckListEngineSetting, engine: AbstractEngine) -> None:
		"""Updates a checklist box control."""
		paramIds = getattr(engine, setting.name, [])
		try:
			options = getattr(engine, setting.optionsPropertyName).values()
			idToDescription = {opt.id: opt.displayName for opt in options}
			checkedStrings = [idToDescription[id] for id in paramIds if id in idToDescription]
			getattr(self, f"{setting.name}CheckListBox").SetCheckedStrings(checkedStrings)
		except (AttributeError, ValueError):
			pass

	def _updateChoiceControl(self, setting: ChoiceEngineSetting, engine: AbstractEngine) -> None:
		"""Updates a choice (combobox) control."""
		try:
			options = list(getattr(engine, setting.optionsPropertyName).values())
			choiceCtrl = getattr(self, f"{setting.name}List")
			choiceCtrl.Clear()
			choiceCtrl.AppendItems([opt.displayName for opt in options])
			currentValue = getattr(engine, setting.name)
			index = [opt.id for opt in options].index(currentValue)
			choiceCtrl.SetSelection(index)
		except (AttributeError, ValueError):
			pass

	def _updateEditableChoiceControl(
		self,
		setting: EditableChoiceEngineSetting,
		engine: AbstractEngine,
	) -> None:
		"""Updates an editable choice control."""
		try:
			options = list(getattr(engine, setting.optionsPropertyName).values())
			choiceCtrl = getattr(self, f"{setting.name}ComboBox")
			with wx.EventBlocker(choiceCtrl):
				choiceCtrl.Clear()
				choiceCtrl.AppendItems([opt.displayName for opt in options])
				currentValue = str(getattr(engine, setting.name))
				currentDisplay = next(
					(opt.displayName for opt in options if opt.id == currentValue),
					currentValue,
				)
				choiceCtrl.ChangeValue(currentDisplay)
		except AttributeError:
			pass

	def _makeNumericSettingControl(
		self,
		setting: NumericEngineSetting,
		engine: AbstractEngine,
	) -> wx.BoxSizer:
		sizer = wx.BoxSizer(wx.HORIZONTAL)
		label = wx.StaticText(self, label=f"{setting.displayNameWithAccelerator}:")
		slider = EnhancedInputSlider(self, minValue=setting.minVal, maxValue=setting.maxVal)
		setattr(self, f"{setting.name}Slider", slider)
		slider.Bind(wx.EVT_SLIDER, EngineSettingChanger(setting, engine))
		slider.SetLineSize(setting.minStep)
		slider.SetPageSize(setting.largeStep)
		slider.SetValue(getattr(engine, setting.name))
		sizer.Add(label, flag=wx.ALIGN_CENTER_VERTICAL)
		sizer.Add(slider, proportion=1, flag=wx.EXPAND | wx.LEFT, border=5)
		if self._lastControl:
			slider.MoveAfterInTabOrder(self._lastControl)
		self._lastControl = slider
		return sizer

	def _makeStringSettingControl(self, setting: ChoiceEngineSetting, engine: AbstractEngine) -> wx.BoxSizer:
		labelText = f"{setting.displayNameWithAccelerator}:"
		choices = [x.displayName for x in getattr(engine, setting.optionsPropertyName).values()]
		labeledControl = guiHelper.LabeledControlHelper(self, labelText, wx.Choice, choices=choices)
		choiceCtrl = labeledControl.control
		setattr(self, f"{setting.name}List", choiceCtrl)
		self._updateChoiceControl(setting, engine)
		choiceCtrl.Bind(wx.EVT_CHOICE, StringEngineSettingChanger(setting, engine, self))
		if self._lastControl:
			choiceCtrl.MoveAfterInTabOrder(self._lastControl)
		self._lastControl = choiceCtrl
		return labeledControl.sizer

	def _makeEditableChoiceSettingControl(
		self,
		setting: EditableChoiceEngineSetting,
		engine: AbstractEngine,
	) -> wx.BoxSizer:
		labelText = f"{setting.displayNameWithAccelerator}:"
		choices = [x.displayName for x in getattr(engine, setting.optionsPropertyName).values()]
		labeledControl = guiHelper.LabeledControlHelper(
			self,
			labelText,
			wx.ComboBox,
			choices=choices,
			style=wx.CB_DROPDOWN,
		)
		choiceCtrl = labeledControl.control
		setattr(self, f"{setting.name}ComboBox", choiceCtrl)
		currentValue = str(getattr(engine, setting.name))
		currentDisplay = next(
			(
				opt.displayName
				for opt in getattr(engine, setting.optionsPropertyName).values()
				if opt.id == currentValue
			),
			currentValue,
		)
		choiceCtrl.SetValue(currentDisplay)
		changer = TextInputEngineSettingChanger(setting, engine, self)
		choiceCtrl.Bind(wx.EVT_TEXT, changer)
		choiceCtrl.Bind(wx.EVT_COMBOBOX, changer)
		if self._lastControl:
			choiceCtrl.MoveAfterInTabOrder(self._lastControl)
		self._lastControl = choiceCtrl
		return labeledControl.sizer

	def _unbindEditableChoiceControl(self, setting: EditableChoiceEngineSetting) -> None:
		choiceCtrl = getattr(self, f"{setting.name}ComboBox")
		choiceCtrl.Unbind(wx.EVT_TEXT)
		choiceCtrl.Unbind(wx.EVT_COMBOBOX)

	def _makeButtonSettingControl(self, setting: ButtonEngineSetting, engine: AbstractEngine) -> wx.Sizer:
		sizer = wx.BoxSizer(wx.HORIZONTAL)
		button = wx.Button(self, label=setting.displayNameWithAccelerator)
		setattr(self, f"{setting.name}Button", button)
		button.Bind(wx.EVT_BUTTON, getattr(engine, f"{setting.name}Changer"))
		sizer.Add(button)
		if self._lastControl:
			button.MoveAfterInTabOrder(self._lastControl)
		self._lastControl = button
		return sizer

	def _makeCheckListSettingControl(
		self,
		setting: CheckListEngineSetting,
		engine: AbstractEngine,
	) -> wx.Sizer:
		sizer = wx.WrapSizer(wx.HORIZONTAL)
		label = wx.StaticText(self, label=f"{setting.displayNameWithAccelerator}:")
		sizer.Add(label)
		items = [x.displayName for x in getattr(engine, setting.optionsPropertyName).values()]
		checkListBox = CustomCheckListBox(self, choices=items)
		sizer.Add(checkListBox, flag=wx.EXPAND | wx.LEFT, border=5)
		setattr(self, f"{setting.name}CheckListBox", checkListBox)
		self._updateCheckListBox(setting, engine)
		checkListBox.Select(0)
		checkListBox.Bind(wx.EVT_CHECKLISTBOX, CheckListEngineSettingChanger(setting, engine, checkListBox))
		if self._lastControl:
			checkListBox.MoveAfterInTabOrder(self._lastControl)
		self._lastControl = checkListBox
		return sizer

	def _makeBooleanSettingControl(self, setting: BooleanEngineSetting, engine: AbstractEngine) -> wx.Sizer:
		sizer = wx.BoxSizer(wx.HORIZONTAL)
		checkbox = wx.CheckBox(self, label=setting.displayNameWithAccelerator)
		setattr(self, f"{setting.name}Checkbox", checkbox)
		checkbox.Bind(wx.EVT_CHECKBOX, lambda evt: setattr(engine, setting.name, evt.IsChecked()))
		checkbox.SetValue(bool(getattr(engine, setting.name)))
		sizer.Add(checkbox)
		if self._lastControl:
			checkbox.MoveAfterInTabOrder(self._lastControl)
		self._lastControl = checkbox
		return sizer

	def _makeTextInputSettingControl(
		self,
		setting: TextInputEngineSetting,
		engine: AbstractEngine,
	) -> wx.BoxSizer:
		labelText = f"{setting.displayNameWithAccelerator}:"
		if setting.multiline:
			sizer = wx.BoxSizer(wx.VERTICAL)
			label = wx.StaticText(self, label=labelText)
			textCtrl = wx.TextCtrl(
				self,
				size=(-1, self.scaleSize(75)),
				style=wx.TE_MULTILINE,
			)
			sizer.Add(label)
			sizer.AddSpacer(guiHelper.SPACE_BETWEEN_ASSOCIATED_CONTROL_VERTICAL)
			sizer.Add(textCtrl, proportion=1, flag=wx.EXPAND)
		else:
			labeledControl = guiHelper.LabeledControlHelper(
				self,
				labelText,
				wx.TextCtrl,
				size=(self.scaleSize(250), -1),
			)
			textCtrl = labeledControl.control
			sizer = labeledControl.sizer
		setattr(self, f"{setting.name}TextCtrl", textCtrl)
		textCtrl.SetValue(str(getattr(engine, setting.name)))
		textCtrl.Bind(
			wx.EVT_TEXT,
			TextInputEngineSettingChanger(
				setting,
				engine,
				self if setting.refreshSettingsOnChange else None,
			),
		)
		if self._lastControl:
			textCtrl.MoveAfterInTabOrder(self._lastControl)
		self._lastControl = textCtrl
		return sizer

	def _makeReadOnlySettingControl(
		self,
		setting: ReadOnlyEngineSetting,
		engine: AbstractEngine,
	) -> wx.BoxSizer:
		labelText = f"{setting.displayNameWithAccelerator}:"
		labeledControl = guiHelper.LabeledControlHelper(
			self,
			labelText,
			ExpandoTextCtrl,
			style=wx.TE_READONLY,
		)
		expandoTextCtrl = labeledControl.control
		setattr(self, f"{setting.name}ExpandoTextCtrl", expandoTextCtrl)
		expandoTextCtrl.SetValue(str(getattr(engine, setting.name)))
		if self._lastControl:
			expandoTextCtrl.MoveAfterInTabOrder(self._lastControl)
		self._lastControl = expandoTextCtrl
		return labeledControl.sizer

	def _rememberLoadedEngine(self, engine: AbstractEngine) -> None:
		"""Tracks loaded engine instances so unsaved in-memory edits can be discarded."""
		if not any(engine is loadedEngine for loadedEngine in self._loadedEngines):
			self._loadedEngines.append(engine)

	def _unbindCurrentControls(self) -> None:
		"""Unbinds events for controls belonging to the currently displayed engine."""
		engine = self._engine
		if not engine:
			return
		for setting in engine.supportedSettings:
			try:
				unbinder = self._eventUnbinders.get(type(setting))
				if unbinder:
					unbinder(setting)
			except AttributeError:
				pass

	def _discardLoadedEngines(self, savedEngine: Optional[AbstractEngine] = None) -> None:
		"""Reloads settings for loaded engine instances except the one just saved."""
		for engine in self._loadedEngines:
			if engine is not savedEngine:
				engine.loadSettings()
		self._loadedEngines = [savedEngine] if savedEngine else []

	def onDiscard(self) -> None:
		"""Discards changes made in the panel."""
		self._settingsActionGeneration = object()
		currentEngine = self._getEngine()
		if currentEngine:
			self._rememberLoadedEngine(currentEngine)
		self._unbindCurrentControls()
		self._discardLoadedEngines()

	def onSave(self) -> None:
		"""Saves the settings from the panel."""
		self._settingsActionGeneration = object()
		engine = self._getEngine()
		if engine:
			self._saveSettingsWithoutAutomaticRecognitionOverrides(engine)
		self._discardLoadedEngines(savedEngine=engine)

	def _saveSettingsWithoutAutomaticRecognitionOverrides(self, engine: AbstractEngine) -> None:
		"""Saves regular engine settings without overwriting automatic recognition overrides."""
		if not engine.configSectionName:
			engine.saveSettings()
			return
		conf = config.conf[engine.configSectionName][engine.name]
		preservedValues = {
			_getSettingConfigKey(setting): conf[_getSettingConfigKey(setting)]
			for setting in engine.supportedSettings
			if setting.name.startswith("autoRecognition") and _getSettingConfigKey(setting) in conf
		}
		try:
			engine.saveSettings()
		finally:
			for name, value in preservedValues.items():
				conf[name] = value


class AbstractEngineSettingsPanel(SettingsPanel, ABC):
	"""Abstract base panel for engine selection and configuration."""

	# Translators: The default title for an engine settings panel.
	title = _("Engine")
	handler: Type[AbstractEngineHandler]
	_engineSettingPanel: "SpecificEnginePanel"
	_descEngineNameCtrl: ExpandoTextCtrl
	_selectedEngine: Optional[AbstractEngine] = None
	_enableEngineCheckBox: Optional[wx.CheckBox] = None
	_pendingDisabledEngineNames: set[str]

	def makeSettings(self, settingsSizer: wx.BoxSizer) -> None:
		"""
		Creates the settings controls for the panel.

		:param settingsSizer: The sizer to which controls will be added.
		"""
		self._enableEngineCheckBox = None
		self._pendingDisabledEngineNames = set(self.handler.getDisabledEngineNames())
		self._selectedEngine = self._getInitialSelectedEngine()
		settingsSizerHelper = guiHelper.BoxSizerHelper(self, sizer=settingsSizer)
		# Translators: The label for the group box to select an engine.
		engineLabel = _("Engines")
		engineBox = wx.StaticBox(self, label=engineLabel)
		engineGroupSizer = wx.StaticBoxSizer(engineBox, wx.HORIZONTAL)
		engineGroupHelper = guiHelper.BoxSizerHelper(self, sizer=engineGroupSizer)
		settingsSizerHelper.addItem(engineGroupHelper)

		engineDesc = self._selectedEngine.description if self._selectedEngine else ""
		self._descEngineNameCtrl = ExpandoTextCtrl(
			self,
			size=(self.scaleSize(250), -1),
			value=engineDesc,
			style=wx.TE_READONLY,
		)
		self._descEngineNameCtrl.Bind(wx.EVT_CHAR_HOOK, self._onEnterTriggersChangeEngine)

		# Translators: The label for a button to change the current engine.
		changeEngineBtn = wx.Button(self, label=_("C&hange..."))
		engineGroupHelper.addItem(guiHelper.associateElements(self._descEngineNameCtrl, changeEngineBtn))
		changeEngineBtn.Bind(wx.EVT_BUTTON, self._onChangeEngine)

		self._enableEngineCheckBox = wx.CheckBox(
			self,
			# Translators: The label for a checkbox that enables or disables the selected engine.
			label=_("&Enable this engine"),
		)
		engineGroupHelper.addItem(self._enableEngineCheckBox)
		self._updateEnableEngineCheckBox()

		self._engineSettingPanel = SpecificEnginePanel(self, self.handler, self._getSelectedEngine)
		settingsSizerHelper.addItem(self._engineSettingPanel, flag=wx.EXPAND)
		self.makeGeneralSettings(settingsSizerHelper)

	def _getInitialSelectedEngine(self) -> Optional[AbstractEngine]:
		"""Returns the engine initially selected for editing in this settings panel."""
		currentEngine = self.handler.getCurrentEngine()
		if currentEngine and currentEngine.name != "empty":
			return currentEngine
		for engineName, _description in self.handler.getConfigurableEngineList():
			try:
				return self.handler.getEngineInstance(engineName)
			except Exception:
				log.warning(f"Failed to load engine '{engineName}' for settings.", exc_info=True)
		return currentEngine

	def _getSelectedEngine(self) -> Optional[AbstractEngine]:
		"""Returns the engine currently selected for editing."""
		return self._selectedEngine

	def makeGeneralSettings(self, settingsSizerHelper: guiHelper.BoxSizerHelper) -> None:
		"""
		A placeholder for subclasses to add general settings below engine-specific ones.

		:param settingsSizerHelper: The sizer helper for adding controls.
		"""
		pass

	def _onEnterTriggersChangeEngine(self, evt: wx.KeyEvent) -> None:
		"""Handles the Enter key on the engine name control to open the selection dialog."""
		if evt.KeyCode == wx.WXK_RETURN:
			self._onChangeEngine(evt)
		else:
			evt.Skip()

	def _onChangeEngine(self, evt: wx.Event) -> None:
		"""Opens the engine selection dialog."""
		self._recordSelectedEngineEnabledState()
		initialEngineName = self._selectedEngine.name if self._selectedEngine else None
		dialog = EnginesSelectionDialog(
			self,
			self.handler,
			multiInstanceAllowed=True,
			disabledEngineNames=self._pendingDisabledEngineNames,
			initialEngineName=initialEngineName,
		)
		if dialog.ShowModal() == wx.ID_OK:
			self.Freeze()
			if dialog.selectedEngineName and self._selectEngine(dialog.selectedEngineName):
				self._engineSettingPanel.onPanelActivated()
				self._sendLayoutUpdatedEvent()
			self.Thaw()
		dialog.Destroy()

	def _selectEngine(self, engineName: str) -> bool:
		"""
		Selects an engine for editing without changing runtime engine enablement.

		:param engineName: The engine to display in this settings panel.
		:returns: True if the engine was loaded for editing.
		"""
		currentEngine = self.handler.getCurrentEngine()
		try:
			if currentEngine and currentEngine.name == engineName:
				self._selectedEngine = currentEngine
			else:
				self._selectedEngine = self.handler.getEngineInstance(engineName)
		except Exception:
			# Translators: An error message shown when an engine fails to load.
			gui.messageBox(
				_("Could not load the %s engine.") % engineName,
				# Translators: The title of an error dialog.
				_("Engine Error"),
				wx.OK | wx.ICON_WARNING,
				self,
			)
			return False
		self.updateCurrentEngine()
		return True

	def updateCurrentEngine(self) -> None:
		"""Updates the displayed selected engine name."""
		engine = self._selectedEngine
		self._descEngineNameCtrl.SetValue(engine.description if engine else "")
		self._updateEnableEngineCheckBox()

	def _updateEnableEngineCheckBox(self) -> None:
		"""Updates the current engine enablement checkbox."""
		if not self._enableEngineCheckBox:
			return
		engine = self._selectedEngine
		isConfigurableEngine = bool(engine and engine.name != "empty")
		self._enableEngineCheckBox.Enable(isConfigurableEngine)
		self._enableEngineCheckBox.SetValue(
			bool(isConfigurableEngine and engine.name not in self._pendingDisabledEngineNames),
		)

	def _recordSelectedEngineEnabledState(self) -> None:
		"""Stores the current checkbox state in the pending disabled engine set."""
		if not self._enableEngineCheckBox:
			return
		engine = self._selectedEngine
		if not engine or engine.name == "empty":
			return
		if self._enableEngineCheckBox.GetValue():
			self._pendingDisabledEngineNames.discard(engine.name)
		else:
			self._pendingDisabledEngineNames.add(engine.name)

	def _getSelectedEnabledEngineName(self) -> Optional[str]:
		"""Returns the selected engine name when it can be made active."""
		engine = self._selectedEngine
		if not engine or engine.name == "empty" or engine.name in self._pendingDisabledEngineNames:
			return None
		return engine.name

	def isValid(self) -> bool:
		"""Validates current engine enablement before settings are saved."""
		return self._ensurePendingEngineEnablementIsValid()

	def _ensurePendingEngineEnablementIsValid(self) -> bool:
		"""
		Ensures pending engine enablement leaves a usable current engine.

		:returns: True when saving can continue.
		"""
		self._recordSelectedEngineEnabledState()
		configurableEngineNames = [name for name, _description in self.handler.getConfigurableEngineList()]
		enabledEngineNames = [
			name for name in configurableEngineNames if name not in self._pendingDisabledEngineNames
		]
		if not enabledEngineNames:
			self._updateEnableEngineCheckBox()
			gui.messageBox(
				# Translators: A warning shown when trying to disable every engine in a category.
				_("At least one engine must remain enabled."),
				# Translators: The title of a warning shown for invalid engine enablement settings.
				_("Engine Configuration"),
				wx.OK | wx.ICON_WARNING,
				self,
			)
			return False
		selectedEngine = self._selectedEngine
		if (
			selectedEngine
			and selectedEngine.name != "empty"
			and selectedEngine.name in self._pendingDisabledEngineNames
		):
			gui.messageBox(
				# Translators: Shown when the selected engine is disabled and cannot be saved as the active engine.
				_(
					"The selected engine {engine} is disabled. Please choose another enabled engine, "
					"or re-enable it.",
				).format(
					engine=selectedEngine.description,
				),
				# Translators: The title of a warning shown for invalid engine enablement settings.
				_("Engine Configuration"),
				wx.OK | wx.ICON_WARNING,
				self,
			)
			return False
		currentEngine = self.handler.getCurrentEngine()
		if (
			currentEngine
			and currentEngine.name != "empty"
			and currentEngine.name in self._pendingDisabledEngineNames
		):
			if self._getSelectedEnabledEngineName():
				return True
			gui.messageBox(
				# Translators: Shown when the current engine is disabled and the user must choose or enable an engine before saving.
				_(
					"The current engine {engine} is disabled. Please choose another enabled engine, "
					"or re-enable it.",
				).format(
					engine=currentEngine.description,
				),
				# Translators: The title of a warning shown for invalid engine enablement settings.
				_("Engine Configuration"),
				wx.OK | wx.ICON_WARNING,
				self,
			)
			return False
		return True

	def onPanelActivated(self) -> None:
		"""Refreshes the engine-specific panel when this panel is activated."""
		self._engineSettingPanel.onPanelActivated()
		super().onPanelActivated()

	def onPanelDeactivated(self) -> None:
		"""Notifies the engine-specific panel when this panel is deactivated."""
		self._engineSettingPanel.onPanelDeactivated()
		super().onPanelDeactivated()

	def onSave(self) -> None:
		"""Saves settings from the engine-specific panel."""
		if not self._ensurePendingEngineEnablementIsValid():
			raise ValueError("Invalid engine enablement configuration.")
		self._engineSettingPanel.onSave()
		oldDisabledEngineNames = self.handler.getDisabledEngineNames()
		self.handler.setDisabledEngineNames(list(self._pendingDisabledEngineNames))
		newCurrentEngineName = self._getSelectedEnabledEngineName()
		if newCurrentEngineName:
			if not self.handler.setCurrentEngine(newCurrentEngineName):
				self.handler.setDisabledEngineNames(oldDisabledEngineNames)
				self._pendingDisabledEngineNames = set(oldDisabledEngineNames)
				# Translators: An error message shown when an engine fails to load.
				gui.messageBox(
					_("Could not load the %s engine.") % newCurrentEngineName,
					# Translators: The title of an error dialog.
					_("Engine Error"),
					wx.OK | wx.ICON_WARNING,
					self,
				)
				raise ValueError(f"Could not load engine {newCurrentEngineName}")
		self._pendingDisabledEngineNames = set(self.handler.getDisabledEngineNames())
		if newCurrentEngineName:
			currentEngine = self.handler.getCurrentEngine()
			if currentEngine and currentEngine.name != "empty":
				self._selectedEngine = currentEngine
		self.updateCurrentEngine()
		self._engineSettingPanel.onPanelActivated()

	def onDiscard(self) -> None:
		"""Discards changes from the engine-specific panel."""
		self._engineSettingPanel.onDiscard()
		self._pendingDisabledEngineNames = set(self.handler.getDisabledEngineNames())
		self._updateEnableEngineCheckBox()


class EnginesSelectionDialog(SettingsDialog):
	"""A dialog to select an engine from a list."""

	# Translators: The title of the engine selection dialog.
	title = _("Select Engine")
	_engineNames: List[str] = []
	_engineListCtrl: Optional[wx.Choice] = None
	handler: Type[AbstractEngineHandler]
	selectedEngineName: Optional[str] = None

	def __init__(
		self,
		parent: wx.Window,
		handler: Type[AbstractEngineHandler],
		multiInstanceAllowed: bool = True,
		disabledEngineNames: Optional[set[str]] = None,
		initialEngineName: Optional[str] = None,
	):
		"""
		Initializes the dialog.

		:param parent: The parent window.
		:param handler: The engine handler for which to select an engine.
		:param multiInstanceAllowed: Whether to allow multiple instances of this dialog.
		:param disabledEngineNames: Engine names treated as disabled in this selection session.
		:param initialEngineName: Engine name to preselect.
		"""
		self.handler = handler
		self._disabledEngineNames = disabledEngineNames or set()
		self._initialEngineName = initialEngineName
		self.selectedEngineName = None
		super().__init__(parent, multiInstanceAllowed=multiInstanceAllowed)

	def makeSettings(self, settingsSizer: wx.BoxSizer) -> None:
		"""
		Creates the settings controls for the dialog.

		:param settingsSizer: The sizer to which controls will be added.
		"""
		settingsSizerHelper = guiHelper.BoxSizerHelper(self, sizer=settingsSizer)
		# Translators: The label for the choice control to select an engine.
		engineListLabelText = _("&Engines:")
		self._engineListCtrl = settingsSizerHelper.addLabeledControl(
			engineListLabelText,
			wx.Choice,
			choices=[],
		)
		self._updateEngineList()

	def postInit(self) -> None:
		"""Sets focus to the engine list after initialization."""
		if self._engineListCtrl:
			self._engineListCtrl.SetFocus()

	def _updateEngineList(self) -> None:
		"""Populates the engine list control with available engines."""
		if not self._engineListCtrl:
			return
		driverList = self.handler.getConfigurableEngineList()
		self._engineNames = [x[0] for x in driverList]
		options = [self._formatEngineOption(name, description) for name, description in driverList]
		self._engineListCtrl.Clear()
		self._engineListCtrl.AppendItems(options)
		if self._initialEngineName:
			try:
				index = self._engineNames.index(self._initialEngineName)
				self._engineListCtrl.SetSelection(index)
			except ValueError:
				pass
		if self._engineListCtrl.GetSelection() == wx.NOT_FOUND and self._engineNames:
			self._engineListCtrl.SetSelection(0)

	def _formatEngineOption(self, engineName: str, engineDescription: str) -> str:
		"""
		Formats an engine option with its pending enablement status.

		:param engineName: The engine name.
		:param engineDescription: The localized engine description.
		:returns: The option shown in the selection control.
		"""
		if engineName in self._disabledEngineNames:
			# Translators: A disabled engine option in the engine selection dialog.
			return _("{engine} (disabled)").format(engine=engineDescription)
		# Translators: An enabled engine option in the engine selection dialog.
		return _("{engine} (enabled)").format(engine=engineDescription)

	def onOk(self, evt: wx.CommandEvent) -> None:
		"""Handles the OK button press to set and save the selected engine."""
		if not self._engineNames or not self._engineListCtrl:
			return
		newEngineName = self._engineNames[self._engineListCtrl.GetSelection()]
		try:
			self.handler.getEngine(newEngineName)
		except Exception:
			# Translators: An error message shown when an engine fails to load.
			gui.messageBox(
				_("Could not load the %s engine.") % newEngineName,
				# Translators: The title of an error dialog.
				_("Engine Error"),
				wx.OK | wx.ICON_WARNING,
				self,
			)
			return

		self.selectedEngineName = newEngineName
		super().onOk(evt)
