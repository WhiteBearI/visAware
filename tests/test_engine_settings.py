from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1] / "addon/globalPlugins/visAware"
PACKAGE = "_visaware_settings_test"


@pytest.fixture
def framework(monkeypatch):
	def stub(name, **attributes):
		module = ModuleType(name)
		module.__dict__.update(attributes)
		monkeypatch.setitem(sys.modules, name, module)
		return module

	def load(name):
		spec = importlib.util.spec_from_file_location(f"{PACKAGE}.{name}", ROOT / f"{name}.py")
		module = importlib.util.module_from_spec(spec)
		monkeypatch.setitem(sys.modules, spec.name, module)
		spec.loader.exec_module(module)
		return module

	stub(PACKAGE, __path__=[str(ROOT)])
	monkeypatch.setattr("builtins._", lambda text: text, raising=False)
	stub("addonHandler", initTranslation=lambda: None)
	stub("baseObject", AutoPropertyObject=type("AutoPropertyObject", (), {}))
	config = stub("config", conf={})
	stub("configobj", __path__=[])
	stub("configobj.validate", is_boolean=lambda value: value == "True", is_integer=int, is_list=list)
	stub(
		"synthDriverHandler",
		StringParameterInfo=lambda id, displayName: SimpleNamespace(id=id, displayName=displayName),
	)
	stub("logHandler", log=SimpleNamespace(debug=lambda *args: None, debugWarning=lambda *args: None))
	settings = load("engineSettings")
	# Runtime configuration must remain importable without settings windows.
	with monkeypatch.context() as blocked:
		blocked.setitem(sys.modules, "wx", None)
		blocked.setitem(sys.modules, "gui", None)
		core = load("abstractEngine")

	threads, callbacks = [], []
	stub("wx", __path__=[], CallAfter=lambda callback, *args: callbacks.append(lambda: callback(*args)))
	stub("wx.lib", __path__=[])
	stub("wx.lib.expando", ExpandoTextCtrl=object)
	stub("gui", __path__=[])
	stub("gui.guiHelper", BoxSizerHelper=object)
	stub("gui.nvdaControls", CustomCheckListBox=object, EnhancedInputSlider=object)
	stub("gui.settingsDialogs", SettingsPanel=type("SettingsPanel", (), {}), SettingsDialog=object)
	load("engineGUIHelper")
	ui = load("engineSettingsUI")
	monkeypatch.setattr(
		ui, "Thread", lambda **kwargs: SimpleNamespace(start=lambda: threads.append(kwargs["target"]))
	)
	return SimpleNamespace(
		core=core, settings=settings, ui=ui, config=config, threads=threads, callbacks=callbacks
	)


@pytest.mark.parametrize("saveFails", [False, True])
def test_config_round_trip_preserves_aliases_types_and_automatic_overrides(framework, saveFails):
	f = framework
	setting = f.settings.TextInputEngineSetting(
		"autoRecognitionPrompt", "Automatic prompt", configKey="autoRecognitionPromptV2"
	)
	declarations = [
		setting,
		f.settings.BooleanEngineSetting("streaming", "Streaming"),
		f.settings.NumericEngineSetting("quality", "Quality"),
		f.settings.ButtonEngineSetting("fetch", "Fetch"),
	]

	class Engine(f.core.AbstractEngine):
		name = "test"
		configSectionName = "ocr"
		supportedSettings = declarations
		autoRecognitionPrompt = "edited in another panel"
		streaming = False
		quality = 75

		@classmethod
		def check(cls):
			return True

		def saveSettings(self):
			super().saveSettings()
			if saveFails:
				raise OSError("config save failure")

	engine = Engine()
	conf = {"autoRecognitionPromptV2": "stored override", "streaming": "True", "quality": "80"}
	f.config.conf = {"ocr": {"test": conf}}
	panel = object.__new__(f.ui.SpecificEnginePanel)
	if saveFails:
		with pytest.raises(OSError):
			panel._saveSettingsWithoutAutomaticRecognitionOverrides(engine)
	else:
		panel._saveSettingsWithoutAutomaticRecognitionOverrides(engine)
	assert conf == {"autoRecognitionPromptV2": "stored override", "streaming": False, "quality": 75}
	assert "autoRecognitionPrompt" not in engine.getConfigSpec()
	assert "fetch" not in conf
	conf.update(streaming="True", quality="80")
	engine.loadSettings()
	assert engine.autoRecognitionPrompt == "stored override"
	assert engine.streaming is True
	assert engine.quality == 80


class Button:
	def __init__(self, parent):
		self.parent = parent
		self.enabled = True
		self.alive = True

	def __bool__(self):
		return self.alive

	def GetParent(self):
		assert self.alive
		return self.parent

	def Enable(self, enabled=True):
		assert self.alive
		self.enabled = enabled

	def Disable(self):
		assert self.alive
		self.enabled = False


@pytest.mark.parametrize(
	"outcome", ["success", "error", "destroy", "switch", "credentials", "save", "discard"]
)
def test_settings_action_applies_only_to_current_editor_on_main_thread(framework, outcome):
	f = framework
	engine = object()
	results, failures, refreshes = [], [], []
	parent = object.__new__(f.ui.SpecificEnginePanel)
	parent._getEngine = lambda: engine
	parent._loadedEngines = []
	parent._saveSettingsWithoutAutomaticRecognitionOverrides = lambda candidate: None
	parent._discardLoadedEngines = lambda savedEngine=None: None
	parent.updateDriverSettings = lambda: refreshes.append(True)
	button = Button(parent)
	current = [True]

	def action():
		if outcome == "error":
			raise ValueError("network failure")
		return ["model"]

	args = (button, engine, action, results.append, failures.append, lambda: current[0])
	f.ui.runSettingsAction(*args)
	f.ui.runSettingsAction(*args)
	assert len(f.threads) == 1
	assert not button.enabled
	f.threads.pop()()
	assert not results and not failures and not refreshes
	if outcome == "destroy":
		button.alive = False
	elif outcome == "switch":
		parent.isEditingEngine = lambda candidate: False
	elif outcome == "credentials":
		current[0] = False
	elif outcome in {"save", "discard"}:
		getattr(parent, "onSave" if outcome == "save" else "onDiscard")()
	f.callbacks.pop()()
	assert results == ([["model"]] if outcome == "success" else [])
	assert failures == (["network failure"] if outcome == "error" else [])
	assert refreshes == ([] if outcome == "destroy" else [True])
	if button.alive:
		assert button.enabled


def test_hidden_custom_url_keeps_creation_order_when_it_becomes_visible(framework):
	f = framework
	settings = [
		f.settings.TextInputEngineSetting("key", "Key"),
		f.settings.ChoiceEngineSetting("service", "Service", "services"),
		f.settings.TextInputEngineSetting("url", "URL", refreshSettingsOnChange=True),
		f.settings.ButtonEngineSetting("detect", "Detect"),
		f.settings.EditableChoiceEngineSetting("model", "Model", "models"),
	]
	custom = [False]
	engine = SimpleNamespace(supportedSettings=settings, isSupported=lambda name: name != "url" or custom[0])
	panel = object.__new__(f.ui.SpecificEnginePanel)
	panel._getEngine = lambda: engine
	panel._loadedEngines = []
	panel._sizerDict = {}
	created, shown, hidden = [], [], []
	panel.settingsSizer = SimpleNamespace(Hide=hidden.append, Layout=lambda: None)

	def create(setting, engine):
		created.append(setting.name)
		panel._sizerDict[setting.name] = setting.name

	panel._createControl = create
	panel._updateControlValue = lambda setting, engine: shown.append(setting.name)
	panel.updateDriverSettings()
	assert created == ["key", "service", "url", "detect", "model"]
	assert hidden == ["url"]
	custom[0] = True
	panel.updateDriverSettings("service")
	assert created == ["key", "service", "url", "detect", "model"]
	assert "url" in shown
