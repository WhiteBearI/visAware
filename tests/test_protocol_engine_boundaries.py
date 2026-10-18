from types import SimpleNamespace
import sys
import types

import pytest

from tests.test_kimi_image_describer import _load_module, load_kimi_image_module
from tests.test_kimi_ocr import load_kimi_ocr_module
from tests.test_kimi_agent_interactions import load_kimi_agent_module
from tests.test_request_log_redaction import load_recognition_settings_module
from tests import test_engine_settings
from tests.test_engine_settings import Button
from tests.test_deepseek_agent import _install_stubs as installDeepSeekStubs

framework = test_engine_settings.framework


@pytest.mark.parametrize("loader", [load_kimi_ocr_module, load_kimi_image_module])
@pytest.mark.parametrize("saveFails", [False, True])
def test_auto_model_save_restores_runtime_model(loader, saveFails, monkeypatch):
	module = loader()
	engine = module.CustomContentRecognizer()
	engine._model = "kimi-k3"
	engine._autoModelSelected = True
	saved = []

	def saveSettings(self):
		saved.append(self.model)
		if saveFails:
			raise OSError("Cannot save config")

	baseClass = module.CustomContentRecognizer.__bases__[1]
	monkeypatch.setattr(baseClass, "saveSettings", saveSettings, raising=False)
	if saveFails:
		with pytest.raises(OSError):
			engine.saveSettings()
	else:
		engine.saveSettings()
	assert saved == ["__auto__"]
	assert engine.model == "kimi-k3"
	assert engine._autoModelSelected

	engine.model = "custom-model"
	if saveFails:
		with pytest.raises(OSError):
			engine.saveSettings()
	else:
		engine.saveSettings()
	assert saved[-1] == "custom-model"
	assert engine.model == "custom-model"


@pytest.mark.parametrize("data", [None, [], [{"id": " b "}, {}, None, {"id": 1}, {"id": "a"}, {"id": "b"}]])
def test_model_discovery_preserves_order_and_transport_controls(data):
	load_kimi_image_module()
	module = sys.modules["addon.globalPlugins.visAware.openaiCompatible"]
	calls = []
	transport = SimpleNamespace(
		sendRequest=lambda *args, **kwargs: calls.append((args, kwargs))
		or SimpleNamespace(json=lambda: {"data": data}),
	)
	client = module.OpenAICompatibleClient(
		module.OpenAICompatibleSettings("secret", "https://api.example/v1"), transport,
	)

	def checker():
		pass

	assert client.listModels(timeout=4, cancelCheck=checker, retry=False) == (["b", "a"] if data else [])
	args, kwargs = calls[0]
	assert args == ("GET", "https://api.example/v1/models")
	assert kwargs["timeout"] == 4
	assert kwargs["cancelCheck"] is checker
	assert kwargs["retry"] is False


def test_model_discovery_uses_root_models_for_full_root_chat_endpoint():
	load_kimi_image_module()
	module = sys.modules["addon.globalPlugins.visAware.openaiCompatible"]

	assert module.buildModelsUrl("https://api.example/chat/completions") == ("https://api.example/models")


def test_responses_client_reuses_request_and_model_transport():
	load_kimi_image_module()
	module = _load_module(
		"addon.globalPlugins.visAware.openaiResponses",
		"addon/globalPlugins/visAware/openaiResponses.py",
	)
	calls = []
	responses = iter(
		[
			SimpleNamespace(json=lambda: {"id": "response-1", "output": []}),
			SimpleNamespace(json=lambda: {"data": [{"id": " z "}, {"id": "a"}, {"id": "z"}]}),
		],
	)
	transport = SimpleNamespace(
		sendRequest=lambda *args, **kwargs: calls.append((args, kwargs)) or next(responses),
	)
	client = module.ResponsesClient("secret", "https://api.example/v1", transport)

	def checker():
		pass

	assert client.create({"model": "vision"}, timeout=7, cancelCheck=checker)["id"] == "response-1"
	assert client.listModels(timeout=8, cancelCheck=checker) == ["a", "z"]
	assert calls[0][1]["url"] == "https://api.example/v1/responses"
	assert calls[0][1]["cancelCheck"] is checker
	assert calls[1][0] == ("GET", "https://api.example/v1/models")


@pytest.mark.parametrize("provider", ["kimi", "openai", "ollama"])
@pytest.mark.parametrize("stored", [" old-model ", [" b ", "", "a", "b"]])
def test_remembered_models_follow_engine_lifecycle_without_sharing_instances(provider, stored):
	installDeepSeekStubs()
	imageModule = load_kimi_image_module()
	modelSettings = _load_module(
		"addon.globalPlugins.visAware._modelSettings", "addon/globalPlugins/visAware/_modelSettings.py",
	)
	baseCalls = []

	class BaseEngine(imageModule.BaseDescriber):
		configSectionName = "visAwareAgent"

		def loadSettings(self, onlyChanged=False):
			baseCalls.append(("load", onlyChanged))

		def saveSettings(self):
			baseCalls.append(("save", self.model))

	settings = types.ModuleType("addon.globalPlugins.visAware.agent.engine")
	settings.BaseAgentEngine = BaseEngine
	sys.modules[settings.__name__] = settings
	if provider == "ollama":
		_load_module(
			"addon.globalPlugins.visAware._ollamaClient", "addon/globalPlugins/visAware/_ollamaClient.py",
		)
		module = _load_module(
			"addon.globalPlugins.visAware._ollamaEngine", "addon/globalPlugins/visAware/_ollamaEngine.py",
		)

		class Engine(module.OllamaEngineMixin, BaseEngine):
			name = "ollama"
	else:
		if provider == "kimi":
			load_kimi_agent_module()
		else:
			sys.modules["addon.globalPlugins.visAware.agent.actions"].NORMALIZED_SCALE = 1000
			_load_module(
				"addon.globalPlugins.visAware._openaiEngine",
				"addon/globalPlugins/visAware/_openaiEngine.py",
			)
			_load_module(
				"addon.globalPlugins.visAware.agent.openai", "addon/globalPlugins/visAware/agent/openai.py",
			)
		module = _load_module(
			f"addon.globalPlugins.visAware.agent.engines.{provider}",
			f"addon/globalPlugins/visAware/agent/engines/{provider}.py",
		)
		Engine = module.AgentEngine
	modelSettings.config.conf = {"visAwareAgent": {provider: {"knownModels": stored}}}
	engine = Engine()
	engine.loadSettings(onlyChanged=True)
	assert engine._getKnownModels() == (["old-model"] if isinstance(stored, str) else ["b", "a"])
	assert baseCalls == [("load", True)]
	assert engine.engineConfigSpec["knownModels"] == "list(default=list())"
	engine._setKnownModels([" new-model ", "", "new-model", "another"])
	copy = engine._getKnownModels()
	copy.append("should-not-leak")
	assert engine._getKnownModels() == ["new-model", "another"]
	assert Engine()._getKnownModels() == []
	engine.saveSettings()
	assert modelSettings.config.conf["visAwareAgent"][provider]["knownModels"] == ["new-model", "another"]
	assert baseCalls[-1][0] == "save"
	engine.configSectionName = None
	engine.saveSettings()
	assert baseCalls[-1][0] == "save"


@pytest.mark.parametrize(
	"imageUrl", ["data:image/jpeg;base64,AAAA", "https://images.example/private?signature=secret"],
)
@pytest.mark.parametrize("nested", [False, True])
def test_request_logging_handles_images_without_loading_recognition(monkeypatch, imageUrl, nested):
	installDeepSeekStubs()
	module = _load_module(
		"addon.globalPlugins.visAware.requestLogging", "addon/globalPlugins/visAware/requestLogging.py",
	)
	monkeypatch.delitem(sys.modules, "addon.globalPlugins.visAware.recogHandler", raising=False)
	original = {
		"headers": {"Authorization": "Bearer secret"},
		"input": [{"image_url": {"url": imageUrl} if nested else imageUrl}],
	}
	redacted = module.redactForLog(original)
	assert redacted["headers"]["Authorization"] == "<redacted>"
	image = redacted["input"][0]["image_url"]
	assert (image["url"] if nested else image).startswith("<redacted data URL:")
	assert original["headers"]["Authorization"] == "Bearer secret"
	assert "addon.globalPlugins.visAware.recogHandler" not in sys.modules
	module.config.conf = {}
	assert not module.isVerboseDebugLoggingEnabled()
	module.config.conf = {"visAwareGeneral": {"verboseDebugLogging": True}}
	assert module.isVerboseDebugLoggingEnabled()


def _loadGoogleEngine(role):
	ocrModule = load_kimi_ocr_module()
	imageModule = load_kimi_image_module()
	sys.modules["addon.globalPlugins.visAware.recogHandler"].BaseRecognizer = ocrModule.BaseRecognizer
	sys.modules["logHandler"].log.debug = lambda *args, **kwargs: None
	sys.modules["config"] = SimpleNamespace(conf={})
	_load_module(
		"addon.globalPlugins.visAware.requestLogging", "addon/globalPlugins/visAware/requestLogging.py",
	)
	_load_module("addon.globalPlugins.visAware.geminiModels", "addon/globalPlugins/visAware/geminiModels.py")
	_load_module(
		"addon.globalPlugins.visAware.geminiGenerateContent",
		"addon/globalPlugins/visAware/geminiGenerateContent.py",
	)
	_load_module(
		"addon.globalPlugins.visAware._modelSettings",
		"addon/globalPlugins/visAware/_modelSettings.py",
	)
	_load_module(
		"addon.globalPlugins.visAware._geminiEngine",
		"addon/globalPlugins/visAware/_geminiEngine.py",
	)
	path = "contentRecognizers/geminiOCR" if role == "ocr" else f"imageDescribers/{role}"
	module = _load_module(
		f"addon.globalPlugins.visAware.{path.replace('/', '.')}", f"addon/globalPlugins/visAware/{path}.py",
	)
	return module, imageModule.RecognitionRequest


@pytest.mark.parametrize("role", ["ocr", "gemini", "gemma"])
def test_generate_content_role_requests_keep_headers_and_payload(role):
	module, requestClass = _loadGoogleEngine(role)
	engine = module.CustomContentRecognizer()
	engine.apiKey = "test-google-key"
	request = requestClass(streamResult=role == "gemini")
	request.prompt = "Describe this screen"
	params = engine._buildRequestParams(b"base64-image", request)
	method = "streamGenerateContent?alt=sse" if role == "gemini" else "generateContent"
	assert params["url"] == f"https://generativelanguage.googleapis.com/v1beta/models/{engine.model}:{method}"
	assert params["method"] == "POST"
	assert params["headers"]["x-goog-api-key"] == "test-google-key"
	parts = params["json"]["contents"][0]["parts"]
	assert parts[0]["inline_data"]["data"] == "base64-image"
	assert parts[1]["text"] == (module.OCR_PROMPT if role == "ocr" else request.prompt)
	if role == "ocr":
		assert params["json"]["generationConfig"]["responseJsonSchema"] == module.OCR_RESULT_SCHEMA
	elif role == "gemma":
		assert "systemInstruction" in params["json"]
		assert params["json"]["generationConfig"]["responseMimeType"] == "application/json"


def test_gemini_model_discovery_filters_models_by_family_and_protocol(monkeypatch):
	_loadGoogleEngine("gemini")
	module = sys.modules["addon.globalPlugins.visAware._geminiEngine"]
	responsePayload = {
		"models": [
			{"name": "models/gemini-4-argon", "supportedGenerationMethods": ["generateContent"]},
			{
				"name": "models/gemini-3.1-flash-lite-preview",
				"supportedGenerationMethods": ["generateContent"],
			},
			{"name": "models/gemini-3.1-flash-lite", "supportedGenerationMethods": ["generateContent"]},
			{
				"name": "models/gemini-2.5-flash-image-preview",
				"supportedGenerationMethods": ["generateContent"],
			},
			{
				"name": "models/gemini-2.5-flash-native-audio-preview",
				"supportedGenerationMethods": ["generateContent"],
			},
			{"name": "models/gemma-4-31b-it", "supportedGenerationMethods": ["generateContent"]},
			{"name": "models/gemma-3-1b-it", "supportedGenerationMethods": ["generateContent"]},
			{"name": "models/gemini-embedding-001", "supportedGenerationMethods": ["embedContent"]},
			{"name": "models/gemini-4-argon", "supportedGenerationMethods": ["generateContent"]},
		],
	}
	monkeypatch.setattr(
		module.network,
		"sendRequest",
		lambda *args, **kwargs: SimpleNamespace(json=lambda: responsePayload),
	)

	assert module.listGeminiModelNames("key") == ["gemini-4-argon", "gemini-3.1-flash-lite"]
	assert module.listGeminiModelNames("key", "gemma") == ["gemma-4-31b-it"]


@pytest.mark.parametrize("role,stream", [("gemini", False), ("gemini", True), ("gemma", False)])
def test_generate_content_follow_up_preserves_conversation(role, stream):
	module, _ = _loadGoogleEngine(role)
	engine = module.CustomContentRecognizer()
	engine.apiKey = "test-google-key"
	context = SimpleNamespace(
		image=object(),
		initialText="Original answer",
		turns=[SimpleNamespace(role="assistant", text="Previous answer")],
	)
	params = engine._buildQuestionRequestParams(context, "New question", stream=stream)
	method = "streamGenerateContent?alt=sse" if stream else "generateContent"
	assert params["url"].endswith(f":{method}")
	assert params["headers"]["x-goog-api-key"] == "test-google-key"
	contents = params["json"]["contents"]
	assert [item["role"] for item in contents] == ["user", "model", "model", "user"]
	assert contents[0]["parts"][0]["inline_data"]["data"] == "encoded-image"
	assert contents[1]["parts"] == [{"text": "Original answer"}]
	assert contents[2]["parts"] == [{"text": "Previous answer"}]
	assert contents[-1]["parts"] == [{"text": "New question"}]


def test_kimi_agent_keeps_auto_selection_and_discovered_models_when_saving():
	imageModule = load_kimi_image_module()
	load_kimi_agent_module()
	_load_module("addon.globalPlugins.visAware._kimiEngine", "addon/globalPlugins/visAware/_kimiEngine.py")

	class BaseAgentEngine(imageModule.BaseDescriber):
		configSectionName = "visAwareAgent"
		imageQuality = 90

		def saveSettings(self):
			sys.modules["config"].conf[self.configSectionName][self.name]["model"] = self.model

	settingsModule = types.ModuleType("addon.globalPlugins.visAware.agent.engine")
	settingsModule.BaseAgentEngine = BaseAgentEngine
	_load_module(
		"addon.globalPlugins.visAware._modelSettings", "addon/globalPlugins/visAware/_modelSettings.py",
	)
	sys.modules[settingsModule.__name__] = settingsModule
	module = _load_module(
		"addon.globalPlugins.visAware.agent.engines.kimi",
		"addon/globalPlugins/visAware/agent/engines/kimi.py",
	)
	sys.modules["config"].conf = {"visAwareAgent": {"kimi": {}}}
	engine = module.AgentEngine()
	engine.apiKey = "secret"
	engine._onConnectionResolved(
		(engine.apiKey, engine.service, engine._baseUrl),
		"https://api.moonshot.ai/v1",
		["kimi-k3", "kimi-k3", " "],
	)
	engine.saveSettings()
	assert sys.modules["config"].conf["visAwareAgent"]["kimi"] == {
		"model": "__auto__",
		"knownModels": ["kimi-k3"],
	}
	assert engine.model == "kimi-k3"
	client = engine.createClient()
	assert client.baseUrl == "https://api.moonshot.ai/v1"
	assert client.model == "kimi-k3"
	assert client._connectionResolved
	engine.apiKey = "new-secret"
	assert engine._getKnownModels() == []
	assert engine.model == "__auto__"
	assert not engine.createClient()._connectionResolved


@pytest.mark.parametrize("loader", [load_kimi_ocr_module, load_kimi_image_module])
def test_automatic_recognition_panel_can_detect_kimi_models(loader, monkeypatch, framework):
	recogModule = load_recognition_settings_module()
	engine = loader().CustomContentRecognizer()
	panel = recogModule.AutomaticRecognitionPanel()
	panel._autoRecognitionEngine = engine
	refreshed = []
	panel.updateDriverSettings = lambda: refreshed.append(True)
	button = Button(panel)
	event = SimpleNamespace(Skip=lambda: None, GetEventObject=lambda: button)
	monkeypatch.setitem(sys.modules, "addon.globalPlugins.visAware.engineSettingsUI", framework.ui)
	monkeypatch.setattr(
		sys.modules["addon.globalPlugins.visAware._kimiEngine"],
		"resolveKimiConnection",
		lambda *args, **kwargs: ("https://api.moonshot.cn/v1", ["kimi-k3"]),
	)
	monkeypatch.setitem(sys.modules, "ui", SimpleNamespace(message=lambda message: None))
	panel._onFetchAutoRecognitionModels(event)
	framework.threads.pop()()
	assert refreshed == []
	framework.callbacks.pop()()
	assert refreshed == [True]
	assert engine._resolvedModels == ["kimi-k3"]
	assert button.enabled


@pytest.mark.parametrize("loader", [load_kimi_ocr_module, load_kimi_image_module])
def test_runtime_discovery_rejects_connection_settings_changed_during_request(loader, monkeypatch):
	engine = loader().CustomContentRecognizer()
	engine.apiKey = "old-key"
	module = sys.modules["addon.globalPlugins.visAware.kimiConnection"]

	def resolve(apiKey, service, baseUrl, model, **kwargs):
		assert apiKey == "old-key"
		engine.apiKey = "new-key"
		return "https://old.example/v1", ["old-model"]

	monkeypatch.setattr(module, "resolveKimiConnection", resolve)
	with pytest.raises(module.CancellationError):
		engine._ensureConnection(forceDiscovery=True)
	assert engine._resolvedBaseUrl is None
	assert engine._resolvedModels == []
	assert engine.model == "__auto__"
	assert not module._KIMI_CONNECTION_CACHE


@pytest.mark.parametrize("models", [("saved-model", "chosen-model", "new-model"), ("new-model",)])
def test_auto_recognition_model_refresh_keeps_unsaved_selection(models):
	module = load_recognition_settings_module()
	panel = module.AutomaticRecognitionPanel()
	panel._autoRecognitionEngine = SimpleNamespace(
		autoRecognitionModel="saved-model",
		isSupported=lambda name: name == "model",
		availableModels={name: SimpleNamespace(id=name, displayName=name) for name in models},
	)
	panel.autoRecognitionModelChoices = [("", "Default"), ("chosen-model", "Chosen")]
	control = SimpleNamespace(selection=1, labels=[])
	control.GetSelection = lambda: control.selection
	control.Clear = control.labels.clear
	control.AppendItems = control.labels.extend
	control.SetSelection = lambda value: setattr(control, "selection", value)
	panel.autoRecognitionModelList = control
	panel._updateAutoRecognitionControlState = lambda: None
	panel.updateDriverSettings()
	assert panel.autoRecognitionModelChoices[control.selection][0] == "chosen-model"
	assert "new-model" in control.labels
	assert panel._autoRecognitionEngine.autoRecognitionModel == "saved-model"


def test_auto_recognition_fetch_button_tracks_pending_action_and_off_state():
	module = load_recognition_settings_module()
	panel = module.AutomaticRecognitionPanel()
	panel._autoRecognitionEngine = SimpleNamespace(fetchModelsChanger=lambda evt: None)
	mode = ["ocr"]
	panel._getSelectedAutoRecognitionType = lambda: mode[0]
	panel.autoRecognitionEngineChoices = [("current", "Current")]
	panel.autoRecognitionEngineList = SimpleNamespace(Enable=lambda enabled: None)
	button = panel.autoRecognitionFetchModelsButton = Button(panel)
	button._visAwareSettingsAction = True
	panel._updateAutoRecognitionControlState()
	assert not button.enabled
	button._visAwareSettingsAction = False
	panel._updateAutoRecognitionControlState()
	assert button.enabled
	mode[0] = "off"
	panel._updateAutoRecognitionControlState()
	assert not button.enabled


@pytest.mark.parametrize("provider", ["ollama", "openai"])
@pytest.mark.parametrize("edit", ["none", "credentials", "model"])
def test_model_fetch_preserves_connection_and_model_edits(provider, edit, framework, monkeypatch):
	installDeepSeekStubs()
	imageModule = load_kimi_image_module()
	_load_module(
		"addon.globalPlugins.visAware._modelSettings", "addon/globalPlugins/visAware/_modelSettings.py",
	)
	if provider == "ollama":
		_load_module(
			"addon.globalPlugins.visAware._ollamaClient", "addon/globalPlugins/visAware/_ollamaClient.py",
		)
		module = _load_module(
			"addon.globalPlugins.visAware._ollamaEngine", "addon/globalPlugins/visAware/_ollamaEngine.py",
		)

		class Engine(module.OllamaEngineMixin, imageModule.BaseDescriber):
			name = "ollama"

		engine = Engine()
	else:
		sys.modules["addon.globalPlugins.visAware.agent.actions"].NORMALIZED_SCALE = 1000
		_load_module(
			"addon.globalPlugins.visAware._openaiEngine",
			"addon/globalPlugins/visAware/_openaiEngine.py",
		)
		_load_module(
			"addon.globalPlugins.visAware.agent.openai", "addon/globalPlugins/visAware/agent/openai.py",
		)
		monkeypatch.setitem(
			sys.modules,
			"addon.globalPlugins.visAware.agent.engine",
			SimpleNamespace(BaseAgentEngine=imageModule.BaseDescriber),
		)
		module = _load_module(
			"addon.globalPlugins.visAware.agent.engines.openai",
			"addon/globalPlugins/visAware/agent/engines/openai.py",
		)
		engine = module.AgentEngine()
	monkeypatch.setitem(sys.modules, "addon.globalPlugins.visAware.engineSettingsUI", framework.ui)
	engine.apiKey = "old-key"
	requests = []

	def makeClient(*args):
		key = args[1] if provider == "ollama" else args[0].apiKey

		def fetch():
			requests.append(key)
			return ["discovered-model"]

		return SimpleNamespace(getModelNames=fetch, listModels=fetch)

	if provider == "ollama":
		monkeypatch.setattr(module, "OllamaClient", makeClient)
	else:
		engine.imageQuality = 90
		monkeypatch.setattr(module, "OpenAIAgentClient", makeClient)
	initialModel = engine.model
	parent = SimpleNamespace(
		isEditingEngine=lambda candidate: candidate is engine, updateDriverSettings=lambda: None,
	)
	button = Button(parent)
	engine.fetchModelsChanger(SimpleNamespace(Skip=lambda: None, GetEventObject=lambda: button))
	if edit == "credentials":
		engine.apiKey = "new-key"
	elif edit == "model":
		engine.model = "manual-model"
	framework.threads.pop()()
	assert requests == ["old-key"]
	assert engine._getKnownModels() == []
	framework.callbacks.pop()()
	assert engine._getKnownModels() == ([] if edit == "credentials" else ["discovered-model"])
	expectedModel = (
		"manual-model" if edit == "model" else initialModel or ("discovered-model" if edit == "none" else "")
	)
	assert engine.model == expectedModel


@pytest.mark.parametrize("changeKey", [False, True])
def test_kimi_agent_connection_callback_uses_client_snapshot_on_main_thread(changeKey, monkeypatch):
	installDeepSeekStubs()
	load_kimi_agent_module()
	module = _load_module(
		"addon.globalPlugins.visAware.agent.engines.kimi",
		"addon/globalPlugins/visAware/agent/engines/kimi.py",
	)
	callbacks = []
	monkeypatch.setattr(
		sys.modules["wx"],
		"CallAfter",
		lambda callback, *args: callbacks.append(lambda: callback(*args)),
		raising=False,
	)
	engine = module.AgentEngine()
	engine.apiKey = "old-key"
	client = engine.createClient()
	client._onConnectionResolved("https://resolved.example/v1", ["kimi-k3"])
	assert engine._resolvedBaseUrl is None
	if changeKey:
		engine.apiKey = "new-key"
	callbacks.pop()()
	assert engine._resolvedBaseUrl == (None if changeKey else "https://resolved.example/v1")
	assert engine._getKnownModels() == ([] if changeKey else ["kimi-k3"])
	assert engine.model == ("__auto__" if changeKey else "kimi-k3")


def test_openai_model_fetch_reports_missing_key_through_action_error(framework, monkeypatch):
	installDeepSeekStubs()
	imageModule = load_kimi_image_module()
	_load_module(
		"addon.globalPlugins.visAware._modelSettings", "addon/globalPlugins/visAware/_modelSettings.py",
	)
	monkeypatch.setitem(
		sys.modules,
		"addon.globalPlugins.visAware.agent.engine",
		SimpleNamespace(BaseAgentEngine=imageModule.BaseDescriber),
	)
	sys.modules["addon.globalPlugins.visAware.agent.actions"].NORMALIZED_SCALE = 1000
	_load_module(
		"addon.globalPlugins.visAware._openaiEngine",
		"addon/globalPlugins/visAware/_openaiEngine.py",
	)
	_load_module("addon.globalPlugins.visAware.agent.openai", "addon/globalPlugins/visAware/agent/openai.py")
	module = _load_module(
		"addon.globalPlugins.visAware.agent.engines.openai",
		"addon/globalPlugins/visAware/agent/engines/openai.py",
	)
	monkeypatch.setitem(sys.modules, "addon.globalPlugins.visAware.engineSettingsUI", framework.ui)
	messages = []
	monkeypatch.setattr(sys.modules["ui"], "message", messages.append)
	engine = module.AgentEngine()
	engine.imageQuality = 90
	parent = SimpleNamespace(
		isEditingEngine=lambda candidate: candidate is engine, updateDriverSettings=lambda: None,
	)
	button = Button(parent)
	engine.fetchModelsChanger(SimpleNamespace(Skip=lambda: None, GetEventObject=lambda: button))
	assert not button.enabled
	framework.threads.pop()()
	framework.callbacks.pop()()
	assert button.enabled
	assert messages[-1].startswith("Could not fetch OpenAI models:")
	assert "API key is missing" in messages[-1]


def test_pending_model_fetch_does_not_enable_button_after_automatic_mode_is_off(framework):
	module = load_recognition_settings_module()
	panel = module.AutomaticRecognitionPanel()
	engine = object()
	panel._autoRecognitionEngine = engine
	panel._getSelectedAutoRecognitionType = lambda: "off"
	panel.autoRecognitionEngineChoices = []
	panel.autoRecognitionEngineList = SimpleNamespace(Enable=lambda enabled: None)
	button = panel.autoRecognitionFetchModelsButton = Button(panel)
	panel.updateDriverSettings = panel._updateAutoRecognitionControlState
	results = []
	framework.ui.runSettingsAction(
		button, engine, lambda: ["model"], results.append, results.append, lambda: True,
	)
	panel._autoRecognitionEngine = None
	panel._updateAutoRecognitionControlState()
	framework.threads.pop()()
	framework.callbacks.pop()()
	assert not button.enabled
	assert not results
