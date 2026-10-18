import json
import sys
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

from tests.test_deepseek_agent import _install_stubs, _Screenshot
from tests.test_kimi_image_describer import _load_module, load_kimi_image_module
from tests.test_vivo_auth import load_vivo_auth_module


@pytest.fixture
def vivo(monkeypatch):
	_install_stubs()
	imageBase = load_kimi_image_module()
	auth = load_vivo_auth_module()
	for method in ("warning", "debugWarning"):
		setattr(sys.modules["logHandler"].log, method, lambda *args, **kwargs: None)
	_load_module("addon.globalPlugins.visAware.exceptions", "addon/globalPlugins/visAware/exceptions.py")
	_load_module(
		"addon.globalPlugins.visAware.agent.decision", "addon/globalPlugins/visAware/agent/decision.py",
	)
	sys.modules["addon.globalPlugins.visAware.engineSettings"].NumericEngineSetting = sys.modules[
		"addon.globalPlugins.visAware.engineSettings"
	].ChoiceEngineSetting
	sys.modules["addon.globalPlugins.visAware.conversation"].QuestionStreamEvent = object
	protocol = _load_module(
		"addon.globalPlugins.visAware.vivoCompletions", "addon/globalPlugins/visAware/vivoCompletions.py",
	)
	image = _load_module(
		"addon.globalPlugins.visAware.imageDescribers.vivoBlueLLM",
		"addon/globalPlugins/visAware/imageDescribers/vivoBlueLLM.py",
	)
	agent = _load_module(
		"addon.globalPlugins.visAware.agent.vivo", "addon/globalPlugins/visAware/agent/vivo.py",
	)
	monkeypatch.setattr(auth, "getNvdacnCredentials", lambda: ("user", "password"))
	signed = []

	def sign(*args):
		signed.append(args)
		return {"X-AI-GATEWAY-SIGNATURE": "signature"}

	monkeypatch.setattr(auth, "genSignHeaders", sign)
	return SimpleNamespace(
		protocol=protocol, image=image, agent=agent, request=imageBase.RecognitionRequest, signed=signed,
	)


def assert_signed_request(vivo, params, stream):
	url = urlsplit(params["url"])
	assert url.netloc == "api-ai.vivo.com.cn"
	assert url.path == ("/vivogpt/completions/stream" if stream else "/vivogpt/completions")
	user, password, method, uri, query = vivo.signed[-1]
	assert (user, password, method, uri) == ("user", "password", "POST", url.path)
	assert parse_qs(url.query) == {key: [value] for key, value in query.items()}
	assert params["headers"] == {"X-AI-GATEWAY-SIGNATURE": "signature", "Content-Type": "application/json"}
	assert params["json"]["model"] == "BlueLM-Vision-Aid"
	assert params["json"]["provider"] == "vivo"
	assert params["json"]["sessionId"] != query["requestId"]


@pytest.mark.parametrize("stream", [False, True])
def test_image_request_and_follow_up_sign_the_actual_endpoint(vivo, stream):
	engine = vivo.image.CustomContentRecognizer()
	request = vivo.request(streamResult=stream)
	request.prompt = "Describe this screen"
	engine.thinkingLevel = "high"
	params = engine._buildRequestParams(b"encoded-image", request)
	assert_signed_request(vivo, params, stream)
	assert "timeout" not in params
	assert params["json"]["messages"] == [
		{"role": "user", "content": "data:image/JPEG;base64,encoded-image", "contentType": "image"},
		{"role": "user", "content": request.prompt, "contentType": "text"},
	]
	assert params["json"]["extra"] == {
		"enable_thinking": True,
		"temperature": engine.temperature / 100,
		"max_tokens": 1024,
	}
	context = SimpleNamespace(
		image=object(),
		initialText="Original answer",
		turns=[SimpleNamespace(role="assistant", text="Previous answer")],
	)
	messages = engine._buildQuestionMessages(context, "New question")
	params = engine._buildRequestParamsFromMessages(messages, request)
	assert_signed_request(vivo, params, stream)
	assert [item["role"] for item in params["json"]["messages"]] == [
		"user",
		"user",
		"assistant",
		"assistant",
		"user",
	]
	assert params["json"]["messages"][-1]["content"] == "New question"


def test_agent_request_keeps_tool_schema_timeout_and_decision(vivo, monkeypatch):
	requests = []
	arguments = {
		"status": "action",
		"message": "Click",
		"action": "click_at",
		"finished": False,
		"x": 500,
		"y": 400,
	}

	def send(**params):
		requests.append(params)
		return SimpleNamespace(
			json=lambda: {
				"code": 0,
				"data": {"toolCalls": [{"name": "agent_decision", "arguments": arguments}]},
			},
		)

	monkeypatch.setattr(vivo.agent.network, "sendRequest", send)
	client = vivo.agent.VivoAgentClient(vivo.agent.VivoAgentSettings(enableThinking=True))
	decision = client.nextAction("Open settings", _Screenshot(), ["Previous action"])
	params = requests[0]
	assert_signed_request(vivo, params, False)
	assert params["timeout"] == 75
	assert params["json"]["extra"]["max_tokens"] == 512
	assert params["json"]["extra"]["enable_thinking"]
	tool = params["json"]["extra"]["tools"][0]["function"]
	assert tool["name"] == "agent_decision"
	assert "propertyOrdering" not in tool["parameters"]
	assert decision.action.name == "click_at"
	assert decision.action.arguments == {"x": 500, "y": 400}


def test_stream_keeps_text_thinking_replacement_and_api_error(vivo):
	engine = vivo.image.CustomContentRecognizer()
	request = vivo.request()
	for chunk in (
		b": keepalive",
		b"data:",
		b"data: [DONE]",
		b"data: broken",
		b'data: {"type":"thinking","message":"private"}',
		b'data: {"type":"tool","message":"private"}',
	):
		assert engine.processStreamChunk(chunk, request) is None
	assert engine.processStreamChunk(b'data: {"type":"text","message":"answer"}', request) == "answer"
	with pytest.raises(vivo.protocol.StreamReplacementError) as replacement:
		engine.processStreamChunk(b'data: {"reply":"replacement"}', request)
	assert replacement.value.replacementText == "replacement"
	with pytest.raises(vivo.protocol.ApiError) as error:
		engine.processStreamChunk(b'data: {"code":401,"msg":"Denied"}', request)
	assert error.value.errorCode == 401
	assert "Denied" in str(error.value)


@pytest.mark.parametrize(
	"content,expected",
	[
		("plain answer", "plain answer"),
		('"JSON string"', "JSON string"),
		('[{"text":"list answer"}]', "list answer"),
		("[]", ""),
	],
)
def test_image_response_keeps_vivo_content_formats(vivo, content, expected):
	engine = vivo.image.CustomContentRecognizer()
	result = {"code": 0, "data": {"content": content}}
	assert engine.processApiResult(json.dumps(result).encode()) is False
	assert engine.extractText(result) == expected


@pytest.mark.parametrize("result", [{"code": 403, "msg": "Denied"}, {}])
def test_roles_reject_api_errors_with_the_same_message(vivo, result):
	message = vivo.image.CustomContentRecognizer().processApiResult(json.dumps(result).encode())
	with pytest.raises(vivo.agent.ApiError, match="Vivo API error") as error:
		vivo.agent._extractToolArguments(result)
	assert str(error.value) == message


def test_agent_repairs_known_tool_argument_formats_and_rejects_other_tools(vivo):
	result = {
		"code": 0,
		"data": {
			"functionCall": {"name": "agent_decision", "arguments": '{"finished": False, "x": None, "y": }'},
		},
	}
	assert vivo.agent._extractToolArguments(result) == {"finished": False, "x": None, "y": None}
	result["data"]["functionCall"]["name"] = "unknown"
	with pytest.raises(vivo.agent.ApiError, match="unsupported"):
		vivo.agent._extractToolArguments(result)
