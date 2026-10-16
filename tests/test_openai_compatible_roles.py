from __future__ import annotations

import sys
import types

from tests.test_kimi_agent_interactions import _FakeResponse, _load_module, load_kimi_agent_module
from tests.test_kimi_image_describer import load_kimi_image_module
from tests.test_kimi_ocr import load_kimi_ocr_module


def _load_shared_engine_helpers() -> None:
	settingsModule = sys.modules["addon.globalPlugins.visAware.engineSettings"]
	if not hasattr(settingsModule, "BooleanEngineSetting"):
		settingsModule.BooleanEngineSetting = settingsModule.TextInputEngineSetting
	configModule = types.ModuleType("config")
	configModule.conf = {}
	sys.modules[configModule.__name__] = configModule
	_load_module(
		"addon.globalPlugins.visAware._modelSettings",
		"addon/globalPlugins/visAware/_modelSettings.py",
	)
	_load_module(
		"addon.globalPlugins.visAware._openaiEngine",
		"addon/globalPlugins/visAware/_openaiEngine.py",
	)


def test_agent_uses_non_streaming_function_tool_call(monkeypatch):
	load_kimi_agent_module()
	module = _load_module(
		"addon.globalPlugins.visAware.agent.openaiCompatible",
		"addon/globalPlugins/visAware/agent/openaiCompatible.py",
	)
	response = _FakeResponse(
		{
			"choices": [
				{
					"finish_reason": "tool_calls",
					"message": {
						"role": "assistant",
						"tool_calls": [
							{
								"id": "call-1",
								"type": "function",
								"function": {
									"name": "agent_decision",
									"arguments": '{"status":"finish","message":"done"}',
								},
							},
						],
					},
				}
			],
		}
	)
	calls = []
	monkeypatch.setattr(
		module.network,
		"sendRequest",
		lambda **kwargs: calls.append(kwargs) or response,
	)
	client = module.OpenAICompatibleAgentClient(
		module.OpenAICompatibleAgentSettings(
			apiKey="secret", baseUrl="https://example.test/v1", model="vision"
		)
	)
	decision = client.nextAction("finish the task", module.Screenshot(), [])

	assert decision == {"status": "finish", "message": "done"}
	assert len(calls) == 1
	payload = calls[0]["json"]
	assert payload.get("stream") is None
	assert payload["tool_choice"]["function"]["name"] == "agent_decision"
	assert payload["max_tokens"] == 4096


def test_ocr_uses_standard_responses_for_text_and_json_coordinates():
	load_kimi_ocr_module()
	_load_shared_engine_helpers()
	module = _load_module(
		"addon.globalPlugins.visAware.contentRecognizers.openaiCompatibleOCR",
		"addon/globalPlugins/visAware/contentRecognizers/openaiCompatibleOCR.py",
	)
	engine = module.CustomContentRecognizer()
	engine.apiKey = "secret"
	engine.model = "vision"

	textParams = engine._buildRequestParams(
		b"image",
		module.RecognitionRequest(textResult=True, streamResult=True),
	)
	structuredParams = engine._buildRequestParams(
		b"image",
		module.RecognitionRequest(textResult=False, streamResult=True),
	)

	assert textParams["json"].get("stream") is None
	assert "response_format" not in textParams["json"]
	assert structuredParams["json"].get("stream") is None
	assert structuredParams["json"]["response_format"] == {"type": "json_object"}
	assert "json" in structuredParams["json"]["messages"][1]["content"][1]["text"].lower()


def test_image_description_uses_sse_and_preserves_assistant_history():
	load_kimi_image_module()
	_load_shared_engine_helpers()
	module = _load_module(
		"addon.globalPlugins.visAware.imageDescribers.openaiCompatibleImage",
		"addon/globalPlugins/visAware/imageDescribers/openaiCompatibleImage.py",
	)
	engine = module.CustomContentRecognizer()
	engine.apiKey = "secret"
	engine.model = "vision"
	request = module.RecognitionRequest(textResult=True, streamResult=True)
	params = engine._buildRequestParams(b"encoded-image", request)

	assert params["json"]["stream"] is True
	assert "stream_options" not in params["json"]
	assert params["json"]["messages"][0]["content"][0]["type"] == "image_url"
	assert params["json"]["max_tokens"] == 4096
	assistant = engine._assistantMessage(
		{"choices": [{"message": {"role": "assistant", "content": "initial"}}]},
		"fallback",
	)
	assert assistant == {"role": "assistant", "content": "initial"}


def test_ocr_empty_prompt_restores_plain_text_mode():
	load_kimi_ocr_module()
	_load_shared_engine_helpers()
	module = _load_module(
		"addon.globalPlugins.visAware.contentRecognizers.openaiCompatibleOCR",
		"addon/globalPlugins/visAware/contentRecognizers/openaiCompatibleOCR.py",
	)
	engine = module.CustomContentRecognizer()

	engine.prompt = "custom"
	engine.prompt = "  "

	assert engine.prompt == module.PLAIN_OCR_PROMPT


def test_plain_ocr_preserves_json_like_visible_text():
	load_kimi_ocr_module()
	_load_shared_engine_helpers()
	module = _load_module(
		"addon.globalPlugins.visAware.contentRecognizers.openaiCompatibleOCR",
		"addon/globalPlugins/visAware/contentRecognizers/openaiCompatibleOCR.py",
	)
	engine = module.CustomContentRecognizer()
	response = {
		"choices": [
			{
				"finish_reason": "stop",
				"message": {"role": "assistant", "content": '{"lines":[]}'},
			}
		]
	}

	assert engine.extractText(response) == '{"lines":[]}'


def test_openai_models_use_completion_token_parameter():
	load_kimi_image_module()
	_load_shared_engine_helpers()
	module = _load_module(
		"addon.globalPlugins.visAware.imageDescribers.openaiCompatibleImage",
		"addon/globalPlugins/visAware/imageDescribers/openaiCompatibleImage.py",
	)
	engine = module.CustomContentRecognizer()
	engine.apiKey = "secret"
	engine.model = "gpt-5-mini"
	params = engine._buildRequestParams(b"encoded-image", module.RecognitionRequest(streamResult=False))

	assert params["json"]["max_completion_tokens"] == 4096
	assert "max_tokens" not in params["json"]
