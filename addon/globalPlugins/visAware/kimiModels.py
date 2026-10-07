# Copyright (C) 2026 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Shared Kimi model presets and Chat Completions request helpers."""

from collections import OrderedDict
import time
from typing import Any
from urllib.parse import urlsplit

import addonHandler

from . import network
from .exceptions import ApiError, AuthenticationError

addonHandler.initTranslation()

# Kimi Code exposes this endpoint and the same OpenAI-compatible protocol as
# the public Kimi API. Public platform endpoints use api.moonshot.cn or
# api.moonshot.ai, depending on the platform where the key was created.
DEFAULT_KIMI_BASE_URL = "https://api.kimi.com/coding/v1"
DEFAULT_KIMI_MODEL = "kimi-for-coding"
DEFAULT_KIMI_AGENT_MODEL = "kimi-for-coding"
DEFAULT_KIMI_PUBLIC_MODEL = "kimi-k3"
KIMI_AUTO_SERVICE = "auto"
KIMI_CUSTOM_SERVICE = "custom"
KIMI_AUTO_MODEL = "__auto__"

KIMI_SERVICE_ENDPOINTS = OrderedDict(
	{
		"kimiCodeCn": (_("Kimi Code (China mainland)"), "https://api.kimi.com/coding/v1"),
		"kimiCodeGlobal": (_("Kimi Code (overseas)"), "https://api.kimi.ai/coding/v1"),
		"platformCn": (_("Kimi Open Platform (China mainland)"), "https://api.moonshot.cn/v1"),
		"platformGlobal": (_("Kimi Open Platform (overseas)"), "https://api.moonshot.ai/v1"),
	}
)

_KIMI_CONNECTION_CACHE: dict[tuple[str, str, str], tuple[str, list[str]]] = {}

KIMI_REASONING_EFFORTS = OrderedDict(
	{
		# Translators: A choice that disables Kimi model reasoning.
		"none": _("Disabled"),
		# Translators: A low Kimi model reasoning effort.
		"low": _("Low"),
		# Translators: A high Kimi model reasoning effort.
		"high": _("High"),
		# Translators: The maximum Kimi model reasoning effort.
		"max": _("Maximum"),
	},
)
KIMI_K3_REASONING_EFFORTS = OrderedDict(
	{
		# Translators: A low Kimi K3 reasoning effort.
		"low": _("Low"),
		# Translators: A high Kimi K3 reasoning effort.
		"high": _("High"),
		# Translators: The maximum Kimi K3 reasoning effort.
		"max": _("Maximum"),
	},
)
KIMI_K26_REASONING_EFFORTS = OrderedDict(
	{
		# Translators: A choice that disables reasoning for a Kimi K2.x model.
		"none": _("Disabled"),
		# Translators: A choice that enables reasoning for a Kimi K2.x model.
		"high": _("Enabled"),
	},
)

_K3_MODELS = {"k3", "k3-256k", "kimi-k3"}
_MILLION_TOKEN_CONTEXT_MODELS = {"k3", "kimi-k3"}
_K27_MODELS = {
	"kimi-for-coding",
	"kimi-for-coding-highspeed",
	"kimi-k2.7-code",
	"kimi-k2.7-code-highspeed",
}
_K26_MODELS = {"kimi-k2.6"}


def _normalizedModel(model: str | None) -> str:
	return (model or "").strip().lower()


def isKimiK3Model(model: str | None) -> bool:
	"""Returns whether a model uses Kimi K3's reasoning-effort API."""
	return _normalizedModel(model) in _K3_MODELS


def isKimiK27Model(model: str | None) -> bool:
	"""Returns whether a model is Kimi K2.7 Code."""
	return _normalizedModel(model) in _K27_MODELS


def isKimiK26Model(model: str | None) -> bool:
	"""Returns whether a model uses Kimi K2.6-style thinking controls."""
	return _normalizedModel(model) in _K26_MODELS


def getKimiContextWindow(model: str | None) -> int:
	"""Returns the model context window, defaulting custom models conservatively to 256K."""
	return 1_048_576 if _normalizedModel(model) in _MILLION_TOKEN_CONTEXT_MODELS else 262_144


def requiresPreservedThinking(model: str | None, reasoningEffort: str | None) -> bool:
	"""Returns whether assistant reasoning must be retained for later turns."""
	if isKimiK3Model(model) or isKimiK27Model(model):
		return True
	return isKimiK26Model(model) and normalizeKimiReasoningEffort(reasoningEffort) != "none"


def requiresKimiReasoningContent(model: str | None, reasoningEffort: str | None) -> bool:
	"""Returns whether a response must contain reasoning content for continuation."""
	return isKimiK27Model(model) or (
		isKimiK26Model(model) and normalizeKimiReasoningEffort(reasoningEffort) != "none"
	)


def normalizeKimiReasoningEffort(value: str | None, default: str = "high") -> str:
	"""Normalizes a configured effort value without guessing unknown values."""
	value = (value or "").strip().lower()
	return value if value in KIMI_REASONING_EFFORTS else default


def normalizeKimiReasoningEffortForModel(model: str | None, value: str | None) -> str:
	"""Normalizes reasoning effort to a value supported by the selected model."""
	effort = normalizeKimiReasoningEffort(value)
	if isKimiK3Model(model):
		return effort if effort in {"low", "high", "max"} else "high"
	if isKimiK27Model(model):
		return "high"
	if isKimiK26Model(model):
		return effort if effort in {"none", "high"} else "high"
	return effort


def getKimiReasoningEffortChoices(model: str | None) -> OrderedDict[str, str]:
	"""Returns the controls valid for the selected model family."""
	if isKimiK3Model(model):
		return KIMI_K3_REASONING_EFFORTS
	if isKimiK26Model(model):
		return KIMI_K26_REASONING_EFFORTS
	return OrderedDict()


def applyKimiThinking(payload: dict[str, Any], model: str, reasoningEffort: str | None) -> None:
	"""Adds only the thinking fields supported by the selected Kimi model.

	K3 uses top-level ``reasoning_effort``. K2.7 always thinks and therefore
	omits both controls. K2.6 uses ``thinking`` and can preserve reasoning for
	multi-turn calls.
	"""
	effort = normalizeKimiReasoningEffortForModel(model, reasoningEffort)
	payload.pop("reasoning_effort", None)
	payload.pop("thinking", None)
	if isKimiK3Model(model):
		payload["reasoning_effort"] = effort
	elif isKimiK26Model(model):
		payload["thinking"] = (
			{"type": "disabled"}
			if effort == "none"
			else {
				"type": "enabled",
				"keep": "all",
			}
		)


def buildKimiChatCompletionsUrl(baseUrl: str) -> str:
	"""Builds the OpenAI-compatible Chat Completions endpoint URL."""
	baseUrl = baseUrl.rstrip("/")
	if baseUrl.endswith("/chat/completions"):
		return baseUrl
	if baseUrl.endswith("/v1"):
		return f"{baseUrl}/chat/completions"
	return f"{baseUrl}/v1/chat/completions"


def buildKimiModelsUrl(baseUrl: str) -> str:
	"""Builds the model-list endpoint URL."""
	baseUrl = baseUrl.rstrip("/")
	if baseUrl.endswith("/models"):
		return baseUrl
	if baseUrl.endswith("/chat/completions"):
		baseUrl = baseUrl[: -len("/chat/completions")]
	if baseUrl.endswith("/v1"):
		return f"{baseUrl}/models"
	return f"{baseUrl}/v1/models"


_KIMI_CODE_MODEL_CHOICES = OrderedDict(
	{
		# Translators: The display name for a Kimi model preset.
		"k3-256k": _("Kimi K3 (256K context)"),
		# Translators: The display name for a Kimi model preset.
		"k3": _("Kimi K3 (1M context)"),
		# Translators: The display name for a Kimi model preset.
		"kimi-for-coding": _("Kimi K2.7 Code"),
		# Translators: The display name for a Kimi model preset.
		"kimi-for-coding-highspeed": _("Kimi K2.7 Code (high speed)"),
	},
)
_KIMI_PUBLIC_MODEL_CHOICES = OrderedDict(
	{
		# Translators: The display name for a Kimi model preset.
		"kimi-k3": _("Kimi K3 (public API)"),
		# Translators: The display name for a Kimi model preset.
		"kimi-k2.7-code": _("Kimi K2.7 Code (public API)"),
		# Translators: The display name for a Kimi model preset.
		"kimi-k2.7-code-highspeed": _("Kimi K2.7 Code (public API high speed)"),
		# Translators: The display name for a Kimi model preset.
		"kimi-k2.6": _("Kimi K2.6 (public API)"),
	},
)


def _kimiApiHost(baseUrl: str) -> str:
	return (urlsplit(baseUrl).hostname or "").lower()


def isOfficialKimiBaseUrl(baseUrl: str) -> bool:
	"""Returns whether a URL targets an official Kimi API host."""
	return _kimiApiHost(baseUrl) in {"api.kimi.com", "api.kimi.ai", "api.moonshot.ai", "api.moonshot.cn"}


def isKimiOfficialServiceUrl(baseUrl: str) -> bool:
	"""Returns whether a URL is one of Kimi's documented service roots."""
	baseUrl = baseUrl.rstrip("/")
	for suffix in ("/chat/completions", "/models"):
		if baseUrl.endswith(suffix):
			baseUrl = baseUrl[: -len(suffix)]
	return baseUrl in {url.rstrip("/") for _label, url in KIMI_SERVICE_ENDPOINTS.values()}


def getKimiModelChoices(baseUrl: str = DEFAULT_KIMI_BASE_URL) -> OrderedDict[str, str]:
	"""Returns model presets supported by the configured official endpoint."""
	host = _kimiApiHost(baseUrl)
	if host in {"api.kimi.com", "api.kimi.ai"}:
		return _KIMI_CODE_MODEL_CHOICES.copy()
	if host in {"api.moonshot.ai", "api.moonshot.cn"}:
		return _KIMI_PUBLIC_MODEL_CHOICES.copy()
	return _KIMI_CODE_MODEL_CHOICES | _KIMI_PUBLIC_MODEL_CHOICES


def getDefaultKimiModel(baseUrl: str) -> str:
	"""Returns a valid default model for an official Kimi endpoint."""
	if _kimiApiHost(baseUrl) in {"api.moonshot.ai", "api.moonshot.cn"}:
		return DEFAULT_KIMI_PUBLIC_MODEL
	return DEFAULT_KIMI_MODEL


def getPreferredKimiModel(baseUrl: str, models: list[str]) -> str:
	"""Selects a known vision model or the first model from a custom service."""
	available = set(models)
	defaultModel = getDefaultKimiModel(baseUrl)
	if defaultModel in available:
		return defaultModel
	for model in getKimiModelChoices(baseUrl):
		if model in available:
			return model
	if models and not isKimiOfficialServiceUrl(baseUrl):
		return models[0]
	if models:
		raise ApiError(_("No known Kimi vision model was found. Select a model manually."))
	return getDefaultKimiModel(baseUrl)


def filterKimiModels(baseUrl: str, models: list[str]) -> list[str]:
	"""Keeps selectable models that can accept image input."""
	uniqueModels = list(dict.fromkeys(model.strip() for model in models if model.strip()))
	if not isKimiOfficialServiceUrl(baseUrl):
		return uniqueModels
	supportedModels = set(getKimiModelChoices(baseUrl))
	return [model for model in uniqueModels if model in supportedModels]


def getKimiServiceChoices() -> OrderedDict[str, str]:
	"""Returns the user-facing service choices."""
	return OrderedDict(
		{
			KIMI_AUTO_SERVICE: _("Auto-detect (on first use)"),
			**{key: value[0] for key, value in KIMI_SERVICE_ENDPOINTS.items()},
			KIMI_CUSTOM_SERVICE: _("Custom service URL"),
		}
	)


def getKimiServiceBaseUrl(service: str, customBaseUrl: str = "") -> str:
	"""Returns the configured URL for a service choice."""
	if service == KIMI_CUSTOM_SERVICE:
		return customBaseUrl.strip().rstrip("/")
	return KIMI_SERVICE_ENDPOINTS.get(service, ("", DEFAULT_KIMI_BASE_URL))[1]


def resolveKimiConnection(
	apiKey: str,
	service: str,
	customBaseUrl: str = "",
	model: str | None = None,
	forceDiscovery: bool = False,
	cancellationChecker: Any | None = None,
	discoveryTimeout: float | None = None,
) -> tuple[str, list[str]]:
	"""Authenticates against one service and returns its URL and model IDs."""
	if not apiKey.strip():
		raise ValueError(_("API key is missing."))
	officialServiceUrls = {url.rstrip("/") for _label, url in KIMI_SERVICE_ENDPOINTS.values()}
	if service == KIMI_AUTO_SERVICE:
		legacyBaseUrl = customBaseUrl.strip().rstrip("/")
		if legacyBaseUrl and legacyBaseUrl not in officialServiceUrls:
			candidates = [legacyBaseUrl]
		else:
			candidates = [url for _label, url in KIMI_SERVICE_ENDPOINTS.values()]
	else:
		candidates = [getKimiServiceBaseUrl(service, customBaseUrl)]
	allowUndiscoveredModels = service == KIMI_CUSTOM_SERVICE or (
		service == KIMI_AUTO_SERVICE
		and bool(customBaseUrl.strip())
		and customBaseUrl.strip().rstrip("/") not in officialServiceUrls
	)
	if allowUndiscoveredModels and model and model != KIMI_AUTO_MODEL and not forceDiscovery:
		baseUrl = candidates[0] if candidates else ""
		if baseUrl:
			return baseUrl.rstrip("/"), []
	lastError: Exception | None = None
	deadline = time.monotonic() + discoveryTimeout if discoveryTimeout is not None else None
	for baseUrl in candidates:
		if not baseUrl:
			continue
		if callable(cancellationChecker):
			cancellationChecker()
		if deadline is not None:
			remaining = deadline - time.monotonic()
			if remaining <= 0:
				break
			requestTimeout = min(10, remaining)
		else:
			requestTimeout = 10
		try:
			response = network.sendRequest(
				"GET",
				buildKimiModelsUrl(baseUrl),
				headers={"Authorization": f"Bearer {apiKey.strip()}"},
				timeout=requestTimeout,
				retry=False,
				cancelCheck=cancellationChecker,
			)
			if callable(cancellationChecker):
				cancellationChecker()
			payload = response.json()
			models = payload.get("data", []) if isinstance(payload, dict) else []
			modelNames = [
				str(item["id"]).strip()
				for item in models
				if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"].strip()
			]
			visionModels = filterKimiModels(baseUrl, modelNames)
			if visionModels:
				return baseUrl.rstrip("/"), visionModels
			lastError = ValueError(_("The Kimi service returned no models."))
		except Exception as e:
			if type(e).__name__ == "CancellationError":
				raise
			if allowUndiscoveredModels and type(e).__name__ in {"AuthenticationError", "NetworkError"}:
				raise
			if allowUndiscoveredModels and isinstance(e, ApiError):
				if getattr(e, "errorCode", None) not in (404, 405, 501):
					raise
			lastError = e
	if allowUndiscoveredModels and candidates and candidates[0]:
		if not model or model == KIMI_AUTO_MODEL:
			raise ApiError(
				_("Could not detect models from this service. Enter a model name manually."),
			)
		return candidates[0].rstrip("/"), []
	if lastError:
		if isinstance(lastError, ValueError):
			raise ApiError(str(lastError)) from lastError
		raise lastError
	raise ValueError(_("No Kimi service URL is configured."))


def invalidateKimiConnection(engine: Any) -> None:
	if engine._connectionKey:
		_KIMI_CONNECTION_CACHE.pop(engine._connectionKey, None)
	engine._resolvedBaseUrl = None
	engine._resolvedModels = []
	engine._connectionKey = None
	if hasattr(engine, "_knownModels"):
		engine._knownModels = []
	if getattr(engine, "_autoModelSelected", False):
		engine._model = KIMI_AUTO_MODEL


def ensureKimiConnection(
	engine: Any,
	*,
	forceDiscovery: bool = False,
	cancellationChecker: Any | None = None,
	discoveryTimeout: float | None = None,
	missingKeyMessage: str,
) -> None:
	if not engine.apiKey:
		raise AuthenticationError(missingKeyMessage)
	key = (engine.apiKey, engine.service, engine._baseUrl)
	cachedConnection = _KIMI_CONNECTION_CACHE.get(key)
	if not forceDiscovery and cachedConnection:
		baseUrl, models = cachedConnection
		engine._resolvedBaseUrl, engine._resolvedModels, engine._connectionKey = baseUrl, list(models), key
		setKnownModels = getattr(engine, "_setKnownModels", None)
		if callable(setKnownModels):
			setKnownModels(models)
		if engine.model == KIMI_AUTO_MODEL:
			engine._model = getPreferredKimiModel(baseUrl, models)
			engine._normalizeReasoningEffort()
		return
	if not forceDiscovery and engine._connectionKey == key and engine._resolvedBaseUrl:
		if engine.model == KIMI_AUTO_MODEL:
			engine._model = getPreferredKimiModel(engine._resolvedBaseUrl, engine._resolvedModels)
			engine._normalizeReasoningEffort()
		return
	baseUrl, models = resolveKimiConnection(
		engine.apiKey,
		engine.service,
		engine._baseUrl,
		engine.model,
		forceDiscovery=forceDiscovery,
		cancellationChecker=cancellationChecker,
		discoveryTimeout=discoveryTimeout,
	)
	if models:
		_KIMI_CONNECTION_CACHE[key] = (baseUrl, list(models))
	else:
		_KIMI_CONNECTION_CACHE.pop(key, None)
	engine._resolvedBaseUrl, engine._resolvedModels, engine._connectionKey = baseUrl, models, key
	setKnownModels = getattr(engine, "_setKnownModels", None)
	if callable(setKnownModels):
		setKnownModels(models)
	if engine.model == KIMI_AUTO_MODEL:
		engine._model = getPreferredKimiModel(baseUrl, models)
		engine._normalizeReasoningEffort()


def startKimiModelDetection(engine: Any, evt: Any) -> None:
	import ui
	from threading import Thread

	evt.Skip()
	if getattr(engine, "_kimiModelDetectionInProgress", False):
		ui.message(_("Kimi service and model detection is already in progress."))
		return
	engine._kimiModelDetectionInProgress = True
	button = evt.GetEventObject()
	ui.message(_("Detecting Kimi service and models"))
	Thread(target=_detectKimiModels, args=(engine, button), daemon=True).start()


def _detectKimiModels(engine: Any, button: Any) -> None:
	import wx

	try:
		engine._ensureConnection(forceDiscovery=True)
		message = _("Kimi models loaded: {count}").format(count=len(engine._resolvedModels))
	except Exception as e:
		message = _("Could not detect Kimi service: {message}").format(message=str(e))
	wx.CallAfter(_finishKimiModelDetection, engine, button, message)


def _finishKimiModelDetection(engine: Any, button: Any, message: str) -> None:
	import ui

	engine._kimiModelDetectionInProgress = False
	ui.message(message)
	parent = button.GetParent()
	if hasattr(parent, "updateDriverSettings"):
		parent.updateDriverSettings()
