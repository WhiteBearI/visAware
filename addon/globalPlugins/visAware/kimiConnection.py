# Copyright (C) 2026 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Kimi connection discovery and model cache."""

import time
from typing import Any

import addonHandler

from . import network
from .exceptions import ApiError, AuthenticationError, CancellationError
from .openaiCompatible import OpenAICompatibleClient, OpenAICompatibleSettings
from .kimiProvider import (
	KIMI_AUTO_MODEL,
	KIMI_AUTO_SERVICE,
	KIMI_CUSTOM_SERVICE,
	KIMI_SERVICE_ENDPOINTS,
	filterKimiModels,
	getKimiServiceBaseUrl,
	getPreferredKimiModel,
)

addonHandler.initTranslation()

_KIMI_CONNECTION_CACHE: dict[tuple[str, str, str], tuple[str, list[str]]] = {}


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
			client = OpenAICompatibleClient(
				OpenAICompatibleSettings(apiKey=apiKey, baseUrl=baseUrl),
				transport=network,
			)
			modelNames = client.listModels(
				timeout=requestTimeout,
				retry=False,
				cancelCheck=cancellationChecker,
			)
			if callable(cancellationChecker):
				cancellationChecker()
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
		applyKimiConnection(engine, key, baseUrl, models)
		return
	if not forceDiscovery and engine._connectionKey == key and engine._resolvedBaseUrl:
		if engine.model == KIMI_AUTO_MODEL:
			engine._model = getPreferredKimiModel(engine._resolvedBaseUrl, engine._resolvedModels)
			engine._normalizeReasoningEffort()
		return
	baseUrl, models = resolveKimiConnection(
		*key,
		engine.model,
		forceDiscovery=forceDiscovery,
		cancellationChecker=cancellationChecker,
		discoveryTimeout=discoveryTimeout,
	)
	if key != (engine.apiKey, engine.service, engine._baseUrl):
		raise CancellationError("Kimi connection settings changed during discovery.")
	applyKimiConnection(engine, key, baseUrl, models)


def applyKimiConnection(
	engine: Any,
	key: tuple[str, str, str],
	baseUrl: str,
	models: list[str],
) -> None:
	if models:
		_KIMI_CONNECTION_CACHE[key] = (baseUrl, list(models))
	else:
		_KIMI_CONNECTION_CACHE.pop(key, None)
	engine._resolvedBaseUrl, engine._resolvedModels, engine._connectionKey = baseUrl, list(models), key
	setKnownModels = getattr(engine, "_setKnownModels", None)
	if callable(setKnownModels):
		setKnownModels(models)
	if engine.model == KIMI_AUTO_MODEL:
		engine._model = getPreferredKimiModel(baseUrl, models)
		engine._normalizeReasoningEffort()
