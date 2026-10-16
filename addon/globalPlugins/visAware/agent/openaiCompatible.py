# Copyright (C) 2026 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""OpenAI-compatible Chat Completions client for the computer-use agent."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from collections.abc import Callable
from typing import Any

import addonHandler
from logHandler import log

from .. import network
from ..exceptions import ApiError, AuthenticationError
from ..openaiCompatible import (
	OpenAICompatibleClient,
	OpenAICompatibleSettings,
	getApiErrorMessage,
	setCompletionTokenLimit,
	validateCompletion,
)
from ..requestLogging import isVerboseDebugLoggingEnabled, redactForLog
from .actions import JPEG_QUALITY, Screenshot, formatScreenshotPromptContext
from .decision import AGENT_ACTION_SCHEMA, AgentDecision, parseAgentDecision

addonHandler.initTranslation()

AGENT_DECISION_TOOL_NAME = "agent_decision"
MAX_HISTORY_ITEMS = 8
MAX_CONVERSATION_TURNS = 4
MAX_REQUEST_BYTES = 1_900_000
REQUEST_TIMEOUT = 180
MAX_COMPLETION_TOKENS = 4_096


@dataclass(frozen=True)
class OpenAICompatibleAgentSettings:
	apiKey: str = ""
	baseUrl: str = "https://api.openai.com/v1"
	model: str = ""
	imageQuality: int = JPEG_QUALITY
	source: str = ""


class OpenAICompatibleAgentClient:
	def __init__(self, settings: OpenAICompatibleAgentSettings) -> None:
		self.apiKey = settings.apiKey.strip()
		self.baseUrl = settings.baseUrl.rstrip("/") or "https://api.openai.com/v1"
		self.model = settings.model.strip()
		self.imageQuality = settings.imageQuality
		if not self.apiKey:
			# Translators: Reported when no API key is configured for the selected AI Agent.
			raise AuthenticationError(
				_("API key is missing. Please configure it in the AI Agent settings."),
			)
		if not self.model:
			# Translators: Reported when no model is configured for the selected AI Agent.
			raise ApiError(_("Model is missing. Please configure it in the AI Agent settings."))
		log.info(
			f"Vis Aware agent using OpenAI-compatible settings from "
			f"{settings.source or 'Agent settings'}, model={self.model}, "
			f"imageQuality={self.imageQuality}",
		)
		self._messages: list[dict[str, Any]] = []
		self._pendingToolUseId: str | None = None
		self._cancellationChecker: Callable[[], None] | None = None

	def setCancellationChecker(self, cancellationChecker: Callable[[], None]) -> None:
		self._cancellationChecker = cancellationChecker

	def nextAction(self, goal: str, screenshot: Screenshot, history: list[str]) -> AgentDecision:
		self._buildMessages(goal, screenshot, history)
		payload: dict[str, Any] = {
			"model": self.model,
			"messages": [
				{"role": "system", "content": _buildSystemInstruction()},
				*deepcopy(self._messages),
			],
			"tools": [_buildAgentDecisionTool()],
			"tool_choice": {
				"type": "function",
				"function": {"name": AGENT_DECISION_TOOL_NAME},
			},
		}
		setCompletionTokenLimit(payload, self.model, MAX_COMPLETION_TOKENS)
		client = OpenAICompatibleClient(
			OpenAICompatibleSettings(apiKey=self.apiKey, baseUrl=self.baseUrl),
			transport=network,
		)
		requestParams = client.requestParams(
			payload,
			timeout=REQUEST_TIMEOUT,
			cancelCheck=self._cancellationChecker,
		)
		if isVerboseDebugLoggingEnabled():
			log.debug(f"OpenAI-compatible agent request params: {redactForLog(requestParams)}")
		apiResult = client.complete(
			payload,
			timeout=REQUEST_TIMEOUT,
			cancelCheck=self._cancellationChecker,
		)
		if isVerboseDebugLoggingEnabled():
			log.debug(f"OpenAI-compatible agent response: {redactForLog(apiResult)}")
		return self._parseDecision(apiResult)

	def _buildMessages(self, goal: str, screenshot: Screenshot, history: list[str]) -> None:
		if self._pendingToolUseId:
			self._messages.append(
				{
					"role": "tool",
					"tool_call_id": self._pendingToolUseId,
					"content": _formatToolResult(history),
				},
			)
			self._messages.append(
				{"role": "user", "content": self._buildUserContent(goal, screenshot, history)},
			)
			self._pendingToolUseId = None
		else:
			self._messages.append(
				{"role": "user", "content": self._buildUserContent(goal, screenshot, history)},
			)
		self._trimMessages()

	@staticmethod
	def _buildUserContent(goal: str, screenshot: Screenshot, history: list[str]) -> list[dict[str, Any]]:
		return [
			{
				"type": "image_url",
				"image_url": {
					"url": f"data:{screenshot.mimeType};base64,{screenshot.imageBase64}",
				},
			},
			{
				"type": "text",
				"text": (
					f"Task: {goal}.\n"
					f"Foreground App: {screenshot.window.appName}. "
					f"{formatScreenshotPromptContext(screenshot)}\n"
					f"Recent actions and observations:\n{_formatPromptHistory(history)}"
				),
			},
		]

	def _trimMessages(self) -> None:
		if len(self._messages) <= 1:
			self._checkMessageSize()
			return
		turns: list[list[dict[str, Any]]] = []
		for message in self._messages:
			if message.get("role") == "user" or not turns:
				turns.append([])
			turns[-1].append(message)
		turns = turns[-MAX_CONVERSATION_TURNS:]
		self._messages = [message for turn in turns for message in turn]
		_dropOldScreenshots(self._messages)
		while len(turns) > 1 and _messageSize(self._messages) > MAX_REQUEST_BYTES:
			turns.pop(0)
			self._messages = [message for turn in turns for message in turn]
			_dropOldScreenshots(self._messages)
		self._checkMessageSize()

	def _checkMessageSize(self) -> None:
		if _messageSize(self._messages) > MAX_REQUEST_BYTES:
			raise ApiError(_("The current screenshot is too large for the OpenAI-compatible request limit."))

	def _parseDecision(self, apiResult: dict[str, Any]) -> AgentDecision:
		apiError = getApiErrorMessage(apiResult)
		if apiError:
			raise ApiError(_("OpenAI-compatible service API error: {}").format(apiError))
		try:
			_choice, message = validateCompletion(apiResult, {"tool_calls"})
		except ApiError as e:
			raise ApiError(_("OpenAI-compatible service returned malformed agent action data.")) from e
		toolCalls = message.get("tool_calls")
		if not isinstance(toolCalls, list) or len(toolCalls) != 1 or not isinstance(toolCalls[0], dict):
			raise ApiError(_("OpenAI-compatible service did not return exactly one agent tool call."))
		toolCall = toolCalls[0]
		if toolCall.get("type") != "function":
			raise ApiError(_("OpenAI-compatible service returned malformed agent action data."))
		function = toolCall.get("function")
		if not isinstance(function, dict) or function.get("name") != AGENT_DECISION_TOOL_NAME:
			raise ApiError(_("OpenAI-compatible service returned an unsupported agent tool call."))
		toolUseId = toolCall.get("id")
		if not isinstance(toolUseId, str) or not toolUseId:
			raise ApiError(_("OpenAI-compatible service returned malformed agent action data."))
		if any(
			isinstance(existing.get("tool_calls"), list)
			and any(call.get("id") == toolUseId for call in existing["tool_calls"] if isinstance(call, dict))
			for existing in self._messages
		):
			raise ApiError(_("OpenAI-compatible service returned a duplicate agent tool call ID."))
		arguments = function.get("arguments")
		if isinstance(arguments, str):
			try:
				arguments = json.loads(arguments)
			except json.JSONDecodeError as e:
				raise ApiError(_("OpenAI-compatible service returned invalid agent action JSON.")) from e
		if not isinstance(arguments, dict):
			raise ApiError(_("OpenAI-compatible service returned malformed agent action data."))
		decision = parseAgentDecision(arguments, "OpenAI-compatible service")
		assistantMessage = deepcopy(message)
		assistantMessage["role"] = "assistant"
		self._messages.append(assistantMessage)
		self._pendingToolUseId = toolUseId
		return decision

	def listModels(self) -> list[str]:
		return OpenAICompatibleClient(
			OpenAICompatibleSettings(apiKey=self.apiKey, baseUrl=self.baseUrl),
			transport=network,
		).listModels(timeout=30, cancelCheck=self._cancellationChecker)


def _buildSystemInstruction() -> str:
	return (
		"You are a Windows operator. Always call the agent_decision function exactly once; do not answer in text. "
		"Use the screenshot to choose one real UI action or finish only after visible confirmation. "
		"Use action, finish, or ask_user status as appropriate. Coordinates are integers normalized from 0 to 1000. "
		"Return JSON matching the tool schema, use null for empty optional fields, and keep every action limited to one step. "
		"Everything must be in the user's language."
	)


def _buildAgentDecisionTool() -> dict[str, Any]:
	schema = deepcopy(AGENT_ACTION_SCHEMA)
	schema.pop("propertyOrdering", None)
	return {
		"type": "function",
		"function": {
			"name": AGENT_DECISION_TOOL_NAME,
			"description": "Return the next Windows desktop action for the local agent host.",
			"parameters": schema,
		},
	}


def _formatPromptHistory(history: list[str]) -> str:
	if not history:
		return "None."
	return "\n".join(" ".join(str(item).split())[:1000] for item in history[-MAX_HISTORY_ITEMS:])


def _messageSize(messages: list[dict[str, Any]]) -> int:
	return len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _dropOldScreenshots(messages: list[dict[str, Any]]) -> None:
	imageMessageIndexes = [
		index
		for index, message in enumerate(messages)
		if isinstance(message.get("content"), list)
		and any(isinstance(part, dict) and part.get("type") == "image_url" for part in message["content"])
	]
	for index in imageMessageIndexes[:-1]:
		messages[index]["content"] = [
			part
			for part in messages[index]["content"]
			if not isinstance(part, dict) or part.get("type") != "image_url"
		]


def _formatToolResult(history: list[str]) -> str:
	result = "The local host handled the previous agent_decision tool call."
	if not history:
		return f"{result} Inspect the current screenshot for the latest state."
	return f"{result}\n{_formatPromptHistory(history[-2:])}"
