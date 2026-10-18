# Copyright (C) 2026 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Vivo BlueLM Vision client for the Vis Aware computer-use agent."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import re
from typing import Any

import addonHandler
from logHandler import log

from .. import network
from .. import vivoAuth, vivoCompletions
from ..exceptions import ApiError
from .actions import JPEG_QUALITY, Screenshot, formatScreenshotPromptContext
from .decision import AGENT_ACTION_SCHEMA, AgentDecision, parseAgentDecision

addonHandler.initTranslation()

MAX_HISTORY_ITEMS = 4
MAX_HISTORY_CHARS = 700


@dataclass(frozen=True)
class VivoAgentSettings:
	imageQuality: int = JPEG_QUALITY
	enableThinking: bool = False
	source: str = ""


class VivoAgentClient:
	def __init__(self, settings: VivoAgentSettings) -> None:
		self.imageQuality = settings.imageQuality
		self.enableThinking = settings.enableThinking
		self.nvdacnUser, self.nvdacnPass = vivoAuth.getNvdacnCredentials()
		log.info(
			f"Vis Aware agent using Vivo settings from {settings.source or 'Agent settings'}, "
			f"model={vivoCompletions.MODEL}, imageQuality={self.imageQuality}, "
			f"enableThinking={self.enableThinking}",
		)

	def nextAction(self, goal: str, screenshot: Screenshot, history: list[str]) -> AgentDecision:
		prompt = self._buildPrompt(goal, screenshot, history)
		extra = {
			"enable_thinking": self.enableThinking,
			"temperature": 0,
			"max_tokens": 512,
			"tools": [_buildAgentDecisionTool()],
		}
		messages = [
			{
				"role": "user",
				"content": f"data:image/jpeg;base64,{screenshot.imageBase64}",
				"contentType": "image",
			},
			{"role": "user", "content": prompt, "contentType": "text"},
		]
		requestParams = vivoCompletions.buildCompletionsRequest(
			self.nvdacnUser,
			self.nvdacnPass,
			messages,
			extra,
		)
		response = network.sendRequest(**requestParams, timeout=75)
		try:
			apiResult = response.json()
		except ValueError as e:
			raise ApiError(_("Invalid response from server.")) from e
		return parseAgentDecision(_extractToolArguments(apiResult), "Vivo")

	def _buildPrompt(self, goal: str, screenshot: Screenshot, history: list[str]) -> str:
		historyText = _formatPromptHistory(history)
		return (
			f"You are a Windows operator. Foreground App: {screenshot.window.appName}. "
			f"{formatScreenshotPromptContext(screenshot)} "
			f"Task: {goal}. STRICT RULES:\n"
			"1. RESPONSE LANGUAGE: Everything MUST be in the user's language.\n"
			"2. OUTPUT: Always call the agent_decision tool exactly once. Do not answer in text.\n"
			'3. STATUS: Use "action" to operate, "finish" only after the visible state confirms completion, '
			'and "ask_user" only when user input is required.\n'
			'4. FINISHED: For an action that is probably final, set "finished": true; '
			"the host will still verify on the next screenshot before ending.\n"
			"- To press Enter after typing, set \"press_enter\": true or append '\\n' to the end of "
			'the "text" parameter.\n'
			"- Use double_click only when the visible target normally requires double activation; "
			"use click for ordinary buttons, menu items, tabs, and links.\n"
			"- Coordinates are normalized from 0 to 1000 relative to the screenshot, not pixels. "
			"When operating inside the foreground app, keep x/y inside the foreground window bounds. "
			"Each coordinate field must be one integer, not an array. "
			"Use JSON null for empty optional fields, never None. "
			"For drag_by, delta_x and delta_y are screen pixels. "
			"For scroll_at, put x and y inside the scrollable area. "
			"- If history says the previous action caused no visible screen change, do not repeat the same "
			"coordinates.\n"
			"Ignore 'Vis Aware Agent' or 'NVDA' windows.\n"
			f"- Recent actions:\n{historyText}"
		)


def _formatPromptHistory(history: list[str]) -> str:
	if not history:
		return "None."
	lines = []
	for item in history[-MAX_HISTORY_ITEMS:]:
		text = " ".join(str(item).split())
		if len(text) > MAX_HISTORY_CHARS:
			text = f"{text[:MAX_HISTORY_CHARS]}..."
		lines.append(text)
	return "\n".join(lines)


def _buildAgentDecisionTool() -> dict[str, Any]:
	schema = copy.deepcopy(AGENT_ACTION_SCHEMA)
	schema.pop("propertyOrdering", None)
	return {
		"type": "function",
		"function": {
			"name": "agent_decision",
			"description": "Return the next Windows desktop action for the agent.",
			"parameters": schema,
		},
	}


def _extractToolArguments(apiResult: dict[str, Any]) -> dict[str, Any]:
	errorMessage = vivoCompletions.getApiErrorMessage(apiResult)
	if errorMessage:
		raise ApiError(errorMessage)
	data = apiResult.get("data")
	if not isinstance(data, dict):
		raise ApiError(_("Vivo returned malformed agent action data."))
	toolCall = _getFirstToolCall(data)
	if not toolCall:
		raise ApiError(_("Vivo did not return an agent tool call."))
	if str(toolCall.get("name") or "") != "agent_decision":
		raise ApiError(_("Vivo returned an unsupported agent tool call."))
	arguments = toolCall.get("arguments")
	if isinstance(arguments, dict):
		return arguments
	if not isinstance(arguments, str) or not arguments.strip():
		raise ApiError(_("Vivo returned malformed agent action data."))
	try:
		parsed = json.loads(arguments)
	except json.JSONDecodeError:
		try:
			parsed = json.loads(_repairVivoToolArgumentsJson(arguments))
		except json.JSONDecodeError as e:
			log.debugWarning(f"Invalid Vivo agent tool arguments: {arguments!r}")
			raise ApiError(_("Vivo returned invalid agent action JSON.")) from e
	if not isinstance(parsed, dict):
		raise ApiError(_("Vivo returned malformed agent action data."))
	return parsed


def _repairVivoToolArgumentsJson(text: str) -> str:
	# ponytail: Vivo sometimes emits Python literals or empty values inside the JSON arguments string.
	text = text.replace(": None", ": null").replace(": True", ": true").replace(": False", ": false")
	return re.sub(r":\s*(?=[,}])", ": null", text)


def _getFirstToolCall(data: dict[str, Any]) -> dict[str, Any] | None:
	toolCalls = data.get("toolCalls")
	if isinstance(toolCalls, list) and toolCalls and isinstance(toolCalls[0], dict):
		return toolCalls[0]
	toolCall = data.get("toolCall")
	if isinstance(toolCall, dict):
		return toolCall
	functionCall = data.get("functionCall")
	if isinstance(functionCall, dict):
		return functionCall
	return None
