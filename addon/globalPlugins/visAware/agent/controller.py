# Copyright (C) 2026 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Owns the foreground-window Agent UI and session lifecycle."""

from __future__ import annotations

from collections.abc import Callable
from threading import Event

import addonHandler
import gui
import ui
import wx
from logHandler import log

from .._screenCapture import isScreenCurtainCaptureSupported
from .._taskCues import TaskCueManager
from ..exceptions import ApiError, AuthenticationError, NetworkError
from .client import createAgentClient
from .session import AgentSession

addonHandler.initTranslation()


class AgentController:
	"""Coordinates Agent prompt dialogs, sessions, and question dialogs."""

	def __init__(self, taskCues: TaskCueManager, isScreenCurtainRunning: Callable[[], bool]) -> None:
		self._taskCues = taskCues
		self._isScreenCurtainRunning = isScreenCurtainRunning
		self._agentSession: AgentSession | None = None
		self._promptDialogActive = False
		self._questionDialog: wx.Dialog | None = None
		self._startTimer: wx.CallLater | None = None
		self._terminated = False

	def promptAndStart(self) -> None:
		if self.stop(isUserInitiated=True):
			return
		if self._terminated or self._promptDialogActive:
			return
		if self._isScreenCurtainRunning() and not isScreenCurtainCaptureSupported():
			# Translators: A message shown when trying to use the AI agent with screen curtain enabled.
			ui.message(_("Please disable screen curtain before starting the AI Agent."))
			return
		try:
			createAgentClient()
		except (AuthenticationError, NetworkError, ApiError) as e:
			ui.message(str(e))
			return
		self._promptDialogActive = True

		def showDialog() -> None:
			if self._terminated:
				self._promptDialogActive = False
				return
			goal = ""
			gui.mainFrame.prePopup()
			try:
				dialog = wx.TextEntryDialog(
					gui.mainFrame,
					# Translators: Prompt shown when starting the AI agent.
					_("What should the AI Agent do?"),
					# Translators: Title for the AI agent command dialog.
					_("Vis Aware Agent"),
				)
				try:
					if dialog.ShowModal() != wx.ID_OK:
						return
					goal = dialog.GetValue().strip()
				finally:
					dialog.Destroy()
			finally:
				gui.mainFrame.postPopup()
				self._promptDialogActive = False
			if goal and not self._terminated:
				self._startTimer = wx.CallLater(250, self.start, goal)

		wx.CallAfter(showDialog)

	def start(self, goal: str) -> None:
		self._startTimer = None
		if self._terminated or (self._agentSession and self._agentSession.isRunning):
			return
		log.info("Starting Vis Aware agent from user prompt.")
		self._agentSession = AgentSession(
			goal,
			self._taskCues,
			onDone=self._onAgentDone,
			askUser=self.askUser,
		)
		self._agentSession.start()

	def _onAgentDone(self, session: AgentSession) -> None:
		if self._agentSession is session:
			self._agentSession = None

	def stop(self, isUserInitiated: bool = True) -> bool:
		if self._startTimer:
			if self._startTimer.IsRunning():
				self._startTimer.Stop()
			self._startTimer = None
			if isUserInitiated:
				# Translators: Reported when a pending Agent start is cancelled by the user.
				ui.message(_("Agent start cancelled."))
			return True
		if self._agentSession and self._agentSession.isRunning:
			if self._agentSession.cancel():
				log.info("Vis Aware agent stop requested.")
			else:
				log.info("Vis Aware agent stop already pending.")
			self.closeQuestionDialog()
			if isUserInitiated:
				# Translators: Reported when the AI agent is being stopped by the user.
				ui.message(_("Agent is stopping."))
			return True
		self._agentSession = None
		return False

	def askUser(self, question: str, cancelEvent: Event) -> str | None:
		doneEvent = Event()
		result: dict[str, str | None] = {"answer": None}

		def showDialog() -> None:
			if cancelEvent.is_set() or self._terminated:
				doneEvent.set()
				return
			dialog = None
			gui.mainFrame.prePopup()
			try:
				dialog = wx.TextEntryDialog(
					gui.mainFrame,
					question,
					# Translators: Title for the AI agent command dialog.
					_("Vis Aware Agent"),
				)
				self._questionDialog = dialog
				if dialog.ShowModal() == wx.ID_OK and not cancelEvent.is_set():
					result["answer"] = dialog.GetValue().strip()
			finally:
				if self._questionDialog is dialog:
					self._questionDialog = None
				if dialog:
					dialog.Destroy()
				gui.mainFrame.postPopup()
				doneEvent.set()

		wx.CallAfter(showDialog)
		closeRequested = False
		while not doneEvent.wait(0.05):
			if cancelEvent.is_set() and not closeRequested:
				closeRequested = True
				wx.CallAfter(self.closeQuestionDialog)
		return result["answer"]

	def closeQuestionDialog(self) -> None:
		dialog = self._questionDialog
		if not dialog:
			return
		try:
			if dialog.IsModal():
				dialog.EndModal(wx.ID_CANCEL)
			else:
				dialog.Close()
		except Exception:
			log.debugWarning("Could not close Agent question dialog.", exc_info=True)

	def terminate(self) -> None:
		self._terminated = True
		self.stop(isUserInitiated=False)
		self.closeQuestionDialog()
		self._agentSession = None
