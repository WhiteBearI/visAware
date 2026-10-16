# Copyright (C) 2026 Cary-rowen <manchen_0528@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

from collections.abc import Callable
from threading import Event
from typing import Any

import addonHandler
import gui
from logHandler import log
import wx

from ..engineSettings import ButtonEngineSetting
from ..exceptions import ApiError, AuthenticationError
from ..recogHandler import BaseDescriber, RecognitionRequest
from ._beMyEyesClient import BeMyEyesClient, getStoredAccountLabel, getStoredToken, saveAccount
from ._beMyEyesGui import BeMyEyesLoginDialog

addonHandler.initTranslation()


class CustomContentRecognizer(BaseDescriber):
	name = "beMyEyes"
	# Translators: The Be My AI image description engine.
	description = _("Be My AI")
	uploadBase64EncodeImage = False
	supportsQuestions = True
	engineConfigSpec = {
		"encryptedToken": "string(default='')",
		"accountLabel": "string(default='')",
	}

	@property
	def supportedSettings(self) -> list[Any]:
		account = getStoredAccountLabel().replace("&", "&&")
		if getStoredToken():
			if account:
				# Translators: Account management button; {account} is the saved name and email.
				label = _("Log out of Be My Eyes ({account})").format(account=account)
			else:
				# Translators: Button for logging out of Be My Eyes.
				label = _("Log out of Be My Eyes")
		elif account:
			# Translators: Account management button when the saved login expired.
			label = _("Log in to Be My Eyes again ({account})...").format(account=account)
		else:
			# Translators: Button for opening the Be My Eyes login dialog.
			label = _("Log in to Be My Eyes...")
		return [ButtonEngineSetting("manageAccount", label)]

	def manageAccountChanger(self, evt: wx.CommandEvent) -> None:
		parent = evt.GetEventObject().GetParent()
		if getStoredToken():
			if (
				gui.messageBox(
					# Translators: Confirmation before deleting the locally saved Be My Eyes login.
					_("Are you sure you want to log out of Be My Eyes?"),
					# Translators: Title of the Be My Eyes logout confirmation.
					_("Log out of Be My Eyes"),
					wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION,
					parent,
				)
				!= wx.YES
			):
				return
			try:
				saveAccount("", "")
			except AuthenticationError as e:
				log.warning("Could not save Be My Eyes logout.", exc_info=True)
				gui.messageBox(str(e), self.description, wx.OK | wx.ICON_ERROR, parent)
		else:
			with BeMyEyesLoginDialog(parent) as dialog:
				dialog.ShowModal()
		parent.updateDriverSettings()

	@classmethod
	def check(cls) -> bool:
		return True

	def _buildRequestParams(self, imageContent: bytes, request: RecognitionRequest) -> dict[str, Any]:
		return {"imageContent": imageContent}

	def _getApiResult(self, requestParams: dict[str, Any], cancellationEvent: Event) -> dict[str, Any]:
		client = BeMyEyesClient(cancellationChecker=lambda: self._checkCancelled(cancellationEvent))
		return client.describeImage(requestParams["imageContent"])

	def askQuestion(
		self,
		context: Any,
		question: str,
		cancellationChecker: Callable[[], None] | None = None,
	) -> str:
		chatId = context.response.get("chatId")
		if type(chatId) is not int or chatId <= 0:
			# Translators: An error when a saved Be My Eyes result has no conversation to continue.
			raise ApiError(_("The previous Be My Eyes result cannot be used for follow-up questions."))
		client = BeMyEyesClient(cancellationChecker=cancellationChecker)
		return self._validateQuestionAnswer(client.askQuestion(chatId, question))

	def extractText(self, apiResult: dict[str, Any]) -> str:
		return apiResult["text"]
