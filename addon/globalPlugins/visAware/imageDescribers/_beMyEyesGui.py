# Copyright (C) 2026 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

from email.utils import parseaddr
from threading import Event, Thread

import addonHandler
import gui
from gui import guiHelper
from gui.nvdaControls import DPIScaledDialog
from logHandler import log
import wx

from ..exceptions import AuthenticationError, CancellationError, OCRError
from ._beMyEyesClient import BeMyEyesClient, getStoredAccountLabel, saveAccount

addonHandler.initTranslation()


class BeMyEyesLoginDialog(DPIScaledDialog):
	def __init__(self, parent: wx.Window) -> None:
		# Translators: Title of the Be My Eyes login dialog.
		super().__init__(parent, title=_("Log in to Be My Eyes"))
		self._cancelEvent = Event()
		self._isClosing = False
		self._makeSettings()
		self.CentreOnParent()
		self.Bind(wx.EVT_CLOSE, self._onClose)

	def _makeSettings(self) -> None:
		mainSizer = wx.BoxSizer(wx.VERTICAL)
		helper = guiHelper.BoxSizerHelper(self, orientation=wx.VERTICAL)
		# Translators: Be My Eyes account email field.
		self._emailText = helper.addLabeledControl(_("&Email"), wx.TextCtrl)
		self._emailText.SetValue(parseaddr(getStoredAccountLabel())[1])
		# Translators: Be My Eyes account password field.
		self._passwordText = helper.addLabeledControl(_("&Password"), wx.TextCtrl, style=wx.TE_PASSWORD)
		buttons = guiHelper.ButtonHelper(wx.HORIZONTAL)
		# Translators: Button for submitting the account credentials.
		self._loginButton = buttons.addButton(self, id=wx.ID_OK, label=_("&Log in"))
		# Translators: Button for cancelling login.
		buttons.addButton(self, id=wx.ID_CANCEL, label=_("&Cancel"))
		helper.addDialogDismissButtons(buttons, separated=True)
		self.Bind(wx.EVT_BUTTON, self._onLogin, id=wx.ID_OK)
		self.Bind(wx.EVT_BUTTON, self._onCancel, id=wx.ID_CANCEL)
		mainSizer.Add(helper.sizer, border=guiHelper.BORDER_FOR_DIALOGS, flag=wx.ALL)
		mainSizer.Fit(self)
		self.SetSizer(mainSizer)
		self.AffirmativeId = wx.ID_OK
		self.EscapeId = wx.ID_CANCEL
		(self._passwordText if self._emailText.GetValue() else self._emailText).SetFocus()

	def _onLogin(self, evt: wx.CommandEvent) -> None:
		if not self._loginButton.IsEnabled():
			return
		email = self._emailText.GetValue().strip()
		password = self._passwordText.GetValue()
		if not email or not password:
			# Translators: An error when a login field is empty.
			self._onLoginFailed(_("Email and password are required."))
			(self._emailText if not email else self._passwordText).SetFocus()
			return
		self._passwordText.ChangeValue("")
		self._setControlsEnabled(False)
		Thread(target=self._loginWorker, args=(email, password), name="BeMyEyesLogin", daemon=True).start()

	def _loginWorker(self, email: str, password: str) -> None:
		try:
			client = BeMyEyesClient(cancellationChecker=self._checkCancelled)
			token, account = client.authenticate(email, password)
		except CancellationError:
			return
		except OCRError as e:
			log.warning("Be My Eyes login failed.", exc_info=True)
			wx.CallAfter(self._onLoginFailed, str(e))
		except Exception:
			log.exception("Unexpected Be My Eyes login failure.")
			# Translators: An unexpected error while logging in to Be My Eyes.
			wx.CallAfter(self._onLoginFailed, _("Be My Eyes login failed with an unexpected error."))
		else:
			wx.CallAfter(self._onLoginSucceeded, token, account)

	def _checkCancelled(self) -> None:
		if self._cancelEvent.is_set():
			raise CancellationError("Be My Eyes login cancelled.", self._cancelEvent)

	def _onLoginSucceeded(self, token: str, account: str) -> None:
		if self._isClosing:
			return
		try:
			saveAccount(token, account)
		except AuthenticationError as e:
			log.warning("Could not save Be My Eyes login.", exc_info=True)
			self._onLoginFailed(str(e))
			return
		self._isClosing = True
		self.EndModal(wx.ID_OK)

	def _onLoginFailed(self, message: str) -> None:
		if self._isClosing:
			return
		self._setControlsEnabled(True)
		gui.messageBox(
			message,
			# Translators: Title of the Be My Eyes login error dialog.
			_("Be My Eyes Login Failed"),
			wx.OK | wx.ICON_ERROR,
			self,
		)
		self._passwordText.SetFocus()

	def _setControlsEnabled(self, enabled: bool) -> None:
		self._emailText.Enable(enabled)
		self._passwordText.Enable(enabled)
		self._loginButton.Enable(enabled)

	def _onCancel(self, evt: wx.CommandEvent) -> None:
		self._isClosing = True
		self._cancelEvent.set()
		self.EndModal(wx.ID_CANCEL)

	def _onClose(self, evt: wx.CloseEvent) -> None:
		self._isClosing = True
		self._cancelEvent.set()
		evt.Skip()

	def Destroy(self) -> bool:
		self._isClosing = True
		self._cancelEvent.set()
		return super().Destroy()
