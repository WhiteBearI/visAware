# Copyright (C) 2026 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from io import BytesIO
import json
from queue import Empty, Queue
from threading import Event, Lock, Thread
from time import monotonic
from typing import Any, cast
from urllib.parse import urlsplit

import addonHandler
import config
import languageHandler
from logHandler import log
from PIL import Image
import requests
import wx

from .. import network
from ..exceptions import ApiError, AuthenticationError, OCRError
from ..secure_storage import SecureStorageError, protectString, unprotectString

addonHandler.initTranslation()

APP_SECRET = "55519e815ff7b09ab971de5564baa282eca53af1eb528385fb98a34f2010e8c7"
USER_AGENT = "okhttp/3.14.9"
API_BASE_URL = "https://api.bemyeyes.com/api/v2/"
GATEWAY_BASE_URL = "https://gateway.bemyeyes.com/api/v2/"
REQUEST_TIMEOUT = 30
DESCRIPTION_TIMEOUT = 120
DEFAULT_MAX_IMAGE_DIMENSION = 2000
_pendingQuestions: set[int] = set()
_questionsLock = Lock()


def _getEngineConfig() -> Any:
	return cast(Any, config.conf).get("ImageDescriber", {}).get("beMyEyes", {})


def getStoredToken() -> str:
	try:
		return unprotectString(_getEngineConfig().get("encryptedToken", ""))
	except SecureStorageError:
		log.warning("Stored Be My Eyes token could not be unprotected.", exc_info=True)
		return ""


def getStoredAccountLabel() -> str:
	return _getEngineConfig().get("accountLabel", "")


def saveAccount(token: str, accountLabel: str) -> None:
	try:
		encryptedToken = protectString(token)
	except SecureStorageError as e:
		# Translators: An error when Be My Eyes account data cannot be saved.
		raise AuthenticationError(_("Could not save Be My Eyes account details.")) from e
	conf = cast(Any, config.conf)
	settings = conf["ImageDescriber"]
	section = settings["beMyEyes"]
	# Preserve only this profile's values so rollback does not turn inheritance into overrides.
	previousValues = conf.profiles[-1].get("ImageDescriber", {}).get("beMyEyes", {}).copy()
	section["encryptedToken"] = encryptedToken
	section["accountLabel"] = accountLabel
	try:
		config.conf.save()
	except OSError as e:
		settings["beMyEyes"] = previousValues
		# Translators: An error when Be My Eyes account data cannot be saved.
		raise AuthenticationError(_("Could not save Be My Eyes account details.")) from e


def _clearExpiredToken(section: Any, token: str) -> None:
	# NVDA rebuilds aggregated sections when switching profiles.
	if _getEngineConfig() is not section or getStoredToken() != token:
		return
	try:
		saveAccount("", getStoredAccountLabel())
	except AuthenticationError:
		log.warning("Could not clear the expired Be My Eyes token.", exc_info=True)


def _getHttpStatus(error: BaseException | None) -> int | None:
	while error is not None:
		if isinstance(error, requests.exceptions.HTTPError) and error.response is not None:
			return error.response.status_code
		error = error.__cause__
	return None


class BeMyEyesClient:
	def __init__(self, cancellationChecker: Callable[[], None] | None = None) -> None:
		self._cancellationChecker = cancellationChecker
		self._configSection = _getEngineConfig()
		self._token = getStoredToken()
		self._gatewayWriteLock = Lock()

	@property
	def _baseHeaders(self) -> dict[str, str]:
		return {
			"User-Agent": USER_AGENT,
			"bemyeyes-app-secret": APP_SECRET,
			"Accept-Language": languageHandler.getLanguage().replace("_", "-"),
		}

	@property
	def _headers(self) -> dict[str, str]:
		return {**self._baseHeaders, "Authorization": f"Token {self._token}"}

	def _checkCancelled(self) -> None:
		if self._cancellationChecker:
			self._cancellationChecker()

	def _request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
		headers = kwargs.pop("headers", self._headers)
		self._checkCancelled()
		try:
			response = network.sendRequest(
				method,
				url,
				headers=headers,
				cancelCheck=self._cancellationChecker,
				timeout=REQUEST_TIMEOUT,
				# Replaying a message or a gateway poll can duplicate work or lose events.
				retry=False,
				**kwargs,
			)
		except OCRError as e:
			self._checkCancelled()
			status = _getHttpStatus(e)
			if status == 401 and "Authorization" in headers:
				wx.CallAfter(_clearExpiredToken, self._configSection, self._token)
				# Translators: An error when the saved Be My Eyes login is no longer valid.
				raise AuthenticationError(_("Be My Eyes login has expired. Please log in again.")) from e
			if status == 403 and "Authorization" in headers:
				# Translators: A permission error which does not invalidate the saved Be My Eyes login.
				raise ApiError(_("Be My Eyes denied access to this resource.")) from e
			raise
		self._checkCancelled()
		return response

	def _apiRequest(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
		response = self._request(method, API_BASE_URL + path, **kwargs)
		try:
			result = response.json()
		except ValueError as e:
			# Translators: An error when the Be My Eyes API response cannot be understood.
			raise ApiError(_("Be My Eyes returned an invalid response.")) from e
		if not isinstance(result, dict):
			# Translators: An error when the Be My Eyes API response cannot be understood.
			raise ApiError(_("Be My Eyes returned an invalid response."))
		return result

	def authenticate(self, email: str, password: str) -> tuple[str, str]:
		if not email or not password:
			# Translators: An error when a login field is empty.
			raise AuthenticationError(_("Email and password are required."))
		result = self._apiRequest(
			"POST",
			"auth/login-email",
			json={
				"email": email,
				"password": password,
				"extra": {
					"device_type": "android",
					"screen_reader_enabled": True,
					"content_size": "1.15",
					"accesibility_content_size_enabled": False,
				},
			},
			headers=self._baseHeaders,
		)
		if result.get("email_verification_required"):
			# Translators: An error when Be My Eyes requires email verification.
			raise AuthenticationError(_("Please verify your Be My Eyes account email before logging in."))
		if result.get("password_change_required"):
			# Translators: An error when Be My Eyes requires a password change.
			raise AuthenticationError(_("Please change your Be My Eyes password before logging in."))
		token = result.get("token")
		if not isinstance(token, str) or not token.strip():
			# Translators: An error when Be My Eyes does not return a login token.
			raise AuthenticationError(_("Be My Eyes login returned an invalid response."))
		user = result.get("user")
		if not isinstance(user, dict):
			return token, email
		name = " ".join(str(user.get(key) or "").strip() for key in ("first_name", "last_name")).strip()
		email = str(user.get("email") or email).strip()
		return token, f"{name} <{email}>" if name else email

	def describeImage(self, imageContent: bytes) -> dict[str, Any]:
		with requests.Session() as session, self._gatewaySession(session) as events:
			userConfig = self._apiRequest("GET", "app-config/user", session=session)
			imageBytes, width, height, imageType = self._prepareImage(imageContent, userConfig)
			chat = self._apiRequest("POST", "chats", json={"context": "chat"}, session=session)
			chatId = chat.get("id")
			if type(chatId) is not int or chatId <= 0:
				# Translators: An error when Be My Eyes cannot create a chat.
				raise ApiError(_("Be My Eyes could not create a chat."))
			upload = self._apiRequest(
				"POST",
				f"chats/{chatId}/request-image-upload",
				json={"format": imageType, "width": width, "height": height},
				session=session,
			)
			self._uploadImage(upload, imageBytes, f"image/{imageType}")
			message = self._apiRequest(
				"POST",
				f"chats/{chatId}/messages",
				json={"role": "user", "type": "text", "chat_image_id": upload["chat_image_id"]},
				session=session,
			)
			images = message.get("images")
			if not (
				isinstance(images, list)
				and images
				and isinstance(images[0], dict)
				and images[0].get("upload_finished")
			):
				# Translators: An error when Be My Eyes cannot confirm image upload completion.
				raise ApiError(_("Be My Eyes could not confirm that the image was uploaded."))
			return {"text": self._receiveDescription(events, chatId), "chatId": chatId}

	def askQuestion(self, chatId: int, question: str) -> str:
		self._checkCancelled()
		with _questionsLock:
			if chatId in _pendingQuestions:
				# Translators: A previous question may still produce a reply in this Be My Eyes chat.
				raise ApiError(
					_(
						"A previous Be My Eyes question is still pending or was interrupted. "
						"Recognize the image again before asking another question."
					)
				)
			_pendingQuestions.add(chatId)
		submitted = answered = False
		try:
			with requests.Session() as session, self._gatewaySession(session) as events:
				self._checkCancelled()
				submitted = True
				try:
					self._apiRequest(
						"POST",
						f"chats/{chatId}/messages",
						json={"role": "user", "type": "text", "data": question},
						session=session,
					)
				except ApiError as e:
					if _getHttpStatus(e) in {400, 401, 403, 404, 422, 429}:
						submitted = False
					raise
				answer = self._receiveDescription(events, chatId)
				answered = True
				return answer
		finally:
			# ponytail: interrupted chats stay blocked for this NVDA session; use reply IDs if the API adds them.
			if not submitted or answered:
				with _questionsLock:
					_pendingQuestions.remove(chatId)

	def _gatewayRequest(
		self,
		method: str,
		sid: str = "",
		data: str | None = None,
		stopped: Event | None = None,
		*,
		session: requests.Session,
	) -> str:
		params = {"EIO": "4", "transport": "polling"}
		if sid:
			params["sid"] = sid
		with self._gatewayWriteLock if method == "POST" else nullcontext():
			if stopped is not None and stopped.is_set():
				return ""
			return self._request(
				method, GATEWAY_BASE_URL + "socket/", params=params, data=data, session=session
			).text

	@contextmanager
	def _gatewaySession(self, session: requests.Session) -> Iterator[Queue[list[Any] | Exception]]:
		if not self._token:
			# Translators: An error when Be My Eyes has no usable saved login.
			raise AuthenticationError(_("Please log in to your Be My Eyes account in the engine settings."))
		opening = self._gatewayRequest("GET", session=session).split(chr(30), 1)[0]
		try:
			if not opening.startswith("0"):
				raise ValueError("Expected an Engine.IO open packet")
			sid = json.loads(opening[1:])["sid"]
			if not isinstance(sid, str) or not sid:
				raise ValueError("Missing gateway session ID")
		except (ValueError, KeyError, TypeError) as e:
			# Translators: An error when the Be My Eyes gateway cannot establish a session.
			raise ApiError(_("Be My Eyes returned an invalid gateway session.")) from e
		events: Queue[list[Any] | Exception] = Queue()
		stopped = Event()
		reader = Thread(
			target=self._pollGateway, args=(sid, stopped, events), name="BeMyEyesGateway", daemon=True
		)
		try:
			if self._gatewayRequest("POST", sid, "40", session=session) != "ok":
				# Translators: An error when Be My Eyes gateway authentication fails.
				raise ApiError(_("Be My Eyes gateway authentication failed."))
			reader.start()
			for event in self._receiveEvents(events, REQUEST_TIMEOUT):
				if event[0] == "AUTHENTICATED":
					break
			else:
				# Translators: An error when Be My Eyes gateway authentication cannot be confirmed.
				raise ApiError(_("Be My Eyes gateway authentication was not confirmed."))
			self._gatewayRequest("POST", sid, '42["ENABLE_CHAT","{}"]', session=session)
			yield events
		finally:
			stopped.set()
			try:
				with self._gatewayWriteLock:
					network.sendRequest(
						"POST",
						GATEWAY_BASE_URL + "socket/",
						headers=self._headers,
						params={"EIO": "4", "transport": "polling", "sid": sid},
						data="1",
						timeout=5,
						retry=False,
						session=session,
					)
			except OCRError:
				log.debugWarning("Could not close the Be My Eyes gateway session.", exc_info=True)
			finally:
				if reader.ident is not None:
					reader.join(REQUEST_TIMEOUT)
					if reader.is_alive():
						log.warning("Be My Eyes gateway reader did not stop before the request timeout.")
		self._checkCancelled()

	def _pollGateway(self, sid: str, stopped: Event, events: Queue[list[Any] | Exception]) -> None:
		try:
			with requests.Session() as session:
				while not stopped.is_set():
					for packet in self._gatewayRequest("GET", sid, stopped=stopped, session=session).split(
						chr(30)
					):
						if stopped.is_set():
							return
						self._checkCancelled()
						if packet == "2":
							self._gatewayRequest("POST", sid, "3", stopped=stopped, session=session)
							continue
						if packet in {"1", "41"} or packet.startswith("44"):
							# Translators: An error when the Be My Eyes chat connection closes before an answer.
							raise ApiError(_("The Be My Eyes chat connection closed. Please try again."))
						if not packet.startswith("42"):
							continue
						try:
							event = json.loads(packet[2:])
						except ValueError:
							log.debugWarning("Be My Eyes ignored a malformed gateway event.")
							continue
						if isinstance(event, list) and event and isinstance(event[0], str):
							events.put(event)
		except Exception as e:
			events.put(e)

	def _receiveEvents(self, events: Queue[list[Any] | Exception], timeout: float) -> Iterator[list[Any]]:
		deadline = monotonic() + timeout
		while (remaining := deadline - monotonic()) > 0:
			self._checkCancelled()
			try:
				event = events.get(timeout=min(0.1, remaining))
			except Empty:
				continue
			if isinstance(event, Exception):
				raise event
			yield event

	def _receiveDescription(self, events: Queue[list[Any] | Exception], chatId: int) -> str:
		for event in self._receiveEvents(events, DESCRIPTION_TIMEOUT):
			if len(event) != 2 or event[0] != "NEW_CHAT_MESSAGE":
				continue
			message = event[1]
			if not isinstance(message, dict):
				log.debugWarning("Be My Eyes ignored an invalid chat message.")
				continue
			if message.get("user") or message.get("session") != chatId:
				continue
			text = message.get("data")
			if isinstance(text, str) and text.strip():
				return text.strip()
		# Translators: An error when Be My Eyes does not return an answer in time.
		raise ApiError(_("Be My Eyes did not return a description before the request timed out."))

	def _uploadImage(self, upload: dict[str, Any], imageBytes: bytes, mimeType: str) -> None:
		url = upload.get("url", "")
		fields = upload.get("fields")
		imageId = upload.get("chat_image_id")
		try:
			if not isinstance(url, str):
				raise ValueError("Invalid upload URL")
			parsedUrl = urlsplit(url)
			if not (
				parsedUrl.scheme == "https"
				and parsedUrl.hostname
				and not parsedUrl.username
				and isinstance(fields, dict)
				and type(imageId) is int
				and imageId > 0
			):
				raise ValueError("Invalid upload URL, fields or image ID")
		except ValueError as e:
			# Translators: An error when Be My Eyes returns incomplete image upload instructions.
			raise ApiError(_("Be My Eyes returned invalid upload configuration.")) from e
		data = {key: str(value) for key, value in fields.items()}
		data.setdefault("Content-Type", mimeType)
		self._request(
			"POST",
			url,
			headers={"User-Agent": USER_AGENT},
			data=data,
			files={"file": ("file", imageBytes, mimeType)},
		)

	@staticmethod
	def _prepareImage(imageContent: bytes, userConfig: dict[str, Any]) -> tuple[bytes, int, int, str]:
		try:
			maxDimension = int(userConfig.get("chat_image_max_dimension") or DEFAULT_MAX_IMAGE_DIMENSION)
			if maxDimension <= 0:
				raise ValueError("Image dimension must be positive")
		except (ValueError, TypeError, OverflowError) as e:
			# Translators: An error when Be My Eyes returns invalid image size limits.
			raise ApiError(_("Be My Eyes returned invalid account configuration.")) from e
		imageType = str(userConfig.get("chat_image_type") or "png").lower()
		imageType = "jpeg" if imageType in {"jpg", "jpeg"} else "png"
		with Image.open(BytesIO(imageContent)) as original:
			with original.convert("RGB") as image:
				image.thumbnail((maxDimension, maxDimension), Image.Resampling.LANCZOS)
				output = BytesIO()
				image.save(output, format=imageType.upper())
				return output.getvalue(), image.width, image.height, imageType
