from __future__ import annotations

import builtins
from collections import ChainMap, deque
import importlib.util
from io import BytesIO
import json
from pathlib import Path
from queue import Queue
import sys
from threading import Event
import types
import unittest
from unittest.mock import Mock, patch

import requests

try:
	from PIL import Image
except ImportError:
	Image = None

from tests.test_request_log_redaction import load_recog_handler_module


PACKAGE = "addon.globalPlugins.visAware"
ROOT = Path(__file__).resolve().parents[1] / "addon/globalPlugins/visAware"


def loadModule(name, path):
	spec = importlib.util.spec_from_file_location(f"{PACKAGE}.{name}", ROOT / path)
	module = importlib.util.module_from_spec(spec)
	sys.modules[spec.name] = module
	spec.loader.exec_module(module)
	return module


def response(payload, status=200):
	result = requests.Response()
	result.status_code = status
	result._content = (payload if isinstance(payload, str) else json.dumps(payload)).encode()
	return result


def chatEvent(text, **fields):
	return "42" + json.dumps(["NEW_CHAT_MESSAGE", {"data": text, **fields}])


class BeMyEyesTest(unittest.TestCase):
	def setUp(self):
		self.addCleanup(patch.stopall)
		patch.dict(sys.modules).start()
		patch.object(builtins, "_", lambda value: value, create=True).start()
		if Image is not None:
			imageClass = Image.Image
			patch.object(Image, "Image", imageClass).start()
		self.handler = load_recog_handler_module()
		if Image is not None:
			Image.Image = imageClass
			sys.modules["PIL"].Image = Image
			sys.modules["PIL.Image"] = Image
		self.log = Mock()
		sys.modules["logHandler"].log = self.log
		sys.modules["languageHandler"] = types.SimpleNamespace(getLanguage=lambda: "zh_CN")
		sys.modules["gui.nvdaControls"].DPIScaledDialog = object
		sys.modules["wx"].Window = object
		sys.modules["wx"].CloseEvent = object
		sys.modules["wx"].ID_OK = 1
		sys.modules["wx"].ID_CANCEL = 2
		self.callbacks = []
		sys.modules["wx"].CallAfter = lambda fn, *args: self.callbacks.append((fn, args))
		self.conf = type("Config", (dict,), {"save": Mock()})({"ImageDescriber": {"beMyEyes": {}}})
		self.conf.profiles = [self.conf]
		sys.modules["config"].conf = self.conf
		self.errors = loadModule("exceptions", "exceptions.py")
		self.network = loadModule("network", "network.py")
		loadModule("secure_storage", "secure_storage.py")
		self.history = loadModule("recogHistory", "recogHistory.py")
		self.handler.recogHistory = self.history
		self.handler.network = self.network
		self.handler.ApiError = self.errors.ApiError
		self.handler.CancellationError = self.errors.CancellationError
		self.handler.wx.CallAfter = sys.modules["wx"].CallAfter
		self.clientModule = loadModule(
			"imageDescribers._beMyEyesClient", "imageDescribers/_beMyEyesClient.py"
		)
		self.guiModule = loadModule("imageDescribers._beMyEyesGui", "imageDescribers/_beMyEyesGui.py")
		sys.modules[f"{PACKAGE}.engineGUIHelper"].ButtonEngineSetting = lambda name, label: (
			types.SimpleNamespace(name=name, displayNameWithAccelerator=label)
		)
		self.engineModule = loadModule("imageDescribers.beMyEyes", "imageDescribers/beMyEyes.py")
		self.clientModule.saveAccount("token", "Reader <reader@example.test>")
		self.check = Mock()
		self.client = self.clientModule.BeMyEyesClient(self.check)
		self.http = patch.object(requests, "request").start()

	def queueResponses(self, *payloads):
		self.http.side_effect = [response(payload) for payload in payloads]

	def mockGateway(self, apiResponses=(), answers=(), authentication=('42["AUTHENTICATED",{}]',)):
		packets = Queue()
		apiResponses = deque(apiResponses)
		pong = Event()

		def send(**kwargs):
			if kwargs["url"].startswith(self.clientModule.GATEWAY_BASE_URL):
				if kwargs["method"] == "GET":
					if "sid" not in kwargs["params"]:
						return response('0{"sid":"session","pingInterval":25000,"pingTimeout":20000}')
					return response(packets.get(timeout=2))
				data = kwargs.get("data")
				if data == "40":
					for packet in authentication:
						packets.put(packet)
				elif data == "3":
					pong.set()
				elif data == "1":
					packets.put("1")
				return response("ok")
			if kwargs["url"].endswith("/messages"):
				for packet in answers:
					packets.put(packet)
			return response(apiResponses.popleft())

		self.http.side_effect = send
		return packets, pong

	def runCallbacks(self):
		for fn, args in self.callbacks:
			fn(*args)
		self.callbacks.clear()

	@unittest.skipIf(Image is None, "Pillow is supplied by NVDA; run with uv run --with pillow pytest")
	def test_image_conversion_obeys_service_size_and_format(self):
		with Image.new("RGBA", (80, 40), (20, 30, 40, 255)) as image:
			output = BytesIO()
			image.save(output, format="PNG")
		for imageType in ("png", "jpg"):
			content, width, height, formatName = self.client._prepareImage(
				output.getvalue(), {"chat_image_max_dimension": 40, "chat_image_type": imageType}
			)
			self.assertEqual((width, height), (40, 20))
			with Image.open(BytesIO(content)) as converted:
				self.assertEqual(converted.size, (40, 20))
				self.assertEqual(converted.format.lower(), formatName)

	def test_description_and_followup_use_shared_results_and_same_chat(self):
		self.mockGateway(
			(
				{"chat_image_max_dimension": 40, "chat_image_type": "jpg"},
				{"id": 7},
				{"url": "https://upload.example.test/", "fields": {"key": "signed-key"}, "chat_image_id": 8},
				"",
				{"images": [{"upload_finished": True}]},
			),
			(
				"2" + chr(30) + chatEvent("own question", user={"id": 1}),
				chatEvent("other chat", session=99) + chr(30) + chatEvent(" A blue door. ", session=7),
			),
		)
		resultModule = sys.modules["contentRecog"]
		resultModule.SimpleTextResult = lambda text: types.SimpleNamespace(text=text)
		self.handler.LinesWordsResult = lambda data, info: types.SimpleNamespace(data=data, imageInfo=info)
		engine = self.engineModule.CustomContentRecognizer()
		engine.originalImage = object()
		params = engine._buildRequestParams(b"image", self.handler.RecognitionRequest(True, False))
		with patch.object(
			self.clientModule.BeMyEyesClient, "_prepareImage", return_value=(b"jpeg", 40, 20, "jpeg")
		):
			result = engine._handleStandardResponse(
				params, None, Event(), self.handler.RecognitionRequest(True, False)
			)
		self.assertEqual(result.text, "A blue door.")
		entry = self.history.getAttachedEntry(result)
		self.assertEqual(entry.response, {"text": "A blue door.", "chatId": 7})
		self.assertIs(entry.image, engine.originalImage)
		calls = [call.kwargs for call in self.http.call_args_list]
		uploadCall = next(call for call in calls if "files" in call)
		self.assertEqual(uploadCall["headers"], {"User-Agent": self.clientModule.USER_AGENT})
		self.assertEqual(uploadCall["files"]["file"][1:], (b"jpeg", "image/jpeg"))
		self.assertTrue(all(call["timeout"] <= 30 for call in calls))
		self.assertTrue(any(call.get("data") == "3" for call in calls))
		self.assertEqual(calls[-1]["data"], "1")
		self.mockGateway(({},), (chatEvent("It is open.", session=7),))
		answer = engine.askQuestion(types.SimpleNamespace(response=entry.response), "Is it open?", self.check)
		self.assertEqual(answer, "It is open.")
		messageCall = next(
			call.kwargs
			for call in reversed(self.http.call_args_list)
			if call.kwargs["url"].endswith("/messages")
		)
		self.assertTrue(messageCall["url"].endswith("chats/7/messages"))
		self.assertEqual(messageCall["json"]["data"], "Is it open?")
		self.check.assert_called()
		with patch.object(engine, "_getApiResult", return_value=entry.response):
			richResult = engine._handleStandardResponse(
				{}, "geometry", Event(), self.handler.RecognitionRequest(False, False)
			)
		self.assertEqual(richResult.data[0][0]["text"], result.text)
		self.assertEqual(richResult.imageInfo, "geometry")

	def test_login_validation_storage_and_failed_save(self):
		for payload, message in (
			({"email_verification_required": True}, "verify"),
			({"password_change_required": True}, "change"),
			({"token": None}, "invalid"),
			({"token": " "}, "invalid"),
		):
			with self.subTest(payload=payload):
				self.queueResponses(payload)
				with self.assertRaisesRegex(self.errors.AuthenticationError, message):
					self.client.authenticate("reader@example.test", "password")
		self.queueResponses(
			{"token": "new-token", "user": {"first_name": "Reader", "email": "reader@example.test"}}
		)
		token, label = self.client.authenticate("reader@example.test", "password")
		self.assertNotIn("Authorization", self.http.call_args.kwargs["headers"])
		self.assertEqual(self.clientModule.getStoredToken(), "token")
		self.clientModule.saveAccount(token, label)
		section = self.conf["ImageDescriber"]["beMyEyes"]
		self.assertNotEqual(section["encryptedToken"], token)
		self.assertNotIn("password", str(section))
		self.assertEqual(self.clientModule.getStoredToken(), "new-token")
		self.conf.save.side_effect = PermissionError("read only")
		with self.assertRaises(self.errors.AuthenticationError):
			self.clientModule.saveAccount("", "")
		self.assertEqual(self.clientModule.getStoredToken(), "new-token")
		self.assertEqual(self.clientModule.getStoredAccountLabel(), label)

	def test_failed_account_save_preserves_profile_inheritance(self):
		originalSection = self.conf["ImageDescriber"]["beMyEyes"].copy()
		for overrides in ({}, {"accountLabel": "Profile label"}):
			with self.subTest(overrides=overrides):
				baseSection = originalSection.copy()
				profileSettings = {"beMyEyes": overrides.copy()}

				class Settings(dict):
					def __setitem__(self, key, value):
						profileSettings[key] = value
						super().__setitem__(key, ChainMap(value, baseSection))

				self.conf.profiles = [
					{"ImageDescriber": {"beMyEyes": baseSection}},
					{"ImageDescriber": profileSettings},
				]
				self.conf["ImageDescriber"] = Settings(
					beMyEyes=ChainMap(profileSettings["beMyEyes"], baseSection)
				)
				self.conf.save.side_effect = PermissionError("read only")
				with self.assertRaises(self.errors.AuthenticationError):
					self.clientModule.saveAccount("", "")
				self.assertEqual(profileSettings["beMyEyes"], overrides)
				self.assertEqual(baseSection, originalSection)
				self.assertEqual(self.clientModule.getStoredToken(), "token")
				self.assertEqual(
					self.clientModule.getStoredAccountLabel(),
					overrides.get("accountLabel", originalSection["accountLabel"]),
				)
				baseSection["encryptedToken"] = self.clientModule.protectString("new-token")
				self.assertEqual(self.clientModule.getStoredToken(), "new-token")

	def test_expiration_is_queued_and_does_not_clear_new_login(self):
		self.http.side_effect = None
		self.http.return_value = response({}, 401)
		with self.assertRaisesRegex(self.errors.AuthenticationError, "expired"):
			self.client.askQuestion(7, "Question")
		self.assertEqual(self.clientModule.getStoredToken(), "token")
		self.clientModule.saveAccount("new-token", "new account")
		self.runCallbacks()
		self.assertEqual(self.clientModule.getStoredToken(), "new-token")
		client = self.clientModule.BeMyEyesClient()
		with self.assertRaises(self.errors.AuthenticationError):
			client.askQuestion(7, "Question")
		self.runCallbacks()
		self.assertEqual(self.clientModule.getStoredToken(), "")
		self.assertEqual(self.clientModule.getStoredAccountLabel(), "new account")

	def test_expiration_does_not_override_another_profile_inheriting_the_same_token(self):
		self.http.return_value = response({}, 401)
		with self.assertRaises(self.errors.AuthenticationError):
			self.client.askQuestion(7, "Question")
		originalSection = self.conf["ImageDescriber"]["beMyEyes"]
		inheritedSection = ChainMap({}, originalSection)
		self.conf["ImageDescriber"]["beMyEyes"] = inheritedSection
		self.conf.save.reset_mock()
		self.runCallbacks()
		self.assertEqual(inheritedSection.maps[0], {})
		self.assertEqual(self.clientModule.getStoredToken(), "token")
		self.conf.save.assert_not_called()
		self.conf["ImageDescriber"]["beMyEyes"] = originalSection
		self.clientModule.saveAccount("new-token", "new account")
		self.conf["ImageDescriber"]["beMyEyes"] = inheritedSection
		self.assertEqual(self.clientModule.getStoredToken(), "new-token")

	def test_upload_authentication_failure_does_not_expire_account(self):
		self.http.return_value = response({}, 403)
		with self.assertRaises(self.errors.AuthenticationError):
			self.client._uploadImage(
				{"url": "https://upload.example.test", "fields": {}, "chat_image_id": 8},
				b"image",
				"image/png",
			)
		self.assertEqual(self.callbacks, [])
		self.assertEqual(self.clientModule.getStoredToken(), "token")

	def test_invalid_upload_instructions_never_send_image(self):
		for fields in (
			{"url": "http://upload.example.test"},
			{"url": "https://[invalid"},
			{"url": None},
			{"fields": []},
			{"chat_image_id": None},
		):
			with self.subTest(fields=fields):
				upload = {"url": "https://upload.example.test", "fields": {}, "chat_image_id": 8, **fields}
				with self.assertRaisesRegex(self.errors.ApiError, "upload configuration"):
					self.client._uploadImage(upload, b"image", "image/png")
		self.http.assert_not_called()

	def test_permission_denied_does_not_expire_account(self):
		self.http.return_value = response({"message": "Forbidden"}, 403)
		with self.assertRaises(self.errors.ApiError) as error:
			self.client._apiRequest("POST", "chats/7/messages", json={"data": "question"})
		self.runCallbacks()
		self.assertEqual(self.clientModule.getStoredToken(), "token")
		self.assertNotIn("expired", str(error.exception))

	def test_rejected_question_can_be_retried_in_the_same_chat(self):
		for status in (429, 403, 401, 400, 404, 422):
			with self.subTest(status=status):
				self.mockGateway()
				send = self.http.side_effect

				def reject(**kwargs):
					if kwargs["url"].endswith("/messages"):
						return response({"message": "Request rejected"}, status)
					return send(**kwargs)

				self.http.side_effect = reject
				with self.assertRaises(self.errors.ApiError):
					self.client.askQuestion(7, "question")
				self.mockGateway(({},), (chatEvent("answer", session=7),))
				self.assertEqual(self.client.askQuestion(7, "retry"), "answer")

	def test_uncertain_http_submission_keeps_chat_blocked(self):
		for chatId, (status, payload) in enumerate(((408, {}), (500, {}), (200, "invalid")), 7):
			with self.subTest(status=status):
				self.mockGateway()
				send = self.http.side_effect

				def fail(**kwargs):
					if kwargs["url"].endswith("/messages"):
						return response(payload, status)
					return send(**kwargs)

				self.http.side_effect = fail
				with self.assertRaises(self.errors.ApiError):
					self.client.askQuestion(chatId, "question")
				self.http.reset_mock()
				with self.assertRaisesRegex(self.errors.ApiError, "Recognize the image again"):
					self.client.askQuestion(chatId, "retry")
				self.http.assert_not_called()

	def test_gateway_failure_after_submission_does_not_unlock_chat(self):
		packets, _ = self.mockGateway(({},))
		send = self.http.side_effect
		submitted = Event()

		def failGateway(**kwargs):
			if kwargs["method"] == "GET" and submitted.is_set():
				return response({}, 429)
			result = send(**kwargs)
			if kwargs["url"].endswith("/messages"):
				submitted.set()
				packets.put("6")
			return result

		self.http.side_effect = failGateway
		with self.assertRaises(self.errors.ApiError):
			self.client.askQuestion(7, "question")
		self.http.reset_mock()
		with self.assertRaisesRegex(self.errors.ApiError, "Recognize the image again"):
			self.client.askQuestion(7, "retry")
		self.http.assert_not_called()

	def test_gateway_matches_the_service_session_field(self):
		events = Queue()
		for message in (
			{"data": "missing chat"},
			{"data": "another chat", "session": 8},
			{"data": "own question", "session": 7, "user": {"id": 1}},
			{"data": "correct answer", "session": 7, "user": None},
		):
			events.put(["NEW_CHAT_MESSAGE", message])
		self.assertEqual(self.client._receiveDescription(events, 7), "correct answer")

	def test_cancellation_takes_precedence_over_network_failure(self):
		def fail(**kwargs):
			self.check.side_effect = self.errors.CancellationError("cancelled")
			raise requests.exceptions.ReadTimeout()

		self.http.side_effect = fail
		with self.assertRaises(self.errors.CancellationError):
			self.client.authenticate("email", "password")
		self.assertEqual(self.http.call_count, 1)

	def test_cancellation_stops_subsequent_requests_and_blocks_late_answer_reuse(self):
		self.mockGateway(({},), (chatEvent("old answer", session=7), chatEvent("new answer", session=7)))
		send = self.http.side_effect

		def cancelAfterSubmission(**kwargs):
			result = send(**kwargs)
			if kwargs["url"].endswith("/messages"):
				self.check.side_effect = self.errors.CancellationError("cancelled")
			return result

		self.http.side_effect = cancelAfterSubmission
		with self.assertRaises(self.errors.CancellationError):
			self.client.askQuestion(7, "old question")
		self.assertEqual(self.http.call_args.kwargs["data"], "1")
		self.http.reset_mock()
		with self.assertRaises(self.errors.CancellationError):
			self.client.askQuestion(7, "new question")
		# Reopening the dialog creates a different client for the same history entry.
		self.mockGateway(({},), (chatEvent("old answer", session=7), chatEvent("new answer", session=7)))
		client = self.clientModule.BeMyEyesClient()
		with self.assertRaisesRegex(self.errors.ApiError, "Recognize the image again"):
			client.askQuestion(7, "new question")
		self.http.assert_not_called()
		self.mockGateway(({},), (chatEvent("fresh answer", session=8),))
		self.assertEqual(client.askQuestion(8, "new question"), "fresh answer")

	def test_network_failure_is_not_retried_and_cleanup_keeps_original_error(self):
		self.http.side_effect = requests.exceptions.ReadTimeout("timeout?sid=private")
		with self.assertRaises(self.errors.NetworkError):
			self.client.authenticate("email", "password")
		self.assertEqual(self.http.call_count, 1)
		self.http.side_effect = [
			response('0{"sid":"session"}'),
			response("invalid"),
			requests.exceptions.ReadTimeout(),
		]
		with self.assertRaisesRegex(self.errors.ApiError, "authentication failed"):
			self.client.askQuestion(7, "Question")
		self.log.debugWarning.assert_called()

	def test_invalid_protocol_data_and_timeout(self):
		for opening in ("invalid", "0[]", '0{"sid":null}', '0{"sid":2}'):
			with self.subTest(opening=opening):
				self.queueResponses(opening)
				with self.assertRaisesRegex(self.errors.ApiError, "gateway session"):
					self.client.askQuestion(7, "Question")
		for payload in ("invalid", []):
			self.queueResponses(payload)
			with self.assertRaisesRegex(self.errors.ApiError, "invalid response"):
				self.client._apiRequest("GET", "app-config/user")
		for packet in ("1", "41", '44{"message":"rejected"}'):
			self.mockGateway(authentication=(packet,))
			with self.assertRaisesRegex(self.errors.ApiError, "connection closed"):
				with self.client._gatewaySession():
					self.fail("A rejected gateway must not yield a session")
		self.mockGateway(({},), ('42["NEW_CHAT_MESSAGE", private text',))
		with patch.object(self.clientModule, "DESCRIPTION_TIMEOUT", 0.01):
			with self.assertRaisesRegex(self.errors.ApiError, "timed out"):
				self.client.askQuestion(7, "question")
		self.assertNotIn("private text", str(self.log.mock_calls))
		for dimension in (-1, "invalid"):
			with self.assertRaisesRegex(self.errors.ApiError, "account configuration"):
				self.client._prepareImage(b"image", {"chat_image_max_dimension": dimension})

	def test_cancelled_login_does_not_save_or_touch_closed_dialog(self):
		dialog = object.__new__(self.guiModule.BeMyEyesLoginDialog)
		dialog._isClosing = True
		dialog._cancelEvent = Event()
		dialog._cancelEvent.set()
		with patch.object(self.guiModule, "saveAccount") as save:
			dialog._onLoginSucceeded("late-token", "late-account")
			dialog._onLoginFailed("late error")
			save.assert_not_called()
		with self.assertRaises(self.errors.CancellationError):
			dialog._checkCancelled()

	def test_login_cancel_and_default_close_end_modal_dialog(self):
		for handler in ("_onCancel", "_onClose"):
			with self.subTest(handler=handler):
				dialog = object.__new__(self.guiModule.BeMyEyesLoginDialog)
				dialog._isClosing = False
				dialog._cancelEvent = Event()
				dialog.EndModal = Mock()
				dialog.Close = Mock()
				event = Mock()
				# wx.Dialog's default close handler dispatches the cancel button event.
				event.Skip.side_effect = lambda: dialog._onCancel(Mock())
				getattr(dialog, handler)(event)
				dialog.EndModal.assert_called_once_with(sys.modules["wx"].ID_CANCEL)
				dialog.Close.assert_not_called()
				self.assertTrue(dialog._isClosing)
				self.assertTrue(dialog._cancelEvent.is_set())

	def test_gateway_waits_for_exact_authentication_event_across_polls(self):
		self.mockGateway(
			({},),
			(chatEvent("answer", session=7),),
			authentication=('40{"sid":"socket"}', "2", '42["AUTHENTICATED",{}]'),
		)
		self.assertEqual(self.client.askQuestion(7, "question"), "answer")
		self.assertTrue(any(call.kwargs.get("data") == "3" for call in self.http.call_args_list))
		self.mockGateway(({},), (chatEvent("next answer", session=7),))
		self.assertEqual(self.client.askQuestion(7, "next question"), "next answer")

	def test_gateway_authentication_timeout_does_not_submit_question(self):
		self.mockGateway(authentication=('40{"sid":"socket"}', '42["UNAUTHENTICATED",{}]'))
		with patch.object(self.clientModule, "REQUEST_TIMEOUT", 0.05):
			with self.assertRaisesRegex(self.errors.ApiError, "authentication was not confirmed"):
				self.client.askQuestion(7, "question")
		self.assertFalse(any(call.kwargs["url"].endswith("/messages") for call in self.http.call_args_list))
		self.assertEqual(self.clientModule._pendingQuestions, set())
		self.assertEqual(self.http.call_args.kwargs["data"], "1")

	def test_gateway_answers_heartbeats_while_image_upload_is_blocked(self):
		packets, pong = self.mockGateway(
			(
				{},
				{"id": 7},
				{"url": "https://upload.example.test/", "fields": {}, "chat_image_id": 8},
				"",
				{"images": [{"upload_finished": True}]},
			),
			(chatEvent("description", session=7),),
		)
		send = self.http.side_effect

		def upload(**kwargs):
			if "files" in kwargs:
				packets.put("2")
				self.assertTrue(pong.wait(1), "Gateway must send pong before the upload finishes")
			return send(**kwargs)

		self.http.side_effect = upload
		with patch.object(self.client, "_prepareImage", return_value=(b"image", 40, 20, "png")):
			result = self.client.describeImage(b"image")
		self.assertEqual(result["text"], "description")
		self.assertEqual(self.http.call_args.kwargs["data"], "1")

	def test_uncertain_question_submission_blocks_retry_but_failed_setup_does_not(self):
		self.mockGateway(({},))
		send = self.http.side_effect

		def failSubmission(**kwargs):
			if kwargs["url"].endswith("/messages"):
				raise requests.exceptions.ReadTimeout()
			return send(**kwargs)

		self.http.side_effect = failSubmission
		with self.assertRaises(self.errors.NetworkError):
			self.client.askQuestion(7, "question")
		self.http.reset_mock()
		with self.assertRaisesRegex(self.errors.ApiError, "Recognize the image again"):
			self.clientModule.BeMyEyesClient().askQuestion(7, "another question")
		self.http.assert_not_called()
		self.http.side_effect = requests.exceptions.ReadTimeout()
		with self.assertRaises(self.errors.NetworkError):
			self.client.askQuestion(8, "question")
		self.mockGateway(({},), (chatEvent("answer", session=8),))
		self.assertEqual(self.client.askQuestion(8, "retry"), "answer")

	def test_engine_settings_can_be_discovered_before_config_exists(self):
		self.conf.clear()
		engine = self.engineModule.CustomContentRecognizer()
		self.assertEqual(engine.supportedSettings[0].name, "manageAccount")
		self.assertFalse(engine.supportsAutomaticRecognition)
		self.assertFalse(engine.supportsStreaming)
		with self.assertRaises(self.errors.AuthenticationError):
			engine._getApiResult({"imageContent": b"image"}, Event())
		self.http.assert_not_called()

	def test_standard_engines_still_validate_before_recording_results(self):
		engine = self.engineModule.CustomContentRecognizer()
		self.queueResponses({"text": "standard result"})
		with patch.object(engine, "processApiResult", return_value=False) as validate:
			result = self.handler.BaseRecognizer._getApiResult(
				engine, {"method": "POST", "url": "https://example.test"}, Event()
			)
			validate.assert_called_once()
		self.assertEqual(result["text"], "standard result")
		self.queueResponses({"text": "rejected"})
		with patch.object(engine, "processApiResult", return_value="service error"):
			with self.assertRaisesRegex(self.errors.ApiError, "service error"):
				self.handler.BaseRecognizer._getApiResult(
					engine, {"method": "POST", "url": "https://example.test"}, Event()
				)


if __name__ == "__main__":
	unittest.main()
