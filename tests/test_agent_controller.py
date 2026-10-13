from __future__ import annotations

import ast
import functools
from pathlib import Path
from threading import Event
import types
import unittest
from unittest.mock import Mock


CONTROLLER_PATH = (
	Path(__file__).resolve().parents[1] / "addon" / "globalPlugins" / "visAware" / "agent" / "controller.py"
)


class FakeTimer:
	def __init__(self, callback=None):
		self.stopped = False
		self.callback = callback

	def IsRunning(self):
		return not self.stopped

	def Stop(self):
		self.stopped = True


def loadMultiPressAction():
	tree = ast.parse((CONTROLLER_PATH.parents[1] / "__init__.py").read_text(encoding="utf-8"))
	function = next(
		node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "multiPressAction"
	)
	module = ast.Module(
		body=[ast.ImportFrom(module="__future__", names=[ast.alias("annotations")]), function],
		type_ignores=[],
	)
	timers = []
	wx = types.SimpleNamespace(
		CallLater=lambda _timeout, callback: timers.append(FakeTimer(callback)) or timers[-1],
	)
	namespace = {"functools": functools, "wx": wx}
	exec(compile(ast.fix_missing_locations(module), "__init__.py", "exec"), namespace)
	return namespace["multiPressAction"], timers


def loadController():
	tree = ast.parse(CONTROLLER_PATH.read_text(encoding="utf-8"))
	controllerClass = next(node for node in tree.body if isinstance(node, ast.ClassDef))
	controllerClass.body = [
		node
		for node in controllerClass.body
		if isinstance(node, ast.FunctionDef)
		and node.name
		in {
			"__init__",
			"start",
			"_onAgentDone",
			"stop",
			"askUser",
			"closeQuestionDialog",
			"terminate",
		}
	]
	module = ast.Module(
		body=[ast.ImportFrom(module="__future__", names=[ast.alias("annotations")]), controllerClass],
		type_ignores=[],
	)
	session = Mock()
	namespace = {
		"Event": Event,
		"AgentSession": session,
		"log": Mock(),
		"ui": types.SimpleNamespace(message=Mock()),
		"wx": types.SimpleNamespace(CallAfter=lambda callback, *args: callback(*args)),
		"_": lambda value: value,
	}
	exec(compile(ast.fix_missing_locations(module), str(CONTROLLER_PATH), "exec"), namespace)
	return namespace[controllerClass.name], session


class AgentControllerLifecycleTestCase(unittest.TestCase):
	def test_multi_press_timer_cannot_run_after_termination(self):
		multiPressAction, timers = loadMultiPressAction()
		calls = []

		@multiPressAction()
		def script(plugin, _gesture, pressCount):
			calls.append(pressCount)

		plugin = types.SimpleNamespace(_terminated=False)
		script(plugin, None)
		plugin._terminated = True
		timers[0].callback()

		self.assertEqual(calls, [])

	def test_session_completion_clears_current_session(self):
		controllerClass, sessionClass = loadController()
		controller = controllerClass.__new__(controllerClass)
		controller._terminated = False
		controller._startTimer = None
		controller._agentSession = None
		controller._taskCues = Mock()
		sessionClass.return_value.isRunning = False

		controller.start("goal")
		session = controller._agentSession
		self.assertIsNotNone(session)
		sessionClass.call_args.kwargs["onDone"](session)

		self.assertIsNone(controller._agentSession)

	def test_terminate_cancels_pending_start_and_blocks_new_session(self):
		controllerClass, sessionClass = loadController()
		controller = controllerClass.__new__(controllerClass)
		controller._terminated = False
		timer = FakeTimer()
		controller._startTimer = timer
		controller._agentSession = None
		controller._questionDialog = None

		controller.terminate()
		controller.start("goal")

		self.assertTrue(timer.stopped)
		self.assertIsNone(controller._startTimer)
		self.assertTrue(controller._terminated)
		sessionClass.assert_not_called()

	def test_cancelled_question_does_not_open_a_dialog(self):
		controllerClass, _sessionClass = loadController()
		controller = controllerClass.__new__(controllerClass)
		controller._terminated = False
		controller._questionDialog = None
		cancelEvent = Event()
		cancelEvent.set()

		result = controller.askUser("question", cancelEvent)

		self.assertIsNone(result)
		self.assertIsNone(controller._questionDialog)


if __name__ == "__main__":
	unittest.main()
