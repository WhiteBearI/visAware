from __future__ import annotations

import builtins
import ctypes
import gc
import importlib.util
from pathlib import Path
import sys
import types
import unittest
import weakref


def loadRuntimeModule():
	modulePath = (
		Path(__file__).resolve().parents[1]
		/ "addon"
		/ "globalPlugins"
		/ "visAware"
		/ "contentRecognizers"
		/ "_rapidOCRRuntime.py"
	)
	spec = importlib.util.spec_from_file_location("visAware_rapidOCRRuntime", modulePath)
	if spec is None or spec.loader is None:
		raise RuntimeError("Failed to load _rapidOCRRuntime.py")
	module = importlib.util.module_from_spec(spec)
	sys.modules[spec.name] = module
	spec.loader.exec_module(module)
	return module


def loadEngineModule():
	builtins._ = lambda value: value
	for moduleName in (
		"addon",
		"addon.globalPlugins",
		"addon.globalPlugins.visAware",
		"addon.globalPlugins.visAware.contentRecognizers",
	):
		module = types.ModuleType(moduleName)
		module.__path__ = []
		sys.modules[moduleName] = module

	addonHandlerModule = types.ModuleType("addonHandler")
	addonHandlerModule.initTranslation = lambda: None
	sys.modules["addonHandler"] = addonHandlerModule

	contentRecogModule = types.ModuleType("contentRecog")
	contentRecogModule.LinesWordsResult = type("LinesWordsResult", (), {})
	contentRecogModule.RecogImageInfo = type("RecogImageInfo", (), {})

	class SimpleTextResult:
		def __init__(self, text: str):
			self.text = text

	contentRecogModule.SimpleTextResult = SimpleTextResult
	sys.modules["contentRecog"] = contentRecogModule

	recogHistoryModule = types.ModuleType("addon.globalPlugins.visAware.recogHistory")
	recogHistoryModule.createEntry = lambda *args: None
	recogHistoryModule.attachEntry = lambda result, _entry: result
	sys.modules[recogHistoryModule.__name__] = recogHistoryModule

	exceptionsModule = types.ModuleType("addon.globalPlugins.visAware.exceptions")
	exceptionsModule.ApiError = type("ApiError", (Exception,), {})
	sys.modules[exceptionsModule.__name__] = exceptionsModule

	class BaseRecognizer:
		originalImage = None

		def _checkCancelled(self, _event):
			return None

	recogHandlerModule = types.ModuleType("addon.globalPlugins.visAware.recogHandler")
	recogHandlerModule.BaseRecognizer = BaseRecognizer
	recogHandlerModule.RecognitionRequest = type("RecognitionRequest", (), {})
	sys.modules[recogHandlerModule.__name__] = recogHandlerModule

	runtimePath = (
		Path(__file__).resolve().parents[1]
		/ "addon"
		/ "globalPlugins"
		/ "visAware"
		/ "contentRecognizers"
		/ "_rapidOCRRuntime.py"
	)
	runtimeName = "addon.globalPlugins.visAware.contentRecognizers._rapidOCRRuntime"
	runtimeSpec = importlib.util.spec_from_file_location(runtimeName, runtimePath)
	if runtimeSpec is None or runtimeSpec.loader is None:
		raise RuntimeError("Failed to load _rapidOCRRuntime.py")
	runtimeModule = importlib.util.module_from_spec(runtimeSpec)
	sys.modules[runtimeName] = runtimeModule
	runtimeSpec.loader.exec_module(runtimeModule)

	enginePath = runtimePath.with_name("rapidOCR.py")
	engineName = "addon.globalPlugins.visAware.contentRecognizers.rapidOCR"
	engineSpec = importlib.util.spec_from_file_location(engineName, enginePath)
	if engineSpec is None or engineSpec.loader is None:
		raise RuntimeError("Failed to load rapidOCR.py")
	engineModule = importlib.util.module_from_spec(engineSpec)
	sys.modules[engineName] = engineModule
	engineSpec.loader.exec_module(engineModule)
	return engineModule, runtimeModule


class RapidOCRRuntimeTestCase(unittest.TestCase):
	def setUp(self) -> None:
		self.module = loadRuntimeModule()

	def _makeRuntime(self):
		runtime = self.module.RapidOCRRuntime(Path("unused.dll"))
		runtime._library = object()
		runtime._handle = ctypes.c_void_p(1)
		return runtime

	def test_abi_and_native_results_are_copied(self) -> None:
		self.assertEqual(24, ctypes.sizeof(self.module.OCR_PARAM))
		self.assertEqual(
			["padding", "maxSideLen", "boxScoreThresh", "boxThresh", "unClipRatio", "doAngle"],
			[name for name, _ctype in self.module.OCR_PARAM._fields_],
		)

		runtime = self._makeRuntime()
		blockText = "界面文字".encode()

		def getBlockText(_handle, _index, buffer, _length):
			if buffer is not None:
				buffer.value = blockText
			return len(blockText)

		def getBlockBox(_handle, _index, box):
			for index, value in enumerate((-5, 10, 90, 10, 90, 40, -5, 40)):
				box[index] = value
			return 8

		runtime._functions = {
			"OcrDetectMemEx": lambda *_args: 1,
			"OcrGetBlockText": getBlockText,
			"OcrGetBlockBox": getBlockBox,
			"OcrGetBlockScore": lambda *_args: 0.9,
		}

		(block,) = runtime.recognizeText(b"png")
		self.assertEqual("界面文字", block.text)
		self.assertEqual((-5, 10, 90, 10, 90, 40, -5, 40), block.box)
		self.assertAlmostEqual(0.9, block.score)

	def test_initializes_one_handle_with_ocr_and_table_models(self) -> None:
		runtime = self.module.RapidOCRRuntime(Path("unused.dll"))
		runtime._library = object()
		threadCounts = []
		runtime._functions = {
			"OcrInitTableEmbedded": lambda threadCount: threadCounts.append(threadCount) or 1,
			"OcrDestroy": lambda _handle: None,
		}

		firstHandle = runtime._ensureHandle()
		secondHandle = runtime._ensureHandle()

		self.assertIs(firstHandle, secondHandle)
		self.assertEqual(1, len(threadCounts))
		self.assertGreaterEqual(threadCounts[0], 1)
		self.assertLessEqual(threadCounts[0], 4)

	def test_native_handle_is_destroyed_when_a_reloaded_runtime_is_collected(self) -> None:
		destroyedHandles = []
		runtime = self.module.RapidOCRRuntime(Path("unused.dll"))
		runtime._library = object()
		runtime._functions = {
			"OcrInitTableEmbedded": lambda _threadCount: 7,
			"OcrDestroy": lambda handle: destroyedHandles.append(handle.value),
		}
		runtime._ensureHandle()
		finalizer = runtime._handleFinalizer
		assert finalizer is not None
		self.assertFalse(finalizer.atexit)
		runtimeReference = weakref.ref(runtime)

		del runtime
		gc.collect()

		self.assertIsNone(runtimeReference())
		self.assertFalse(finalizer.alive)
		self.assertEqual([7], destroyedHandles)

	def test_availability_check_serializes_library_initialization(self) -> None:
		runtime = self.module.RapidOCRRuntime(Path("unused.dll"))
		lockStates = []
		runtime._ensureLibrary = lambda: lockStates.append(runtime._lock.locked()) or object()

		self.assertTrue(runtime.isAvailable())
		self.assertEqual([True], lockStates)

	def test_cancellation_after_initialization_skips_detection(self) -> None:
		class Cancelled(Exception):
			pass

		runtime = self.module.RapidOCRRuntime(Path("unused.dll"))
		runtime._library = object()
		cancelled = False
		detectionCalls = []

		def initialize(_threadCount):
			nonlocal cancelled
			cancelled = True
			return 1

		def checkCancellation():
			if cancelled:
				raise Cancelled

		runtime._functions = {
			"OcrInitTableEmbedded": initialize,
			"OcrDestroy": lambda _handle: None,
			"OcrDetectMemEx": lambda *_args: detectionCalls.append(True) or 0,
		}

		with self.assertRaises(Cancelled):
			runtime.recognizeText(b"png", checkCancellation)
		self.assertEqual([], detectionCalls)

	def test_table_mode_returns_html_and_plain_text(self) -> None:
		runtime = self._makeRuntime()
		htmlContent = b"<table><tr><td>42</td></tr></table>"
		plainText = b"42"
		selectedModes = []

		def copyResult(data: bytes):
			def getResult(_handle, buffer, _length):
				if buffer is not None:
					buffer.value = data
				return len(data)

			return getResult

		runtime._functions = {
			"OcrSetTableMode": lambda _handle, mode: selectedModes.append(mode),
			"OcrDetectTableMem": lambda *_args: True,
			"OcrGetTableLen": lambda _handle: len(htmlContent) + 1,
			"OcrGetTableResult": copyResult(htmlContent),
			"OcrGetTableOcrText": copyResult(plainText),
			"OcrGetTableStructureScore": lambda _handle: 0.98,
			"OcrGetTableCellCount": lambda _handle: 1,
		}

		result = runtime.recognizeTable(b"png", 1)
		self.assertEqual([1], selectedModes)
		self.assertEqual(htmlContent.decode(), result.htmlContent)
		self.assertEqual("42", result.text)
		self.assertAlmostEqual(0.98, result.structureScore)
		self.assertEqual(1, result.cellCount)
		self.assertEqual(1, result.tableMode)

	def test_auto_table_mode_falls_back_to_slanet_inside_one_locked_call(self) -> None:
		runtime = self._makeRuntime()
		htmlContent = b"<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>"
		selectedModes = []

		def copyResult(_handle, buffer, _length):
			if buffer is not None:
				buffer.value = htmlContent
			return len(htmlContent)

		runtime._functions = {
			"OcrSetTableMode": lambda _handle, mode: selectedModes.append(mode),
			"OcrDetectTableMem": lambda *_args: True,
			"OcrGetTableLen": lambda _handle: len(htmlContent),
			"OcrGetTableResult": copyResult,
			"OcrGetTableOcrText": copyResult,
			"OcrGetTableStructureScore": lambda _handle: 0.95,
			"OcrGetTableCellCount": lambda _handle: 4,
		}

		result = runtime.recognizeTableAuto(b"png", lambda candidate: candidate.tableMode == 0)

		assert result is not None
		self.assertEqual([1, 0], selectedModes)
		self.assertEqual(0, result.tableMode)

		selectedModes.clear()
		result = runtime.recognizeTableAuto(b"png", lambda candidate: candidate.tableMode == 1)

		assert result is not None
		self.assertEqual([1], selectedModes)
		self.assertEqual(1, result.tableMode)

	def test_auto_table_mode_propagates_an_error_when_no_mode_is_reliable(self) -> None:
		runtime = self._makeRuntime()
		unreliableResult = self.module.RapidOCRTableResult(
			"A",
			"<table><tr><td>A</td></tr></table>",
			0.0,
			1,
			1,
		)

		def recognizeMode(_handle, _buffer, _size, _param, tableMode, _checker):
			if tableMode == 1:
				return unreliableResult
			raise self.module.RapidOCRRuntimeError("SLANet failed")

		runtime._recognizeTableLocked = recognizeMode
		with self.assertRaisesRegex(self.module.RapidOCRRuntimeError, "SLANet failed"):
			runtime.recognizeTableAuto(b"png", lambda _candidate: False)

	def test_cancel_checker_interrupts_waiting_for_native_lock(self) -> None:
		class Cancelled(Exception):
			pass

		runtime = self._makeRuntime()
		runtime._lock.acquire()
		try:
			with self.assertRaises(Cancelled):
				runtime.recognizeText(b"png", lambda: (_ for _ in ()).throw(Cancelled()))
		finally:
			runtime._lock.release()

	def test_click_coordinates_are_clamped_to_the_recognized_image(self) -> None:
		engineModule, runtimeModule = loadEngineModule()
		block = runtimeModule.RapidOCRTextBlock(
			"button",
			0.9,
			(-10, 5, 120, 5, 120, 60, -10, 60),
		)
		imageInfo = types.SimpleNamespace(recogWidth=100, recogHeight=50, resizeFactor=1)

		result = engineModule.CustomContentRecognizer._normalizeBlocks([block], imageInfo)

		self.assertEqual(
			[{"text": "button", "x": 0, "y": 5, "width": 100, "height": 45, "score": 0.9}],
			result,
		)

	def test_click_coordinates_use_the_actual_upscaled_image_bounds(self) -> None:
		engineModule, runtimeModule = loadEngineModule()
		block = runtimeModule.RapidOCRTextBlock(
			"small button",
			0.9,
			(25, 5, 45, 5, 45, 40, 25, 40),
		)
		imageInfo = types.SimpleNamespace(recogWidth=20, recogHeight=20, resizeFactor=2.5)

		result = engineModule.CustomContentRecognizer._normalizeBlocks([block], imageInfo)

		self.assertEqual(
			[{"text": "small button", "x": 25, "y": 5, "width": 20, "height": 35, "score": 0.9}],
			result,
		)

	def test_table_validation_rejects_partial_or_single_cell_results(self) -> None:
		engineModule, runtimeModule = loadEngineModule()
		resultType = runtimeModule.RapidOCRTableResult
		validator = engineModule.RapidOCRTableRecognizer._isReliableTableResult

		self.assertFalse(
			validator(resultType("A B C D", "<table><tr><td>A B C D</td></tr></table>", 0.99, 1, 0)),
		)
		self.assertFalse(
			validator(
				resultType(
					"A B C D E F G H",
					"<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>",
					0.0,
					4,
					1,
				),
			),
		)
		self.assertFalse(
			validator(
				resultType(
					"AAAA",
					"<table><tr><td>B</td><td>B</td></tr><tr><td>B</td><td>B</td></tr></table>",
					0.95,
					4,
					0,
				),
			),
		)
		self.assertTrue(
			validator(
				resultType(
					"A B C D",
					"<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>",
					0.95,
					4,
					0,
				),
			),
		)


if __name__ == "__main__":
	unittest.main()
