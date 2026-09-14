# Copyright (C) 2026 Cary-rowen <manchen_0528@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Small, process-local ctypes wrapper for the bundled RapidOcrEmbed DLL."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
from threading import Lock
from typing import Any
import weakref


class OCR_PARAM(ctypes.Structure):
	"""The six-field parameter structure used by the bundled DLL ABI."""

	_fields_ = [
		("padding", ctypes.c_int),
		("maxSideLen", ctypes.c_int),
		("boxScoreThresh", ctypes.c_float),
		("boxThresh", ctypes.c_float),
		("unClipRatio", ctypes.c_float),
		("doAngle", ctypes.c_int),
	]


@dataclass(frozen=True)
class RapidOCRTextBlock:
	"""A text block copied out of the native handle."""

	text: str
	score: float
	box: tuple[int, int, int, int, int, int, int, int]


@dataclass(frozen=True)
class RapidOCRTableResult:
	"""A table result copied out of the native handle."""

	text: str
	htmlContent: str
	structureScore: float
	cellCount: int
	tableMode: int


class RapidOCRRuntimeError(Exception):
	"""Raised when the bundled DLL cannot load or complete a native call."""


class RapidOCRRuntime:
	"""Owns one lazy DLL/handle pair shared by all RapidOCR engine entries."""

	_MAX_RESULT_BYTES = 16 * 1024 * 1024

	def __init__(self, dllPath: Path) -> None:
		super().__init__()
		self._dllPath = dllPath
		self._library: Any | None = None
		self._functions: dict[str, Any] = {}
		self._handle: ctypes.c_void_p | None = None
		self._handleFinalizer: weakref.finalize[..., RapidOCRRuntime] | None = None
		self._lock = Lock()

	def isAvailable(self) -> bool:
		"""Return whether the DLL exists and exports the required functions."""
		try:
			with self._locked(None):
				self._ensureLibrary()
		except Exception:
			return False
		return True

	def recognizeText(
		self,
		imageContent: bytes,
		cancellationChecker: Callable[[], None] | None = None,
	) -> list[RapidOCRTextBlock]:
		"""Recognize text and copy all block data before releasing the lock."""
		with self._locked(cancellationChecker):
			self._checkCancellation(cancellationChecker)
			handle = self._ensureHandle()
			self._checkCancellation(cancellationChecker)
			param = OCR_PARAM(50, 1024, 0.6, 0.3, 2.0, 0)
			imageBuffer = self._makeImageBuffer(imageContent)
			try:
				count = self._functions["OcrDetectMemEx"](
					handle,
					imageBuffer,
					len(imageContent),
					ctypes.byref(param),
				)
			except (OSError, ValueError) as e:
				raise RapidOCRRuntimeError(f"text detection failed: {e}") from e
			self._checkCancellation(cancellationChecker)
			if count < 0:
				raise RapidOCRRuntimeError("text detection returned an invalid block count")
			blocks: list[RapidOCRTextBlock] = []
			for index in range(count):
				text = self._readBlockText(handle, index)
				box = (ctypes.c_int * 8)()
				try:
					boxLength = self._functions["OcrGetBlockBox"](handle, index, box)
					score = float(self._functions["OcrGetBlockScore"](handle, index))
				except (OSError, ValueError) as e:
					raise RapidOCRRuntimeError(f"text result extraction failed: {e}") from e
				if boxLength != 8:
					continue
				blocks.append(RapidOCRTextBlock(text, score, tuple(box)))
			return blocks

	def recognizeTable(
		self,
		imageContent: bytes,
		tableMode: int,
		cancellationChecker: Callable[[], None] | None = None,
	) -> RapidOCRTableResult:
		"""Recognize a table and copy its HTML and text while locked."""
		if tableMode not in (0, 1):
			raise ValueError(f"unsupported table mode: {tableMode}")
		with self._locked(cancellationChecker):
			self._checkCancellation(cancellationChecker)
			handle = self._ensureHandle()
			self._checkCancellation(cancellationChecker)
			param = OCR_PARAM(50, 1024, 0.6, 0.3, 2.0, 0)
			imageBuffer = self._makeImageBuffer(imageContent)
			return self._recognizeTableLocked(
				handle,
				imageBuffer,
				len(imageContent),
				param,
				tableMode,
				cancellationChecker,
			)

	def recognizeTableAuto(
		self,
		imageContent: bytes,
		resultChecker: Callable[[RapidOCRTableResult], bool],
		cancellationChecker: Callable[[], None] | None = None,
	) -> RapidOCRTableResult | None:
		"""Try bordered then SLANet table recognition as one atomic native operation."""
		with self._locked(cancellationChecker):
			self._checkCancellation(cancellationChecker)
			handle = self._ensureHandle()
			self._checkCancellation(cancellationChecker)
			param = OCR_PARAM(50, 1024, 0.6, 0.3, 2.0, 0)
			imageBuffer = self._makeImageBuffer(imageContent)
			lastError: RapidOCRRuntimeError | None = None
			for tableMode in (1, 0):
				try:
					result = self._recognizeTableLocked(
						handle,
						imageBuffer,
						len(imageContent),
						param,
						tableMode,
						cancellationChecker,
					)
				except RapidOCRRuntimeError as e:
					lastError = e
					self._checkCancellation(cancellationChecker)
					continue
				if resultChecker(result):
					return result
				self._checkCancellation(cancellationChecker)
			if lastError:
				raise lastError
			return None

	def _recognizeTableLocked(
		self,
		handle: ctypes.c_void_p,
		imageBuffer: Any,
		imageSize: int,
		param: OCR_PARAM,
		tableMode: int,
		cancellationChecker: Callable[[], None] | None,
	) -> RapidOCRTableResult:
		"""Run one table mode while the caller owns the native handle lock."""
		try:
			self._functions["OcrSetTableMode"](handle, tableMode)
			succeeded = self._functions["OcrDetectTableMem"](
				handle,
				imageBuffer,
				imageSize,
				ctypes.byref(param),
			)
		except (OSError, ValueError) as e:
			raise RapidOCRRuntimeError(f"table detection failed: {e}") from e
		self._checkCancellation(cancellationChecker)
		if not succeeded:
			raise RapidOCRRuntimeError("table detection returned no result")
		try:
			htmlLength = self._functions["OcrGetTableLen"](handle)
			htmlContent = self._readString(
				handle,
				"OcrGetTableResult",
				htmlLength,
			)
			textLength = self._functions["OcrGetTableOcrText"](handle, None, 0)
			text = self._readString(handle, "OcrGetTableOcrText", textLength)
			structureScore = float(self._functions["OcrGetTableStructureScore"](handle))
			cellCount = self._functions["OcrGetTableCellCount"](handle)
		except (OSError, ValueError) as e:
			raise RapidOCRRuntimeError(f"table result extraction failed: {e}") from e
		if cellCount < 0:
			raise RapidOCRRuntimeError("table detection returned an invalid cell count")
		return RapidOCRTableResult(text, htmlContent, structureScore, cellCount, tableMode)

	@contextlib.contextmanager
	def _locked(self, cancellationChecker: Callable[[], None] | None) -> Iterator[None]:
		"""Acquire the global native lock without making cancellation unresponsive."""
		# ponytail: global lock; split handles only if profiling proves serialization is insufficient.
		while not self._lock.acquire(timeout=0.05):
			self._checkCancellation(cancellationChecker)
		try:
			yield
		finally:
			self._lock.release()

	def _ensureLibrary(self) -> Any:
		if self._library is not None:
			return self._library
		if not self._dllPath.is_file():
			raise RapidOCRRuntimeError(f"DLL not found: {self._dllPath}")
		try:
			library = ctypes.CDLL(str(self._dllPath))
		except OSError as e:
			raise RapidOCRRuntimeError(f"DLL could not be loaded: {e}") from e
		self._library = library
		try:
			self._bind("OcrInitTableEmbedded", [ctypes.c_int], ctypes.c_void_p)
			self._bind("OcrDestroy", [ctypes.c_void_p], None)
			self._bind(
				"OcrDetectMemEx",
				[ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.POINTER(OCR_PARAM)],
				ctypes.c_int,
			)
			self._bind(
				"OcrGetBlockText",
				[ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int],
				ctypes.c_int,
			)
			self._bind("OcrGetBlockScore", [ctypes.c_void_p, ctypes.c_int], ctypes.c_float)
			self._bind(
				"OcrGetBlockBox",
				[ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)],
				ctypes.c_int,
			)
			self._bind("OcrSetTableMode", [ctypes.c_void_p, ctypes.c_int], None)
			self._bind(
				"OcrDetectTableMem",
				[ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.POINTER(OCR_PARAM)],
				ctypes.c_bool,
			)
			self._bind("OcrGetTableLen", [ctypes.c_void_p], ctypes.c_int)
			self._bind("OcrGetTableResult", [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int], ctypes.c_int)
			self._bind("OcrGetTableStructureScore", [ctypes.c_void_p], ctypes.c_float)
			self._bind("OcrGetTableOcrText", [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int], ctypes.c_int)
			self._bind("OcrGetTableCellCount", [ctypes.c_void_p], ctypes.c_int)
		except Exception:
			self._library = None
			self._functions.clear()
			raise
		return library

	def _ensureHandle(self) -> ctypes.c_void_p:
		if self._handle is not None:
			return self._handle
		self._ensureLibrary()
		try:
			handle = self._functions["OcrInitTableEmbedded"](min(max(1, os.cpu_count() or 1), 4))
		except (OSError, ValueError) as e:
			raise RapidOCRRuntimeError(f"DLL initialization failed: {e}") from e
		if not handle:
			raise RapidOCRRuntimeError("DLL initialization returned an empty handle")
		self._handle = ctypes.c_void_p(handle)
		# Active bound calls keep this runtime alive, so reload cleanup cannot race native inference.
		self._handleFinalizer = weakref.finalize(
			self,
			self._functions["OcrDestroy"],
			self._handle,
		)
		# Process teardown reclaims the DLL; only run the finalizer for module reloads.
		self._handleFinalizer.atexit = False
		return self._handle

	def _bind(self, name: str, argtypes: list[Any], restype: Any) -> None:
		try:
			function = getattr(self._library, name)
		except AttributeError as e:
			raise RapidOCRRuntimeError(f"DLL export missing: {name}") from e
		function.argtypes = argtypes
		function.restype = restype
		self._functions[name] = function

	@staticmethod
	def _makeImageBuffer(imageContent: bytes) -> Any:
		if not imageContent:
			raise RapidOCRRuntimeError("image content is empty")
		return (ctypes.c_ubyte * len(imageContent)).from_buffer_copy(imageContent)

	def _readBlockText(self, handle: ctypes.c_void_p, index: int) -> str:
		try:
			length = self._functions["OcrGetBlockText"](handle, index, None, 0)
		except (OSError, ValueError) as e:
			raise RapidOCRRuntimeError(f"text result length failed: {e}") from e
		return self._readString(handle, "OcrGetBlockText", length, index=index)

	def _readString(
		self,
		handle: ctypes.c_void_p,
		functionName: str,
		length: int,
		index: int | None = None,
	) -> str:
		if length < 0 or length > self._MAX_RESULT_BYTES:
			raise RapidOCRRuntimeError(f"invalid result length: {length}")
		buffer = ctypes.create_string_buffer(length + 1)
		args: tuple[Any, ...]
		if index is None:
			args = (handle, buffer, len(buffer))
		else:
			args = (handle, index, buffer, len(buffer))
		try:
			self._functions[functionName](*args)
		except (OSError, ValueError) as e:
			raise RapidOCRRuntimeError(f"result extraction failed: {e}") from e
		return buffer.value.decode("utf-8", errors="replace")

	@staticmethod
	def _checkCancellation(cancellationChecker: Callable[[], None] | None) -> None:
		if cancellationChecker:
			cancellationChecker()


_runtime = RapidOCRRuntime(Path(__file__).with_name("RapidOcrOnnx_small.dll"))


def getRapidOCRRuntime() -> RapidOCRRuntime:
	"""Return the process-local RapidOCR runtime."""
	return _runtime
