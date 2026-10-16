# Copyright (C) 2026 Cary-rowen <manchen_0528@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Offline PP-OCRv6 Small engines backed by the bundled RapidOcrEmbed DLL."""

from __future__ import annotations

import addonHandler
from collections import Counter
from contentRecog import LinesWordsResult, RecogImageInfo, SimpleTextResult
from html.parser import HTMLParser
from threading import Event
from typing import Any, cast, override

from .. import recogHistory
from ..exceptions import ApiError
from ..recogHandler import BaseRecognizer, RecognitionRequest
from ._rapidOCRRuntime import (
	RapidOCRRuntimeError,
	RapidOCRTableResult,
	RapidOCRTextBlock,
	getRapidOCRRuntime,
)

addonHandler.initTranslation()


class RapidOCRTableTextResult(SimpleTextResult):
	"""Plain table text paired with HTML for accessible structured display."""

	htmlContent: str

	def __init__(self, text: str, htmlContent: str) -> None:
		super().__init__(text)
		self.htmlContent = htmlContent


class _TableShapeParser(HTMLParser):
	"""Collect the small amount of structure needed to reject false table results."""

	def __init__(self) -> None:
		super().__init__(convert_charrefs=True)
		self.rowCellCounts: list[int] = []
		self.cellTexts: list[str] = []
		self._currentRowCells: int | None = None
		self._currentCellParts: list[str] | None = None

	@override
	def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
		tag = tag.lower()
		if tag == "tr":
			self._currentRowCells = 0
		elif tag in ("td", "th") and self._currentRowCells is not None:
			self._currentRowCells += 1
			self._currentCellParts = []

	@override
	def handle_endtag(self, tag: str) -> None:
		tag = tag.lower()
		if tag in ("td", "th") and self._currentCellParts is not None:
			self.cellTexts.append("".join(self._currentCellParts).strip())
			self._currentCellParts = None
		elif tag == "tr" and self._currentRowCells is not None:
			if self._currentRowCells:
				self.rowCellCounts.append(self._currentRowCells)
			self._currentRowCells = None

	@override
	def handle_data(self, data: str) -> None:
		if self._currentCellParts is not None:
			self._currentCellParts.append(data)


class _RapidOCRRecognizer(BaseRecognizer):
	"""Shared adapter for the text and automatic-table RapidOCR entries."""

	supportsAutomaticRecognition = True
	uploadBase64EncodeImage = False
	uploadImageFormat = "PNG"
	recognizesTables = False

	@property
	@override
	def supportedSettings(self) -> list[Any]:
		"""Return no settings because the bundled engine uses fixed parameters."""
		return []

	@classmethod
	@override
	def check(cls) -> bool:
		"""Return whether the bundled DLL can be loaded and has the required ABI."""
		return getRapidOCRRuntime().isAvailable()

	@override
	def _buildRequestParams(self, imageContent: bytes, request: RecognitionRequest) -> dict[str, bytes]:
		return {"imageContent": imageContent}

	@override
	def processApiResult(self, result: bytes) -> str | bool:
		return False

	@override
	def extractText(self, apiResult: dict[str, Any]) -> str:
		text = apiResult.get("text", "")
		return text if isinstance(text, str) else ""

	@override
	def _convertToLineResultFormat(self, apiResult: dict[str, Any]) -> list[list[dict[str, str | int]]]:
		blocksValue = apiResult.get("blocks", [])
		if not isinstance(blocksValue, list):
			return []
		blocks = cast(list[Any], blocksValue)
		lines: list[list[dict[str, str | int]]] = []
		for blockValue in blocks:
			if not isinstance(blockValue, dict):
				continue
			block = cast(dict[str, Any], blockValue)
			text = block.get("text")
			if not isinstance(text, str) or not text.strip():
				continue
			try:
				x = int(block["x"])
				y = int(block["y"])
				width = int(block["width"])
				height = int(block["height"])
				word = {
					"text": text,
					"x": x,
					"y": y,
					"width": width,
					"height": height,
				}
			except (KeyError, TypeError, ValueError):
				continue
			if width > 0 and height > 0:
				lines.append([word])
		return lines

	@override
	def _handleStandardResponse(
		self,
		requestParams: dict[str, Any],
		imageInfo: RecogImageInfo,
		cancellationEvent: Event,
		request: RecognitionRequest,
	) -> Any:
		imageContent = requestParams.get("imageContent")
		if not isinstance(imageContent, bytes):
			# Translators: An error shown when RapidOCR receives invalid image bytes.
			raise ApiError(_("RapidOCR received invalid image content."))
		try:
			if self.recognizesTables:
				return self._recognizeTable(imageContent, cancellationEvent)
			return self._recognizeText(imageContent, imageInfo, cancellationEvent, request)
		except RapidOCRRuntimeError as e:
			# Translators: A low-level error reported by the bundled RapidOCR DLL.
			raise ApiError(_("RapidOCR failed: {}").format(e)) from e

	def _recognizeText(
		self,
		imageContent: bytes,
		imageInfo: RecogImageInfo,
		cancellationEvent: Event,
		request: RecognitionRequest,
	) -> Any:
		blocks = getRapidOCRRuntime().recognizeText(
			imageContent,
			lambda: self._checkCancelled(cancellationEvent),
		)
		response = {
			"text": "\n".join(block.text for block in blocks if block.text.strip()),
			"blocks": self._normalizeBlocks(blocks, imageInfo),
		}
		self._checkCancelled(cancellationEvent)
		historyEntry = self._makeHistoryEntry(response)
		if request.textResult:
			text = self.extractText(response)
			if not text.strip():
				# Translators: An error message for a blank recognition result.
				raise ApiError(_("Recognition result is blank."))
			return recogHistory.attachEntry(SimpleTextResult(text), historyEntry)
		lineResult = self._convertToLineResultFormat(response)
		if not lineResult:
			# Translators: An error message for a blank recognition result.
			raise ApiError(_("Recognition result is blank."))
		return recogHistory.attachEntry(LinesWordsResult(lineResult, imageInfo), historyEntry)

	def _recognizeTable(self, imageContent: bytes, cancellationEvent: Event) -> RapidOCRTableTextResult:
		tableResult = getRapidOCRRuntime().recognizeTableAuto(
			imageContent,
			self._isReliableTableResult,
			lambda: self._checkCancelled(cancellationEvent),
		)
		self._checkCancelled(cancellationEvent)
		if tableResult is None:
			# Translators: An error shown when neither RapidOCR table mode found a reliable table.
			raise ApiError(_("No reliable table structure was detected."))
		response = {
			"text": tableResult.text,
			"html": tableResult.htmlContent,
			"structureScore": tableResult.structureScore,
			"cellCount": tableResult.cellCount,
			"tableMode": "bordered" if tableResult.tableMode == 1 else "slanet",
		}
		result = RapidOCRTableTextResult(tableResult.text, tableResult.htmlContent)
		return recogHistory.attachEntry(result, self._makeHistoryEntry(response))

	@staticmethod
	def _isReliableTableResult(tableResult: RapidOCRTableResult) -> bool:
		if tableResult.cellCount < 4 or not tableResult.htmlContent.strip():
			return False
		parser = _TableShapeParser()
		try:
			parser.feed(tableResult.htmlContent)
			parser.close()
		except Exception:
			return False
		if len(parser.rowCellCounts) < 2 or max(parser.rowCellCounts, default=0) < 2:
			return False
		nonEmptyCellTexts = [text for text in parser.cellTexts if text.strip()]
		if len(nonEmptyCellTexts) < 3:
			return False
		ocrText = "".join(tableResult.text.split()).casefold()
		cellText = "".join("".join(nonEmptyCellTexts).split()).casefold()
		if not ocrText:
			return False
		matchingCharacterCount = sum((Counter(cellText) & Counter(ocrText)).values())
		if matchingCharacterCount / len(ocrText) < 0.7:
			return False
		return tableResult.tableMode == 1 or tableResult.structureScore >= 0.5

	def _makeHistoryEntry(self, response: dict[str, Any]) -> recogHistory.HistoryEntryPayload | None:
		originalImage: object | None = getattr(self, "originalImage", None)
		return recogHistory.createEntry(self, originalImage, response) if originalImage else None

	@staticmethod
	def _normalizeBlocks(
		blocks: list[RapidOCRTextBlock],
		imageInfo: RecogImageInfo,
	) -> list[dict[str, str | int | float]]:
		result: list[dict[str, str | int | float]] = []
		recognizedWidth = int(imageInfo.recogWidth * imageInfo.resizeFactor)
		recognizedHeight = int(imageInfo.recogHeight * imageInfo.resizeFactor)
		for block in blocks:
			if not block.text.strip():
				continue
			xCoordinates = block.box[0::2]
			yCoordinates = block.box[1::2]
			left = max(0, min(min(xCoordinates), recognizedWidth))
			top = max(0, min(min(yCoordinates), recognizedHeight))
			right = max(0, min(max(xCoordinates), recognizedWidth))
			bottom = max(0, min(max(yCoordinates), recognizedHeight))
			if right <= left or bottom <= top:
				continue
			result.append(
				{
					"text": block.text,
					"x": left,
					"y": top,
					"width": right - left,
					"height": bottom - top,
					"score": block.score,
				},
			)
		return result


class CustomContentRecognizer(_RapidOCRRecognizer):
	"""Offline general text recognition with screen coordinates."""

	name = "rapidOCR"
	# Translators: The name of the bundled offline RapidOCR text engine.
	description = _("RapidOCR Text")


class RapidOCRTableRecognizer(_RapidOCRRecognizer):
	"""Offline table recognition with bordered-to-SLANet fallback."""

	name = "rapidOCRTable"
	recognizesTables = True
	# Translators: The name of the bundled RapidOCR table engine.
	description = _("RapidOCR Table")
