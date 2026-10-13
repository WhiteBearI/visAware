# Copyright (C) 2026 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

from typing import Any

from logHandler import log

TEXT_KEYS = ("block_content", "content", "text", "rec_text", "words")
BOX_KEYS = (
	"bbox",
	"box",
	"block_bbox",
	"rec_box",
	"rec_bbox",
	"poly",
	"polygon",
	"points",
	"dt_poly",
	"dt_polys",
	"location",
)
BOX_LIST_KEYS = ("rec_boxes", "rec_polys", "dt_polys", "polys", "boxes")
IMAGE_FIELD_NAMES = {
	"image",
	"inputImage",
	"ocrImage",
	"layoutImage",
	"outputImage",
	"outputImages",
	"pageImage",
	"pageImages",
}


class PaddleOCRResultParser:
	"""Extracts plain text and coordinate results from PaddleOCR response variants."""

	@classmethod
	def extractText(cls, apiResult: dict[str, Any]) -> str:
		"""
		Extracts plain text from PaddleOCR results.

		:param apiResult: A decoded PaddleOCR result dictionary.
		:returns: Plain text in reading order where possible.
		"""
		knownTextParts = cls._extractKnownTextParts(apiResult)
		if knownTextParts:
			return "\n".join(cls._cleanTextParts(knownTextParts))
		fallbackParts: list[str] = []
		cls._collectGenericText(apiResult, fallbackParts)
		return "\n".join(cls._cleanTextParts(fallbackParts))

	@classmethod
	def toLineResult(cls, apiResult: dict[str, Any]) -> list[list[dict[str, Any]]]:
		"""
		Converts a PaddleOCR result into NVDA's line/word result format.

		:param apiResult: A decoded PaddleOCR result dictionary.
		:returns: A list of lines containing clickable OCR words or blocks.
		"""
		lines: list[list[dict[str, Any]]] = []
		seen: set[tuple[str, int, int, int, int]] = set()
		for text, box in cls._iterTextBoxes(apiResult):
			word = cls._makeWord(text, box)
			if not word:
				continue
			signature = (
				word["text"],
				word["x"],
				word["y"],
				word["width"],
				word["height"],
			)
			if signature in seen:
				continue
			seen.add(signature)
			lines.append([word])
		if lines:
			return lines
		return []

	@classmethod
	def _extractKnownTextParts(cls, apiResult: dict[str, Any]) -> list[str]:
		parts: list[str] = []
		result = cls._getResultContainer(apiResult)
		parts.extend(cls._extractOCRTextParts(result))
		for item in cls._iterResultItems(result, "layoutParsingResults"):
			markdownText = cls._extractMarkdownText(item)
			if markdownText:
				parts.append(markdownText)
				continue
			prunedResult = cls._getPrunedResult(item)
			parts.extend(cls._extractParsingTextParts(prunedResult))
			parts.extend(cls._extractOCRTextParts(prunedResult.get("overall_ocr_res")))
		for item in cls._iterResultItems(result, "ocrResults"):
			parts.extend(cls._extractOCRTextParts(cls._getPrunedResult(item)))
		for item in cls._iterResultItems(result, "results"):
			prunedResult = cls._getPrunedResult(item)
			parts.extend(cls._extractOCRTextParts(prunedResult))
			parts.extend(cls._extractParsingTextParts(prunedResult))
		return parts

	@classmethod
	def _getResultContainer(cls, apiResult: dict[str, Any]) -> dict[str, Any]:
		result = apiResult.get("result")
		return result if isinstance(result, dict) else apiResult

	@classmethod
	def _iterResultItems(cls, result: dict[str, Any], key: str) -> list[dict[str, Any]]:
		items = result.get(key)
		if isinstance(items, dict):
			items = [items]
		if not isinstance(items, list):
			return []
		resultItems: list[dict[str, Any]] = []
		for item in items:
			if not isinstance(item, dict):
				continue
			resultItems.append(item)
		return resultItems

	@staticmethod
	def _getPrunedResult(item: dict[str, Any]) -> dict[str, Any]:
		prunedResult = item.get("prunedResult")
		return prunedResult if isinstance(prunedResult, dict) else item

	@staticmethod
	def _extractMarkdownText(item: dict[str, Any]) -> str:
		markdown = item.get("markdown") or item.get("md")
		if isinstance(markdown, str):
			return markdown
		if not isinstance(markdown, dict):
			return ""
		for key in ("text", "markdown", "content"):
			value = markdown.get(key)
			if isinstance(value, str) and value.strip():
				return value
		return ""

	@classmethod
	def _extractParsingTextParts(cls, item: dict[str, Any]) -> list[str]:
		parsingItems = item.get("parsing_res_list") or item.get("parsingResults")
		if not isinstance(parsingItems, list):
			return []
		parts: list[str] = []
		for parsingItem in parsingItems:
			if not isinstance(parsingItem, dict):
				continue
			text = cls._extractTextFromDict(parsingItem)
			if text:
				parts.append(text)
		return parts

	@staticmethod
	def _extractOCRTextParts(item: Any) -> list[str]:
		if not isinstance(item, dict):
			return []
		texts = item.get("rec_texts") or item.get("texts")
		if not isinstance(texts, list):
			return []
		return [str(text) for text in texts if str(text).strip()]

	@classmethod
	def _collectGenericText(cls, value: Any, parts: list[str]) -> None:
		if isinstance(value, dict):
			text = cls._extractTextFromDict(value)
			if text:
				parts.append(text)
			for key, child in value.items():
				if key in IMAGE_FIELD_NAMES:
					continue
				cls._collectGenericText(child, parts)
		elif isinstance(value, list):
			for child in value:
				cls._collectGenericText(child, parts)

	@classmethod
	def _iterTextBoxes(cls, apiResult: dict[str, Any]) -> list[tuple[str, Any]]:
		items: list[tuple[str, Any]] = []
		for item in cls._iterDictionaries(apiResult):
			texts = item.get("rec_texts") or item.get("texts")
			if isinstance(texts, list):
				boxes = cls._getBoxList(item)
				for index, text in enumerate(texts):
					if index >= len(boxes):
						break
					textString = str(text).strip()
					if textString:
						items.append((textString, boxes[index]))
			text = cls._extractTextFromDict(item)
			box = cls._extractBoxFromDict(item)
			if text and box is not None:
				items.append((text, box))
		return items

	@classmethod
	def _iterDictionaries(cls, value: Any) -> list[dict[str, Any]]:
		items: list[dict[str, Any]] = []
		if isinstance(value, dict):
			items.append(value)
			for key, child in value.items():
				if key in IMAGE_FIELD_NAMES:
					continue
				items.extend(cls._iterDictionaries(child))
		elif isinstance(value, list):
			for child in value:
				items.extend(cls._iterDictionaries(child))
		return items

	@staticmethod
	def _getBoxList(item: dict[str, Any]) -> list[Any]:
		for key in BOX_LIST_KEYS:
			value = item.get(key)
			if isinstance(value, list):
				return value
		return []

	@staticmethod
	def _extractTextFromDict(item: dict[str, Any]) -> str:
		for key in TEXT_KEYS:
			value = item.get(key)
			if isinstance(value, str):
				return value.strip()
			if isinstance(value, list) and all(isinstance(part, str) for part in value):
				return "\n".join(part.strip() for part in value if part.strip())
		return ""

	@staticmethod
	def _extractBoxFromDict(item: dict[str, Any]) -> Any:
		for key in BOX_KEYS:
			value = item.get(key)
			if value is not None:
				return value
		return None

	@classmethod
	def _makeWord(cls, text: str, box: Any) -> dict[str, Any] | None:
		rect = cls._boxToRect(box)
		if not rect:
			return None
		left, top, right, bottom = rect
		width = right - left
		height = bottom - top
		if width <= 0 or height <= 0:
			log.debugWarning(f"PaddleOCR skipped a degenerate box: {box!r}")
			return None
		return {
			"text": text.strip(),
			"x": left,
			"y": top,
			"width": width,
			"height": height,
		}

	@classmethod
	def _boxToRect(cls, box: Any) -> tuple[int, int, int, int] | None:
		if isinstance(box, dict):
			return cls._dictBoxToRect(box)
		if not isinstance(box, list):
			return None
		if len(box) == 4 and all(cls._isNumber(value) for value in box):
			left, top, right, bottom = [round(float(value)) for value in box]
			return cls._normalizeRect(left, top, right, bottom)
		if len(box) == 8 and all(cls._isNumber(value) for value in box):
			points = [[box[index], box[index + 1]] for index in range(0, len(box), 2)]
			return cls._pointsToRect(points)
		if all(cls._isPoint(value) for value in box):
			return cls._pointsToRect(box)
		return None

	@classmethod
	def _dictBoxToRect(cls, box: dict[str, Any]) -> tuple[int, int, int, int] | None:
		if {"left", "top", "width", "height"}.issubset(box):
			left = round(float(box["left"]))
			top = round(float(box["top"]))
			right = left + round(float(box["width"]))
			bottom = top + round(float(box["height"]))
			return cls._normalizeRect(left, top, right, bottom)
		if {"x", "y", "width", "height"}.issubset(box):
			left = round(float(box["x"]))
			top = round(float(box["y"]))
			right = left + round(float(box["width"]))
			bottom = top + round(float(box["height"]))
			return cls._normalizeRect(left, top, right, bottom)
		for key in ("points", "poly", "polygon", "vertices"):
			value = box.get(key)
			if isinstance(value, list):
				return cls._boxToRect(value)
		return None

	@classmethod
	def _pointsToRect(cls, points: list[Any]) -> tuple[int, int, int, int] | None:
		xValues: list[float] = []
		yValues: list[float] = []
		for point in points:
			if not cls._isPoint(point):
				return None
			xValues.append(float(point[0]))
			yValues.append(float(point[1]))
		if not xValues or not yValues:
			return None
		return cls._normalizeRect(
			round(min(xValues)),
			round(min(yValues)),
			round(max(xValues)),
			round(max(yValues)),
		)

	@staticmethod
	def _normalizeRect(left: int, top: int, right: int, bottom: int) -> tuple[int, int, int, int]:
		if right < left:
			left, right = right, left
		if bottom < top:
			top, bottom = bottom, top
		return left, top, right, bottom

	@staticmethod
	def _isNumber(value: Any) -> bool:
		return isinstance(value, (int, float)) or (
			isinstance(value, str) and value.replace(".", "", 1).replace("-", "", 1).isdigit()
		)

	@classmethod
	def _isPoint(cls, value: Any) -> bool:
		return (
			isinstance(value, list)
			and len(value) >= 2
			and cls._isNumber(value[0])
			and cls._isNumber(value[1])
		)

	@staticmethod
	def _cleanTextParts(parts: list[str]) -> list[str]:
		cleanedParts: list[str] = []
		for part in parts:
			part = str(part).strip()
			if not part or part.startswith("data:image/"):
				continue
			cleanedParts.append(part)
		return cleanedParts
