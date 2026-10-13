import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


@pytest.fixture
def parser(monkeypatch):
	monkeypatch.setitem(
		sys.modules, "logHandler", SimpleNamespace(log=SimpleNamespace(debugWarning=lambda *args: None))
	)
	path = (
		Path(__file__).resolve().parents[1]
		/ "addon/globalPlugins/visAware/contentRecognizers/_paddleOCRResults.py"
	)
	spec = importlib.util.spec_from_file_location("_paddle_results_test", path)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module.PaddleOCRResultParser


def test_layout_markdown_has_priority_over_duplicate_ocr_text(parser):
	result = {
		"result": {
			"layoutParsingResults": [
				{
					"markdown": {"text": "# Title\n\nDocument"},
					"prunedResult": {"overall_ocr_res": {"rec_texts": ["Title", "Document"]}},
				}
			]
		}
	}
	assert parser.extractText(result) == "# Title\n\nDocument"


def test_coordinate_formats_normalization_and_duplicate_boxes(parser):
	result = {
		"items": [
			{"text": "A", "bbox": [30, 40, 10, 20]},
			{"text": "A", "bbox": [10, 20, 30, 40]},
			{"rec_texts": ["B", "unboxed"], "rec_polys": [[[50, 60], [70, 60], [70, 80], [50, 80]]]},
			{"text": "C", "location": {"left": 90, "top": 100, "width": 20, "height": 10}},
			{"text": "zero", "bbox": [0, 0, 0, 0]},
		]
	}
	assert parser.toLineResult(result) == [
		[{"text": "A", "x": 10, "y": 20, "width": 20, "height": 20}],
		[{"text": "B", "x": 50, "y": 60, "width": 20, "height": 20}],
		[{"text": "C", "x": 90, "y": 100, "width": 20, "height": 10}],
	]


def test_unknown_response_fallback_skips_embedded_images(parser):
	result = {
		"pages": [{"content": " text "}],
		"outputImages": [{"text": "image metadata"}],
		"image": "data:image/png;base64,AAAA",
	}
	assert parser.extractText(result) == "text"
	assert parser.toLineResult(result) == []
