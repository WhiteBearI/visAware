# Copyright (C) 2026 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Shared DeepSeek vision model presets."""

from collections import OrderedDict

import addonHandler

addonHandler.initTranslation()

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_VISION_MODEL = "deepseek-flash"

_DEEPSEEK_VISION_MODEL_CHOICES = OrderedDict(
	{
		# Translators: The display name for the current DeepSeek vision model.
		"deepseek-flash": _("DeepSeek Flash Vision"),
	},
)


def getDeepSeekVisionModelChoices() -> OrderedDict[str, str]:
	"""Returns supported DeepSeek vision model presets."""
	return _DEEPSEEK_VISION_MODEL_CHOICES.copy()
