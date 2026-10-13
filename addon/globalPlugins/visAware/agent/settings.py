# Copyright (C) 2026 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

import addonHandler

from ..engineSettingsUI import AbstractEngineSettingsPanel
from .engine import AgentHandler

addonHandler.initTranslation()


class AgentPanel(AbstractEngineSettingsPanel):
	"""Settings panel for computer-use agent engines."""

	# Translators: The title of the Agent settings panel.
	title = _("AI Agent")
	handler = AgentHandler
