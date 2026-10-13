# Copyright (C) 2026 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

from dataclasses import dataclass, field


@dataclass
class EngineSetting:
	"""Represents a basic engine setting."""

	name: str
	displayNameWithAccelerator: str
	configSpec: str = field(init=False, default="string(default=None)")


@dataclass
class ChoiceEngineSetting:
	"""Represents an engine setting with a list of choices."""

	name: str
	displayNameWithAccelerator: str
	optionsPropertyName: str
	configSpec: str = field(init=False, default="string(default=None)")


@dataclass
class EditableChoiceEngineSetting(ChoiceEngineSetting):
	"""Represents an engine setting with suggested choices and manual input."""


@dataclass
class TextInputEngineSetting:
	"""Represents an engine setting for text input, like an API key."""

	name: str
	displayNameWithAccelerator: str
	multiline: bool = False
	refreshSettingsOnChange: bool = False
	configKey: str | None = None
	configSpec: str = field(init=False, default="string(default=None)")


@dataclass
class ReadOnlyEngineSetting:
	"""Represents a read-only engine setting, like API quota."""

	name: str
	displayNameWithAccelerator: str
	configSpec: str = field(init=False, default="string(default=None)")


@dataclass
class NumericEngineSetting:
	"""Represents a numeric engine setting, like image quality."""

	name: str
	displayNameWithAccelerator: str
	minStep: int = 1
	largeStep: int = 10
	minVal: int = field(init=False, default=0)
	maxVal: int = field(init=False, default=100)
	configSpec: str = field(init=False, default="integer(default=50,min=0,max=100)")

	def __post_init__(self):
		self.largeStep = max(self.largeStep, self.minStep)


@dataclass
class BooleanEngineSetting:
	"""Represents a boolean engine setting."""

	name: str
	displayNameWithAccelerator: str
	configSpec: str = field(init=False, default="boolean(default=False)")


@dataclass
class CheckListEngineSetting:
	"""Represents an engine setting for a series of optional features."""

	name: str
	displayNameWithAccelerator: str
	optionsPropertyName: str
	configSpec: str = field(init=False, default="list(default=list())")


@dataclass
class ButtonEngineSetting:
	"""Represents an engine setting changed via a button-activated dialog."""

	name: str
	displayNameWithAccelerator: str
	configSpec: str = field(init=False, default="string(default=None)")
