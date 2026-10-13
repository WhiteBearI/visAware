# Copyright (C) 2025 Cary-rowen <cary-rowen@outlook.com>
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Abstract base classes for engine handlers and engines."""

from abc import ABC, abstractmethod
import addonHandler
import baseObject
import config
import importlib
import inspect
import pkgutil
from configobj.validate import is_boolean, is_integer, is_list
from logHandler import log
from synthDriverHandler import StringParameterInfo
from typing import List, Tuple, Optional, Any, Type
import types
from .engineSettings import EngineSetting, ButtonEngineSetting

addonHandler.initTranslation()


def _getSettingConfigKey(setting: EngineSetting) -> str:
	"""Returns the persisted configuration key for an engine setting."""
	return getattr(setting, "configKey", None) or setting.name


class AbstractEngineHandler(baseObject.AutoPropertyObject):
	"""
	Manages the lifecycle and configuration of a collection of engines of a specific type.

	This class is responsible for discovering, loading, and switching between different
	engine implementations (e.g., OCR engines, image describer engines).
	"""

	engineList: Optional[List[Tuple[str, str]]] = None
	enginePackageName: str
	enginePackage: types.ModuleType
	configSectionName: str
	engineClass: Type["AbstractEngine"]
	isInitialized: bool = False
	currentEngine: Optional["AbstractEngine"] = None
	mandatoryClassName: Optional[str] = None
	engineClassList: Optional[List[Type["AbstractEngine"]]] = None
	defaultEnginePriorityList: List[str] = ["empty"]
	configSpec: dict = {
		"engine": "string(default=auto)",
		"disabledEngines": "list(default=list())",
	}

	@classmethod
	def _getSafeEngineName(cls, requestedName: str) -> str:
		"""
		Validates the requested engine name against the list of available engines.

		:param requestedName: The name of the engine to validate.
		:returns: The validated name, or "auto" if the name is not found.
		"""
		availableEngineNames = [e[0] for e in cls.getEngineList()]
		return requestedName if requestedName in availableEngineNames else "auto"

	@classmethod
	def initialize(cls) -> None:
		"""
		Initializes the engine handler.

		This method sets up the Python path, registers for config changes,
		discovers available engines, initializes configuration, and loads the
		configured engine.
		"""
		addonHandler.packaging.addDirsToPythonPackagePath(cls.enginePackage)
		config.post_configProfileSwitch.register(cls.handlePostConfigProfileSwitch)
		cls.getAllEngineList()
		cls._initConfig()
		engineNameFromConfig = config.conf[cls.configSectionName]["engine"]
		safeEngineName = cls._getSafeEngineName(engineNameFromConfig)
		cls.setCurrentEngine(safeEngineName)
		cls.isInitialized = True

	@classmethod
	def terminate(cls) -> None:
		"""
		Terminates the engine handler and the current engine.

		This unregisters from config changes and ensures the current engine is
		properly shut down.
		"""
		config.post_configProfileSwitch.unregister(cls.handlePostConfigProfileSwitch)
		if cls.currentEngine:
			cls.currentEngine.terminate()
			cls.currentEngine = None
		cls.isInitialized = False

	@classmethod
	def _initConfig(cls) -> None:
		"""Initializes the configuration specification for the handler and all its engines."""
		config.conf.spec[cls.configSectionName] = cls.configSpec
		for engineClass in cls.engineClassList or []:
			try:
				engineInstance = engineClass()
				engineName = engineInstance.name
				engineSpec = engineInstance.getConfigSpec()
				if engineSpec:
					config.conf.spec[cls.configSectionName][engineName] = engineSpec
			except Exception:
				log.error(f"Failed to get config spec for engine class {engineClass.__name__}", exc_info=True)

	@classmethod
	def getAllEngineList(cls) -> List[Tuple[str, str]]:
		"""
		Returns a list of installed and supported engines, discovering them if necessary.

		The list contains tuples of (engine_name, engine_description).

		:returns: A list of installed engines that pass their availability checks.
		"""
		if cls.engineList is not None:
			return cls.engineList
		cls.engineList = []
		cls.engineClassList = []
		emptyEngine: Optional[Tuple[str, str]] = None
		for loader, name, isPkg in pkgutil.iter_modules(cls.enginePackage.__path__):
			if name.startswith("_"):
				continue
			try:
				engineClass = cls.getEngine(name)
				if engineClass.check():
					if engineClass.name == "empty":
						emptyEngine = (engineClass.name, engineClass.description)
					else:
						cls.engineList.append((engineClass.name, engineClass.description))
						cls.engineClassList.append(engineClass)
				else:
					log.debugWarning(f"Engine '{name}' failed check, excluding from list.")
			except Exception:
				log.error(f"Error importing or checking engine {name}", exc_info=True)
		cls.engineList.sort(key=lambda s: s[1].lower())
		if emptyEngine:
			cls.engineList.append(emptyEngine)
		return cls.engineList

	@classmethod
	def getEngineList(cls) -> List[Tuple[str, str]]:
		"""
		Returns the user-enabled engines.

		This is the high-level list used by engine selection, engine cycling, and
		automatic fallback. Disabled engines remain installed and configurable, but
		they are not offered for normal use.

		:returns: A filtered list of enabled engines.
		"""
		disabledEngineNames = set(cls.getDisabledEngineNames())
		return [
			(name, description)
			for name, description in cls.getAllEngineList()
			if name == "empty" or name not in disabledEngineNames
		]

	@classmethod
	def getConfigurableEngineList(cls) -> List[Tuple[str, str]]:
		"""
		Returns installed engines that can be enabled or disabled by the user.

		The empty placeholder engine is a fallback implementation detail, not a
		user-configurable engine.

		:returns: A list of installed real engines.
		"""
		return [(name, description) for name, description in cls.getAllEngineList() if name != "empty"]

	@classmethod
	def getDisabledEngineNames(cls) -> List[str]:
		"""
		Returns configured disabled engine names after removing stale entries.

		:returns: A list of disabled engine names known to this handler.
		"""
		try:
			disabledEngineNames = config.conf[cls.configSectionName]["disabledEngines"]
		except (KeyError, AttributeError):
			return []
		if isinstance(disabledEngineNames, str):
			disabledEngineNames = [disabledEngineNames]
		availableEngineNames = {name for name, _description in cls.getConfigurableEngineList()}
		result: List[str] = []
		for engineName in disabledEngineNames:
			engineName = str(engineName)
			if engineName in availableEngineNames and engineName not in result:
				result.append(engineName)
		return result

	@classmethod
	def setDisabledEngineNames(cls, disabledEngineNames: List[str]) -> None:
		"""
		Saves the disabled engine list.

		:param disabledEngineNames: Engine names to exclude from normal use.
		"""
		availableEngineNames = [name for name, _description in cls.getConfigurableEngineList()]
		disabledSet = set(disabledEngineNames)
		normalizedDisabledEngineNames = [name for name in availableEngineNames if name in disabledSet]
		config.conf[cls.configSectionName]["disabledEngines"] = normalizedDisabledEngineNames

	@classmethod
	def isEngineEnabled(cls, engineName: str) -> bool:
		"""
		Returns whether an engine is enabled for normal use.

		:param engineName: The engine name to check.
		:returns: True when the engine is available for selection and cycling.
		"""
		return engineName == "empty" or engineName not in cls.getDisabledEngineNames()

	@classmethod
	def _getAutoEngineCandidateNames(cls) -> List[str]:
		"""Returns enabled engine names in automatic selection order."""
		enabledEngineNames = [name for name, _description in cls.getEngineList()]
		enabledNormalEngineNames = [name for name in enabledEngineNames if name != "empty"]
		priorityNames = [name for name in cls.defaultEnginePriorityList if name in enabledNormalEngineNames]
		remainingNames = [name for name in enabledNormalEngineNames if name not in priorityNames]
		candidateNames = priorityNames + remainingNames
		if not candidateNames and "empty" in enabledEngineNames:
			candidateNames.append("empty")
		return candidateNames

	@classmethod
	def getEngineInstance(cls, name: str) -> "AbstractEngine":
		"""
		Instantiates an engine class by name and initializes its settings.

		:param name: The name of the engine to instantiate.
		:returns: An instance of the requested engine.
		"""
		return cls._getEngineInstance(name)

	@classmethod
	def setCurrentEngine(cls, name: str, isFallback: bool = False) -> bool:
		"""
		Sets the active engine by name.

		Handles "auto" mode by selecting from a default priority list and provides
		fallback logic if an engine fails to load.

		:param name: The name of the engine to set as current.
		:param isFallback: Whether this is a fallback attempt.
		:returns: True if the engine was successfully loaded, False otherwise.
		"""
		if name == "auto":
			candidateEngineNames = cls._getAutoEngineCandidateNames()
		elif not cls.isEngineEnabled(name):
			log.warning(f"Engine '{name}' is disabled for {cls.configSectionName}. Falling back to auto.")
			candidateEngineNames = cls._getAutoEngineCandidateNames()
		else:
			candidateEngineNames = [name]
		prevEngineName = cls.currentEngine.name if cls.currentEngine else None
		if (
			prevEngineName
			and prevEngineName not in candidateEngineNames
			and cls.isEngineEnabled(prevEngineName)
		):
			candidateEngineNames.append(prevEngineName)
		for candidateName in cls._getAutoEngineCandidateNames():
			if candidateName not in candidateEngineNames:
				candidateEngineNames.append(candidateName)
		if not candidateEngineNames:
			log.warning(f"Cannot set engine for {cls.configSectionName} with no enabled engines.")
			if cls.currentEngine:
				cls.currentEngine.cancel()
				cls.currentEngine.terminate()
				cls.currentEngine = None
			if not isFallback:
				config.conf[cls.configSectionName]["engine"] = "auto"
			return False
		triedEngineNames: set[str] = set()
		for candidateName in candidateEngineNames:
			if candidateName in triedEngineNames:
				continue
			triedEngineNames.add(candidateName)
			if prevEngineName == candidateName:
				if not isFallback:
					config.conf[cls.configSectionName]["engine"] = candidateName
				return True
			try:
				newEngine = cls._getEngineInstance(candidateName)
				if cls.currentEngine:
					cls.currentEngine.cancel()
					cls.currentEngine.terminate()
				cls.currentEngine = newEngine
				if not isFallback:
					config.conf[cls.configSectionName]["engine"] = candidateName
				log.debug(f"Successfully loaded engine {candidateName}")
				return True
			except Exception:
				log.warning(f"Failed to load engine '{candidateName}'.", exc_info=True)
		log.warning(f"No available engines could be loaded for {cls.configSectionName}.")
		if cls.currentEngine:
			cls.currentEngine.terminate()
			cls.currentEngine = None
		return False

	@classmethod
	def _getEngineInstance(cls, name: str) -> "AbstractEngine":
		"""
		Instantiates an engine class by name and initializes its settings.

		:param name: The name of the engine to instantiate.
		:returns: An instance of the requested engine.
		"""
		engineClass = cls.getEngine(name)
		newEngine = engineClass()
		if not config.conf[cls.configSectionName].isSet(name):
			config.conf[cls.configSectionName][name] = {}
			c = config.conf[cls.configSectionName][name]
			c.spec = newEngine.getConfigSpec()
			newEngine.saveSettings()
		else:
			newEngine.loadSettings()
		return newEngine

	@classmethod
	def getEngine(cls, name: str) -> Type["AbstractEngine"]:
		"""
		Retrieves the engine class for a given engine name by importing its module.

		:param name: The name of the engine module.
		:returns: The engine class.
		:raises ImportError: If a valid engine class cannot be found in the module.
		"""
		engineModule = cls._importModule(cls.enginePackageName, name)
		if cls.mandatoryClassName:
			return getattr(engineModule, cls.mandatoryClassName)
		for item in dir(engineModule):
			moduleAttribute = getattr(engineModule, item)
			if inspect.isclass(moduleAttribute) and issubclass(moduleAttribute, cls.engineClass):
				return moduleAttribute
		raise ImportError(f"Could not find a valid engine class in module {name}")

	@classmethod
	def getCurrentEngine(cls) -> Optional["AbstractEngine"]:
		"""
		Returns the currently active engine instance.

		:returns: The current engine instance or None.
		"""
		return cls.currentEngine

	@classmethod
	def _importModule(cls, packageName: str, moduleName: str) -> Any:
		"""
		Imports a module dynamically.

		:param packageName: The name of the package containing the module.
		:param moduleName: The name of the module to import.
		:returns: The imported module object.
		"""
		if packageName.startswith("."):
			return importlib.import_module(f"{packageName}.{moduleName}", __package__)
		else:
			fullName = f"{packageName}.{moduleName}"
			return importlib.import_module(fullName)

	@classmethod
	def handlePostConfigProfileSwitch(cls) -> None:
		"""Handles configuration profile switches by reloading the engine if necessary."""
		conf = config.conf[cls.configSectionName]
		safeEngineName = cls._getSafeEngineName(conf["engine"])
		currentEngineName = cls.currentEngine.name if cls.currentEngine else None
		if safeEngineName != currentEngineName:
			cls.setCurrentEngine(safeEngineName)
		elif cls.currentEngine:
			cls.currentEngine.loadSettings(onlyChanged=True)


class AbstractEngine(baseObject.AutoPropertyObject, ABC):
	"""Abstract base class for a single recognition engine."""

	name: str = "empty"
	description: str = ""
	configSectionName: Optional[str] = None
	engineConfigSpec: dict = {}

	@property
	@abstractmethod
	def supportedSettings(self) -> List[EngineSetting]:
		"""
		A list of settings supported by this engine.

		Subclasses must implement this property to expose their configurable options.
		:returns: A list of `EngineSetting` objects.
		"""
		pass

	@staticmethod
	def generateStringSettings(settingsDict: dict) -> dict:
		"""
		Generates a dictionary of StringParameterInfo objects from a simple dict.

		:param settingsDict: A dictionary of {id: displayName}.
		:returns: A dictionary of {id: StringParameterInfo(id, displayName)}.
		"""
		return {key: StringParameterInfo(key, value) for key, value in settingsDict.items()}

	@classmethod
	@abstractmethod
	def check(cls) -> bool:
		"""
		Checks whether this engine can be used on the current system.

		:returns: True if the engine is available, False otherwise.
		"""
		pass

	def cancel(self) -> None:
		"""Cancels any ongoing recognition task."""
		pass

	def terminate(self) -> None:
		"""Terminates the engine, cleaning up any resources."""
		pass

	def saveSettings(self) -> None:
		"""Saves the engine's current settings to the configuration file."""
		if not self.configSectionName:
			return
		conf = config.conf[self.configSectionName][self.name]
		for setting in self.supportedSettings:
			if isinstance(setting, ButtonEngineSetting):
				continue
			conf[_getSettingConfigKey(setting)] = getattr(self, setting.name)

	def loadSettings(self, onlyChanged: bool = False) -> None:
		"""
		Loads the engine's settings from the configuration file.

		:param onlyChanged: If True, only settings that have changed will be applied.
		"""
		if not self.configSectionName:
			return
		conf = config.conf[self.configSectionName][self.name]
		for s in self.supportedSettings:
			if isinstance(s, ButtonEngineSetting):
				continue
			configKey = _getSettingConfigKey(s)
			try:
				val = conf[configKey]
			except KeyError:
				continue
			specStr = s.configSpec
			try:
				if specStr.startswith("boolean"):
					val = is_boolean(val)
				elif specStr.startswith("integer"):
					val = is_integer(val)
				elif specStr.startswith("list"):
					if isinstance(val, str):
						from ast import literal_eval

						val = literal_eval(val)
					val = is_list(val)
			except (ValueError, TypeError, SyntaxError):
				log.debugWarning(
					f"Could not convert setting '{configKey}' with value '{val!r}' based on spec '{specStr}'. Skipping.",
				)
				continue
			if val is None:
				continue
			if onlyChanged and getattr(self, s.name, None) == val:
				continue
			setattr(self, s.name, val)

	def getConfigSpec(self) -> dict:
		"""
		Returns the configuration specification for this engine.

		:returns: A dictionary compatible with `configobj`.
		"""
		spec = self.engineConfigSpec.copy()
		for setting in self.supportedSettings:
			spec[_getSettingConfigKey(setting)] = setting.configSpec
		return spec

	def isSupported(self, settingName: str) -> bool:
		"""
		Checks if a specific setting is supported by this engine.

		:param settingName: The name of the setting to check.
		:returns: True if the setting is supported, False otherwise.
		"""
		return any(s.name == settingName for s in self.supportedSettings)
