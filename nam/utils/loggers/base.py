"""Base logger, borrowed from DeepMind"s Acme."""

import abc
import sys
from typing import Any
from typing import Mapping

from loguru import logger

LoggingData = Mapping[str, Any]


class Logger:
  """A logger has a `write` method."""

  def __init__(
      self,
      log_dir: str,
  ) -> None:
    self._logger = logger
    self._logger.add(f"{log_dir}/out.log")

  def write(self, data: LoggingData):
    """Writes `data` to destination (file, terminal, database, etc)."""
    raise NotImplementedError

  def trace(self, *args):
    self._logger.trace(*args)

  def debug(self, *args):
    self._logger.debug(*args)

  def info(self, *args):
    self._logger.info(*args)

  def success(self, *args):
    self._logger.success(*args)

  def warning(self, *args):
    self._logger.warning(*args)

  def error(self, *args):
    self._logger.error(*args)

  def critical(self, *args):
    self._logger.critical(*args)

  @property
  def logger(self):
    return self._logger
