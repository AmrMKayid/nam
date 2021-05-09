"""Base logger, borrowed from DeepMind"s Acme."""

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
        self.info = self._logger.info
        self.trace = self._logger.trace
        self.debug = self._logger.debug
        self.warning = self._logger.warning
        self.success = self._logger.success
        self.error = self._logger.error
        self.critical = self._logger.critical
        self._logger.add(f"{log_dir}/out.log")

    def write(self, data: LoggingData):
        """Writes `data` to destination (file, terminal, database, etc)."""
        raise NotImplementedError

    @property
    def logger(self):
        return self._logger
