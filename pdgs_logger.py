# pdgs_logger.py
"""
Structured logging matching PDGS-ID-XPR-IRIDE-0209 Section 3.1.4 exactly.

Required format (fields separated by 1 blank character):
  1. Date/Time: yyyy-MM-ddThh:mm:ss.nnnnnn (26 chars)
  2. Node Name: hostname (via gethostname)
  3. Processor Name
  4. Processor Version: nn.nn (5 chars)
  5. PID: [nnnnnnnnnn] (12 chars, brackets included)
  6. Working Folder Name: basename of the working directory
  7. Header separator: ':'
  8. Message Type: [D]/[I]/[P]/[W]/[E] (5 chars incl. brackets)
  9. Message Text: free-form, one line

[D]/[I]/[P] -> stdout. [W]/[E] -> stderr.
stdout/stderr must not be buffered; every message is flushed immediately.

Example (from the ICD):
  2004-02-24T04:02:07.458000 ipf1ws1 IPF1 01.04 [0000013875] WORKINGDIR0105 :
  [I] processor starting. The config file is /home/IPF1/order0105.xml
"""

import os
import socket
import sys
from datetime import datetime, timezone


LEVEL_ORDER = ["DEBUG", "INFO", "PROGRESS", "WARNING", "ERROR"]

_TYPE_CHAR = {
    "DEBUG": "D",
    "INFO": "I",
    "PROGRESS": "P",
    "WARNING": "W",
    "ERROR": "E",
}

_TYPE_STREAM = {
    "DEBUG": "stdout",
    "INFO": "stdout",
    "PROGRESS": "stdout",
    "WARNING": "stderr",
    "ERROR": "stderr",
}


class PdgsLogger:
    def __init__(self, processor_name, processor_version, working_dir, logging_level="INFO"):
        """
        processor_version: string in "nn.nn" format (e.g. "01.00").
        logging_level: one of DEBUG/INFO/PROGRESS/WARNING/ERROR, from the
                       Job Order's Ipf_Conf/Logging_Level. Messages at a
                       LOWER verbosity than this setting are suppressed
                       (DEBUG is most verbose, ERROR is least, per the
                       ICD's ordering).
        """
        self.processor_name = processor_name
        self.processor_version = processor_version
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        self.working_folder_name = os.path.basename(str(working_dir).rstrip("/")) or str(working_dir)

        if logging_level not in LEVEL_ORDER:
            logging_level = "INFO"
        self._min_level_index = LEVEL_ORDER.index(logging_level)

    def _enabled(self, level):
        return LEVEL_ORDER.index(level) >= self._min_level_index

    def _emit(self, level, message):
        if not self._enabled(level):
            return

        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y-%m-%dT%H:%M:%S.%f")

        type_char = _TYPE_CHAR[level]
        line = (
            f"{timestamp} {self.hostname} {self.processor_name} "
            f"{self.processor_version} [{self.pid:010d}] "
            f"{self.working_folder_name} : [{type_char}] {message}"
        )

        stream = sys.stdout if _TYPE_STREAM[level] == "stdout" else sys.stderr
        stream.write(line + "\n")
        stream.flush()

    def debug(self, message):
        self._emit("DEBUG", message)

    def info(self, message):
        self._emit("INFO", message)

    def progress(self, message):
        self._emit("PROGRESS", message)

    def warning(self, message):
        self._emit("WARNING", message)

    def error(self, message):
        self._emit("ERROR", message)

    def exception(self, message):
        import traceback
        tb = traceback.format_exc()
        self.error(f"{message}\n{tb}")