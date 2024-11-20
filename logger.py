#!/usr/bin/env python
#coding: utf-8
#!python3

"""
General logging solution.
"""

import time
import logging
import os.path


class DB_CONSTRUCTION_Logging:
    """Global logging class.
    """

    def start_logging(self) -> None:
        """Setup logging configuration.
        
        Returns:
            None
        """
        update_time = time.strftime("%Y%m%dT%H%M", time.localtime())
        log_file = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), 
            'LOG', 
            f"{update_time}_DB-CONSTRUCTION-System.log"
        )
        formatter = "%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s"
        handlers = []
        handlers.append("console")  # To be deleted
        handlers.append("session_handler")
        logging_config = dict(
            version = 1,
            disable_existing_loggers = False,
            formatters = {
                "f": {"format": formatter}
            },
            handlers = {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "f",
                    "level": logging.INFO
                },
                "session_handler": {
                    "class": "logging.FileHandler",
                    "formatter": "f",
                    "level": logging.INFO,
                    "filename": log_file,
                    "encoding": "utf-8",
                    "mode": "w"
                }
            },
            root = {
                "handlers": handlers,
                "level": logging.DEBUG
            }
        )
        logging.config.dictConfig(logging_config)
