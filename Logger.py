import logging
import colorlog

""" Logger Rank: (low -> high)
# 1. DEBUG
# 2. INFO
# 3. WARNING
# 4. ERROR
# 5. CRITICAL
"""


class Logger(object):
    def __init__(self, log_path, logging_name):
        self.log_path = log_path
        self.logging_name = logging_name
        self.dash_line = '-' * 60 + '\n'
        self.level_color = {'DEBUG': 'cyan',
                            'INFO': 'bold_white',
                            'WARNING': 'yellow',
                            'ERROR': 'red',
                            'CRITICAL': 'red'}

    def logger_config(self):
        logger = logging.getLogger(self.logging_name)
        logger.setLevel(level=logging.DEBUG)
        handler = logging.FileHandler(self.log_path, encoding='UTF-8')
        handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                           datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(file_formatter)

        console_formatter = colorlog.ColoredFormatter(
            '%(log_color)s[%(asctime)s] - [%(name)s] - [%(levelname)s]:\n%(message)s', datefmt="%Y-%m-%d %H:%M:%S",
            log_colors=self.level_color)

        console = logging.StreamHandler()
        console.setFormatter(console_formatter)
        console.setLevel(logging.INFO)

        logger.addHandler(handler)
        logger.addHandler(console)
        return logger
