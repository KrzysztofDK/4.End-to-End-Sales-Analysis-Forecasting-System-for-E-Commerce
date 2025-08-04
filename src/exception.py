"""Module with classes and functions to handle exceptions."""

import sys

from src.logger import logging


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """Returns file name, line number and error message for given error.

    Returns:
        str: Error description.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    return (
        f"Error occured in python script name [{file_name}], "
        f"line number [{exc_tb.tb_lineno}], "
        f"error message[{str(error)}]."
    )


class CustomException(Exception):
    """Own custom exception."""

    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message


if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        logging.info("Divide by zero")
        raise CustomException(e, sys) from e
