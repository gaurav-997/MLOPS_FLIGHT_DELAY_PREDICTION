"""Tests for flightdelay.exception.exception"""
import pytest
from flightdelay.exception.exception import CustomException


class TestCustomException:
    def test_inherits_from_exception(self):
        exc = CustomException("something went wrong")
        assert isinstance(exc, Exception)

    def test_str_contains_message(self):
        exc = CustomException("test error", file_name="myfile.py", lineno=42)
        result = str(exc)
        assert "myfile.py" in result
        assert "42" in result
        assert "test error" in result

    def test_explicit_file_and_line(self):
        exc = CustomException("oops", file_name="pipeline.py", lineno=10)
        assert exc.file_name == "pipeline.py"
        assert exc.lineno == 10

    def test_unknown_defaults_when_no_traceback(self):
        exc = CustomException("no tb")
        assert exc.file_name == "<unknown>"
        assert exc.lineno == 0

    def test_raises_and_catches(self):
        with pytest.raises(CustomException):
            raise CustomException("raised correctly")

    def test_str_format(self):
        exc = CustomException("bad input", file_name="a.py", lineno=1)
        s = str(exc)
        assert "a.py" in s
        assert "1" in s
