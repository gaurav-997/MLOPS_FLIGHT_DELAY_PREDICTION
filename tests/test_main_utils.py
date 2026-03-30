"""Tests for flightdelay.utils.main_utils"""
import os
import tempfile
import numpy as np
import pytest
from flightdelay.utils.main_utils import (
    save_object,
    load_object,
    save_numpy_array_data,
    load_numpy_array_data,
)


class TestSaveLoadObject:
    def test_save_and_load_dict(self, tmp_path):
        obj = {"key": "value", "number": 42}
        path = str(tmp_path / "test.pkl")
        save_object(path, obj)
        loaded = load_object(path)
        assert loaded == obj

    def test_save_and_load_list(self, tmp_path):
        obj = [1, 2, 3, "hello"]
        path = str(tmp_path / "list.pkl")
        save_object(path, obj)
        assert load_object(path) == obj

    def test_save_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "nested" / "deep" / "obj.pkl")
        save_object(path, {"a": 1})
        assert os.path.exists(path)

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(Exception):
            load_object(str(tmp_path / "nonexistent.pkl"))

    def test_save_and_load_custom_class(self, tmp_path):
        class Simple:
            def __init__(self, x):
                self.x = x

        path = str(tmp_path / "cls.pkl")
        save_object(path, Simple(99))
        loaded = load_object(path)
        assert loaded.x == 99


class TestSaveLoadNumpyArray:
    def test_save_and_load_1d(self, tmp_path):
        arr = np.array([1.0, 2.0, 3.0])
        path = str(tmp_path / "array.npy")
        save_numpy_array_data(path, arr)
        loaded = load_numpy_array_data(path)
        np.testing.assert_array_equal(arr, loaded)

    def test_save_and_load_2d(self, tmp_path):
        arr = np.arange(12).reshape(3, 4).astype(float)
        path = str(tmp_path / "array2d.npy")
        save_numpy_array_data(path, arr)
        loaded = load_numpy_array_data(path)
        np.testing.assert_array_equal(arr, loaded)

    def test_save_creates_parent_dirs(self, tmp_path):
        arr = np.zeros((5,))
        path = str(tmp_path / "sub" / "dir" / "arr.npy")
        save_numpy_array_data(path, arr)
        assert os.path.exists(path)

    def test_load_missing_numpy_file_raises(self, tmp_path):
        with pytest.raises(Exception):
            load_numpy_array_data(str(tmp_path / "missing.npy"))

    def test_shape_preserved(self, tmp_path):
        arr = np.random.rand(10, 5)
        path = str(tmp_path / "shaped.npy")
        save_numpy_array_data(path, arr)
        loaded = load_numpy_array_data(path)
        assert loaded.shape == (10, 5)
