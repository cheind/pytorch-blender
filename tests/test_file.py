import pytest
from blendtorch.btt.file import FileRecorder, FileReader


@pytest.mark.background
def test_file_recorder_reader(tmp_path):
    with FileRecorder(outpath=tmp_path / "record.mpkl", max_messages=10) as rec:
        for i in range(7):
            rec.save({"value": i}, is_pickled=False)

    r = FileReader(tmp_path / "record.mpkl")
    assert len(r) == 7
    for i in range(7):
        assert r[i]["value"] == i


@pytest.mark.background
def test_file_recorder_reader_exception(tmp_path):
    try:
        with FileRecorder(
            outpath=tmp_path / "record.mpkl", max_messages=10, update_header=1
        ) as rec:
            rec.save({"value": 0}, is_pickled=False)
            rec.save({"value": 1}, is_pickled=False)
            rec.save({"value": 2}, is_pickled=False)
            raise ValueError("err")
    except ValueError:
        pass

    r = FileReader(tmp_path / "record.mpkl")
    assert len(r) == 3
    for i in range(3):
        assert r[i]["value"] == i
