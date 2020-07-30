import pytest
from blendtorch.btt.file import FileRecorder, FileReader

@pytest.mark.offscreen
def test_file_record_write(tmp_path):
    with FileRecorder(outpath=tmp_path/'record.mpkl', max_messages=10) as rec:
        for i in range(7):
            rec.save({'value':i}, is_pickled=False)

    r = FileReader(tmp_path/'record.mpkl')
    assert len(r) == 7
    for i in range(7):
        assert r[i]['value'] == i