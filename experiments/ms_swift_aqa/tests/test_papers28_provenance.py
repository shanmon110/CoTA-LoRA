import json
import tarfile
import pytest
from experiments.ms_swift_aqa.papers28.provenance import create_snapshot, verify_snapshot


def test_sources_restorable_and_recapture_content_addressed(tmp_path):
    source=tmp_path/'src';source.mkdir()
    file=source/'plugin.py';file.write_text('ab = 1\n');file.chmod(0o755)
    named={'west/plugin.py':file}
    env={'peft':'0.15.2'}
    first=create_snapshot(tmp_path/'runs',named,env)
    assert create_snapshot(tmp_path/'runs',named,env)==first
    manifest=verify_snapshot(first)
    assert manifest['files']['west/plugin.py']['executable']
    with tarfile.open(first/'sources.tar.gz') as archive:
        assert archive.extractfile('west/plugin.py').read()==file.read_bytes()
    file.write_text('ab = 2\n')
    second=create_snapshot(tmp_path/'runs',named,env)
    assert first != second
    assert verify_snapshot(first)==manifest
    assert create_snapshot(tmp_path/'runs',named,{'peft':'0.17.1'}) != second
    (first/'sources.tar.gz').write_bytes(b'corruption')
    with pytest.raises(ValueError):
        verify_snapshot(first)


def test_rejects_escaping_name(tmp_path):
    f=tmp_path/'f';f.write_text('source')
    with pytest.raises(ValueError):
        create_snapshot(tmp_path/'out',{'../f':f},{})
