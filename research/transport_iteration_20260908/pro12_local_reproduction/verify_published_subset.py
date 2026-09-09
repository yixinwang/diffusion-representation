"""Verify the delivered text subset, explicitly accounting for omitted executables."""
from pathlib import Path
import hashlib
import json


def main():
    root = Path(__file__).resolve().parent
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    delivery = json.loads((root / 'DELIVERY_SHA256.json').read_text())
    for relative, expected in delivery.items():
        assert sha(root / relative) == expected, relative
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts
              and p.name != 'DELIVERY_SHA256.json'}
    assert actual == set(delivery), 'unexpected or missing delivered files'
    complete = json.loads((root / 'results/COMPLETE.json').read_text())
    omissions = json.loads((root / 'OMITTED_BINARIES.json').read_text())['omitted']
    assert set(omissions) == {'certify', 'global_bounds', 'risk_bounds', 'test_certificate'}
    available = 0
    for relative, expected in complete['payload_sha256'].items():
        path = root / 'results' / relative
        if relative in omissions:
            assert not path.exists(), relative
            assert omissions[relative]['sha256'] == expected, relative
            assert omissions[relative]['bytes'] > 0
        else:
            assert sha(path) == expected, relative
            available += 1
    assert len(complete['payload_sha256']) == available + len(omissions) == 58
    assert complete['full_fixed_grid'] and complete['grid'] == [4, 8, 16, 32, 64]
    print(json.dumps({'status': 'text_subset_verified',
                      'available_original_payloads': available,
                      'explicitly_omitted_executables': len(omissions),
                      'original_COMPLETE_manifest_preserved': True,
                      'integration_rerun': False}, indent=2))


if __name__ == '__main__':
    main()
