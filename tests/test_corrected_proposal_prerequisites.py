"""Exact optional evidence binding; CPU metadata only, no rollout or metrics."""
from copy import deepcopy
import hashlib
import json

import pytest

from scripts.probes import variance_proposal_audit as probe


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    root = tmp_path/'source'
    blobs = {'physmorph/__init__.py': b'guard\n',
             'physmorph/mpm/constitutive.py': b'rotation\n',
             'physmorph/mpm/kernels.py': b'stress\n',
             'physmorph/mpm/function.py': b'bridge\n',
             'scripts/probes/position_sequence.py': b'primitive\n',
             'tests/test_corotated_adjoint.py': b'chain tests\n'}
    for name, value in blobs.items():
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    position = dict(passed=True, device=dict(warp='1.16.0'),
                    checks=[dict(passed=True) for _ in range(52)],
                    code_sha256={name: digest(value) for name, value in blobs.items()
                                 if not name.startswith('tests/')})
    constitutive = dict(passed=True, status=0, device='cuda', warp_version='1.16.0',
                        source_root='/data/frozen', code_sha256={
                            '/data/frozen/'+name: digest(blobs[name]) for name in
                            ('physmorph/mpm/constitutive.py', 'physmorph/mpm/kernels.py',
                             'tests/test_corotated_adjoint.py')})
    xml = ('<testsuites><testsuite tests="10" errors="0" failures="0" skipped="0">'
           + ''.join(f'<testcase name="case{i}"/>' for i in range(10))
           + '</testsuite></testsuites>')
    paths = {'position': tmp_path/'position.json', 'constitutive': tmp_path/'constitutive.json',
             'constitutive_xml': tmp_path/'constitutive.xml'}

    def write(kind, value):
        raw = (json.dumps(value).encode() if kind != 'constitutive_xml' else value.encode())
        paths[kind].write_bytes(raw)
        approved[kind] = digest(raw)

    approved = {}
    monkeypatch.setattr(probe, 'CURRENT_EVIDENCE_SHA', approved)
    for kind, value in [('position', position), ('constitutive', constitutive), ('constitutive_xml', xml)]:
        write(kind, value)
    return dict(root=root, paths=paths, position=position, constitutive=constitutive, xml=xml, write=write)


def validate(e):
    return probe.current_prerequisites(e['paths']['position'], e['paths']['constitutive'], source_root=e['root'])


def test_optional_absence_preserves_legacy_execution_and_partial_evidence_is_rejected(evidence):
    assert probe.current_prerequisites() is None
    for paths in ((evidence['paths']['position'], None), (None, evidence['paths']['constitutive'])):
        with pytest.raises(RuntimeError, match='required together'):
            probe.current_prerequisites(*paths, source_root=evidence['root'])


def test_exact_passes_are_bound_to_sources_without_reclassifying_history(evidence):
    result = validate(evidence)
    assert result['passed'] is True
    assert result['position_passed_checks'] == 52
    assert result['constitutive_passed_tests'] == 10
    assert 'no source-diff exception' in result['source_binding']
    assert 'not the current prerequisite status' in probe.HISTORICAL_PRIMITIVE_SCOPE
    assert result['evidence']['position']['sha256'] == digest(evidence['paths']['position'].read_bytes())
    assert 'tests/test_corotated_adjoint.py' in result['source_dependencies']


@pytest.mark.parametrize('kind', ['position', 'constitutive', 'constitutive_xml'])
def test_even_whitespace_change_to_approved_evidence_is_rejected(evidence, kind):
    path = evidence['paths'][kind]
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(RuntimeError, match='Unapproved current prerequisite bytes'):
        validate(evidence)


@pytest.mark.parametrize('change', ['missing', 'extra', 'failed', 'nonboolean', 'overall', 'runtime'])
def test_position_semantics_fail_closed_independently_of_byte_check(evidence, change):
    value = deepcopy(evidence['position'])
    if change == 'missing':
        value['checks'].pop()
    elif change == 'extra':
        value['checks'].append(dict(passed=True))
    elif change == 'failed':
        value['checks'][17]['passed'] = False
    elif change == 'nonboolean':
        value['checks'][17]['passed'] = 1
    elif change == 'overall':
        value['passed'] = False
    else:
        value['device']['warp'] = '1.9.0'
    evidence['write']('position', value)
    with pytest.raises(RuntimeError):
        validate(evidence)


@pytest.mark.parametrize('key,value', [('passed', False), ('status', 1), ('device', 'cpu'),
                                       ('warp_version', '1.9.0')])
def test_constitutive_success_cannot_be_inferred_from_only_one_flag(evidence, key, value):
    record = deepcopy(evidence['constitutive'])
    record[key] = value
    evidence['write']('constitutive', record)
    with pytest.raises(RuntimeError, match='did not pass'):
        validate(evidence)


@pytest.mark.parametrize('change', ['count', 'skip', 'failure_node', 'missing_case'])
def test_ten_actual_unskipped_cases_are_required(evidence, change):
    xml = evidence['xml']
    if change == 'count':
        xml = xml.replace('tests="10"', 'tests="9"')
    elif change == 'skip':
        xml = xml.replace('skipped="0"', 'skipped="1"')
    elif change == 'failure_node':
        xml = xml.replace('<testcase name="case0"/>', '<testcase name="case0"><failure/></testcase>')
    else:
        xml = xml.replace('<testcase name="case0"/>', '')
    evidence['write']('constitutive_xml', xml)
    with pytest.raises(RuntimeError):
        validate(evidence)


@pytest.mark.parametrize('name', ['physmorph/mpm/constitutive.py', 'physmorph/mpm/kernels.py',
                                 'physmorph/mpm/function.py', 'scripts/probes/position_sequence.py',
                                 'tests/test_corotated_adjoint.py'])
def test_changed_source_or_test_bytes_invalidate_current_evidence(evidence, name):
    path = evidence['root']/name
    path.write_bytes(path.read_bytes()+b'# changed\n')
    with pytest.raises(RuntimeError, match='source differs'):
        validate(evidence)


def test_extra_physics_module_and_missing_constitutive_binding_are_rejected(evidence):
    extra = evidence['root']/'physmorph/extra.py'
    extra.write_bytes(b'# not in passed source\n')
    with pytest.raises(RuntimeError, match='membership'):
        validate(evidence)
    extra.unlink()
    record = deepcopy(evidence['constitutive'])
    del record['code_sha256']['/data/frozen/physmorph/mpm/constitutive.py']
    evidence['write']('constitutive', record)
    with pytest.raises(RuntimeError, match='Missing constitutive prerequisite dependency'):
        validate(evidence)
