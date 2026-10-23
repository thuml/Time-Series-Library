#!/usr/bin/env python3
"""Fetch the LTSF Traffic CSV, with a pinned mirror and content checksum.

The official TimeXer README links the TimesNet benchmark collection on Drive.
This mirror contains the same all_six_datasets/traffic/traffic.csv layout.
Only traffic.csv is extracted; other archive members are never written.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile


REVISION = '4980f52e6c575e4456bbe5943ad40a2158a08753'
URL = ('https://huggingface.co/datasets/ChanceuxMJ/Autoformer/resolve/'
       + REVISION + '/all_six_datasets.zip?download=true')
SHA256 = 'cb06463d56fa17d87f47027cd9389ceae82a69eddee51cdb61480e120dab0b16'
MEMBER = 'all_six_datasets/traffic/traffic.csv'
OFFICIAL = 'https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2'


def checksum(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(destination, archive=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / 'traffic.csv'
    if target.exists():
        if checksum(target) != SHA256:
            raise ValueError(f'{target} has an unexpected checksum; existing file was preserved')
    else:
        with tempfile.TemporaryDirectory(dir=destination) as work:
            if archive is None:
                archive = Path(work) / 'datasets.zip'
                subprocess.run(['curl', '--fail', '--location', '--retry', '2',
                                '--connect-timeout', '20', '--max-time', '600',
                                URL, '--output', str(archive)], check=True)
            pending = Path(work) / 'traffic.csv'
            with zipfile.ZipFile(archive) as source:
                with source.open(MEMBER) as src, pending.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
            if checksum(pending) != SHA256:
                raise ValueError('Traffic checksum mismatch; no dataset was installed')
            os.replace(pending, target)
    metadata = dict(dataset='Traffic', official_repository='https://github.com/thuml/TimeXer',
                    official_collection=OFFICIAL, download_url=URL, archive_member=MEMBER,
                    sha256=SHA256, rows=17544, sensors=862, frequency='h', target='OT',
                    split='chronological 70/10/20; scaler fit on train only',
                    source_note='Downloaded from a third-party mirror of the LTSF collection; '
                                'checksum pins this copy, not a published author checksum.')
    (destination / 'SOURCE.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Traffic ready: {target} (17,544 hours, 862 sensors; SHA-256 verified)')
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'data' / 'traffic')
    parser.add_argument('--archive', type=Path, help='Use an already downloaded all_six_datasets.zip')
    args = parser.parse_args()
    prepare(args.destination, args.archive)
