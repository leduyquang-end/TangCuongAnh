"""Image allowlists are separate for search and final test evaluation."""
import os
from pathlib import Path
import sys


def normalized(path):
    return os.path.normcase(os.path.abspath(os.fsdecode(path)))


class DataGuard:
    def __init__(self, data_root, manifest, splits):
        self.root = normalized(Path(data_root).resolve()) + os.sep
        self.allowed = {name: {normalized(Path(data_root) / row[key]) for row in manifest['splits'][name]
                             for key in ('low', 'high')} for name in splits}
        self.all_allowed = set().union(*self.allowed.values())
        self.opened = {name: set() for name in splits}
        self.blocked = []
        self.test_prefix = normalized(Path(data_root) / 'Test') + os.sep
        self.allow_test = 'test' in splits

    def check(self, path):
        path = normalized(path)
        is_image = path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        forbidden = ((not self.allow_test and path.startswith(self.test_prefix))
                     or (path.startswith(self.root) and is_image and path not in self.all_allowed))
        if forbidden:
            self.blocked.append(path)
            raise PermissionError('Image access outside the frozen split allowlist')
        for name, files in self.allowed.items():
            if path in files:
                self.opened[name].add(path)

    def install(self):
        def audit(event, args):
            if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
                self.check(args[0])
        sys.addaudithook(audit)

    def report(self):
        return {'allowed_splits': list(self.allowed), 'unique_images_opened': {k: len(v) for k, v in self.opened.items()},
                'blocked_attempts': self.blocked, 'test_images_allowed': self.allow_test}
