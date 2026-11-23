"""A cloned checkout uses its bundled data without rewriting the signed split."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from kind_enhance.inference.pipeline import ROOT,configuration


class DatasetLocationTests(unittest.TestCase):
    def test_cloned_checkout_prefers_local_data_and_preserves_manifest(self):
        work=ROOT/'outputs/test_work';work.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work) as folder:
            root=Path(folder);(root/'configs').mkdir();(root/'data').mkdir()
            manifest=root/'data/split.json'
            content=json.dumps({'source':{'data_root':'Z:/old-machine/Real_captured'},'splits':{}}).encode()
            manifest.write_bytes(content)
            (root/'configs/model.json').write_text(json.dumps({'manifest':'data/split.json','manifest_sha256':hashlib.sha256(content).hexdigest()}))
            with patch('kind_enhance.inference.pipeline.ROOT',root):
                _,before=configuration()
                self.assertEqual(before['source']['data_root'],'Z:/old-machine/Real_captured')
                local=root/'LOL-v2/LOL-v2/Real_captured'
                (local/'Train/Low').mkdir(parents=True);(local/'Test/Low').mkdir(parents=True)
                _,after=configuration()
            self.assertEqual(Path(after['source']['data_root']),local.resolve())
            self.assertEqual(manifest.read_bytes(),content)


if __name__=='__main__':unittest.main()
