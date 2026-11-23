"""Behavior tests for real photo IO and the application service."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from kind_enhance.infrastructure.images import read_photo,prepare_photo,save_png,ImageInputError
from kind_enhance.application.enhancement import EnhancementService
from kind_enhance.inference.pipeline import load_inference_model


class CountingModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.calls=0
    def forward(self,x):
        self.calls+=1
        return (.25+.5*x).clamp(0,1)


class PhotoApplicationTests(unittest.TestCase):
    def setUp(self):
        work=Path(__file__).resolve().parents[2]/'outputs/test_work'
        work.mkdir(parents=True,exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(dir=work)
        self.root=Path(self.temporary.name)
    def tearDown(self):self.temporary.cleanup()
    def test_exif_rotation_is_applied(self):
        image=Image.new('RGB',(24,12),'red');exif=Image.Exif();exif[274]=6
        path=self.root/'ảnh xoay.jpg';image.save(path,exif=exif)
        self.assertEqual(read_photo(path).size,(12,24))
    def test_alpha_is_composited_and_gray_is_rgb(self):
        path=self.root/'transparent.png';Image.new('RGBA',(8,8),(255,0,0,0)).save(path)
        image=read_photo(path)
        self.assertEqual(image.mode,'RGB');self.assertEqual(image.getpixel((0,0)),(255,255,255))
        gray=self.root/'gray.png';Image.new('L',(8,8),64).save(gray)
        self.assertEqual(read_photo(gray).getpixel((0,0)),(64,64,64))
    def test_bad_file_has_clear_error(self):
        path=self.root/'broken.png';path.write_text('not an image')
        with self.assertRaises(ImageInputError):read_photo(path)
    def test_resize_is_bounded_and_preserves_source(self):
        original=Image.new('RGB',(2400,1200))
        processed=prepare_photo(original,800)
        self.assertEqual(processed.size,(800,400));self.assertEqual(original.size,(2400,1200))
        self.assertEqual(prepare_photo(original,0).size,original.size)
    def test_saving_never_overwrites_without_opt_in(self):
        output=self.root/'result.png';save_png(Image.new('RGB',(8,8),'red'),output)
        previous=output.read_bytes()
        with self.assertRaises(FileExistsError):save_png(Image.new('RGB',(8,8),'blue'),output)
        self.assertEqual(output.read_bytes(),previous)
        self.assertFalse(list(self.root.glob('.image-*.png')))
    def test_service_uses_model_once_per_photo_and_loads_once(self):
        image=self.root/'ảnh tối có dấu.png';Image.new('RGB',(32,24),(20,40,60)).save(image)
        model=CountingModel();config={'model_sha256':'fixture-checkpoint'}
        with patch('kind_enhance.application.enhancement.ROOT',self.root),patch('kind_enhance.application.enhancement.load_inference_model',return_value=(model,config)) as loader,patch('kind_enhance.application.enhancement.enhance',side_effect=lambda m,x,c:m(x)):
            service=EnhancementService(threads=2,max_side=1280)
            first=service.enhance_file(image);second=service.enhance_file(image)
        self.assertEqual(loader.call_count,1);self.assertEqual(model.calls,2)
        self.assertNotEqual(first.output_path,second.output_path)
        with Image.open(first.output_path) as result:
            self.assertEqual(result.mode,'RGB');self.assertEqual(result.size,(32,24))
            self.assertEqual(result.getpixel((0,0)),(74,84,94))
        self.assertEqual(len(list((self.root/'outputs/runs').glob('*.json'))),2)
    def test_service_protects_original_and_dataset(self):
        source=self.root/'input.png';Image.new('RGB',(8,8)).save(source)
        with patch('kind_enhance.application.enhancement.ROOT',self.root):
            service=EnhancementService()
            with self.assertRaises(ValueError):service.enhance_file(source,source,overwrite=True)
            with self.assertRaises(ValueError):service.enhance_file(source,self.root/'LOL-v2/high.png')
    def test_photo_loader_does_not_require_dataset_manifest(self):
        config_dir=self.root/'configs';config_dir.mkdir()
        cfg=dict(model='missing.pt',model_sha256='fixture',manifest='data/not_shipped.json',manifest_sha256='fingerprint')
        (config_dir/'model.json').write_text(json.dumps(cfg))
        with patch('kind_enhance.inference.pipeline.ROOT',self.root),patch('kind_enhance.inference.pipeline.sha256',return_value='fixture'),patch('kind_enhance.inference.pipeline.read_checkpoint',return_value={'model':{}}),patch('kind_enhance.inference.pipeline.KinDAttentionCurve',CountingModel),patch('kind_enhance.inference.pipeline.configuration',side_effect=AssertionError('dataset access')):
            model,loaded=load_inference_model()
        self.assertEqual(loaded,cfg);self.assertFalse(model.training)


if __name__=='__main__':unittest.main()
