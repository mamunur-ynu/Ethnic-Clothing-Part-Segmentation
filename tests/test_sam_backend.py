"""Runs the real transformers SAM 3 code path on a tiny random-weight model.

The weights are random, so the masks are meaningless; this checks the API calls,
tensor shapes and the per-image routine the Modal GPU job uses. Skipped when torch /
transformers are not installed (they are only needed on Modal).
"""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from ethnoseg import settings  # noqa: E402
from ethnoseg.parts import process_file  # noqa: E402
from ethnoseg.sam_backend import Sam3Prompter  # noqa: E402
from tests.synthetic import person_image  # noqa: E402


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    from transformers import CLIPTokenizer, Sam3Config, Sam3Model, Sam3Processor
    from transformers.models.sam3.image_processing_sam3 import Sam3ImageProcessor

    d = tmp_path_factory.mktemp("tiny_sam3")
    vocab = {"<|startoftext|>": 0, "<|endoftext|>": 1}
    for ch in "abcdefghijklmnopqrstuvwxyz":
        vocab[ch] = len(vocab)
        vocab[ch + "</w>"] = len(vocab)
    (d / "vocab.json").write_text(json.dumps(vocab))
    (d / "merges.txt").write_text("#version: 0.2\n")
    tok = CLIPTokenizer(str(d / "vocab.json"), str(d / "merges.txt"), model_max_length=32)
    small = dict(num_attention_heads=2, intermediate_size=64)
    cfg = Sam3Config(
        vision_config=dict(
            backbone_config=dict(model_type="sam3_vit_model", hidden_size=32, num_hidden_layers=2, image_size=112,
                                 patch_size=14, window_size=4, global_attn_indexes=[1], pretrain_image_size=56,
                                 **small),
            fpn_hidden_size=32, backbone_feature_sizes=[[32, 32], [16, 16], [8, 8]]),
        text_config=dict(vocab_size=len(vocab), hidden_size=32, projection_dim=32, num_hidden_layers=2,
                         max_position_embeddings=32, pad_token_id=1, bos_token_id=0, eos_token_id=1, **small),
        geometry_encoder_config=dict(hidden_size=32, num_layers=1, **small),
        detr_encoder_config=dict(hidden_size=32, num_layers=1, **small),
        detr_decoder_config=dict(hidden_size=32, num_layers=1, num_queries=10, **small),
        mask_decoder_config=dict(hidden_size=32, num_attention_heads=2),
    )
    torch.manual_seed(0)
    Sam3Model(cfg).save_pretrained(d / "model")
    Sam3Processor(image_processor=Sam3ImageProcessor(size={"height": 112, "width": 112}),
                  tokenizer=tok).save_pretrained(d / "model")
    return str(d / "model")


def test_prompter_shapes(tiny_model):
    p = Sam3Prompter(tiny_model, device="cpu", score_threshold=0.0)
    img = person_image(1, w=300, h=420)
    out = p(img, ["person", "skirt", "person"])
    assert list(out) == ["person", "skirt"]                    # duplicates run once
    for insts in out.values():
        assert len(insts) > 0                                   # threshold 0 keeps every query
        for m, s in insts:
            assert m.dtype == bool and m.shape == (420, 300) and 0.0 <= s <= 1.0


def test_process_file_with_real_backend(tiny_model, tmp_path):
    S = settings.load()
    seg = dict(S.section("segment"), work_max_side=256, person_threshold=0.0, part_threshold=0.0)
    src = tmp_path / "x.jpg"
    person_image(2, w=300, h=450).save(src)
    p = Sam3Prompter(tiny_model, device="cpu", score_threshold=0.0)
    s = process_file(p, src, tmp_path, "YI00001", S.classes, S.raw["paint_order"], seg)
    assert s["status"] in ("ok", "multiple_subjects", "no_subject"), s
    assert s["work_size"] == [171, 256]
    written = json.loads((tmp_path / "YI00001.json").read_text())
    assert written["status"] == s["status"]
    if s["status"] == "ok":
        from PIL import Image
        lab = np.asarray(Image.open(tmp_path / "YI00001.png"))
        assert lab.shape == (256, 171)
        assert set(np.unique(lab).tolist()) <= {0, 1, 2, 3, 4, 5, 6, 255}
