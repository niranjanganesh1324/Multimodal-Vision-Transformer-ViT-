"""Unit tests for Image Captioning, VQA, and Cross-Modal Retrieval inference."""

import pytest
import torch
from PIL import Image
import numpy as np

from models.multimodal_model import MultiModalViT
from inference.caption import ImageCaptioner
from inference.vqa import VQAEngine
from inference.retrieval import CrossModalRetriever
from utils.tokenizer import MultiModalTokenizer


@pytest.fixture
def dummy_setup():
    tokenizer = MultiModalTokenizer(model_name="bert-base-uncased")
    model = MultiModalViT(
        hidden_dim=256,
        projection_dim=128,
        num_fusion_layers=2,
        num_heads=4,
        vocab_size=len(tokenizer),
        vqa_num_classes=10,
    )
    img = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
    return model, tokenizer, img


def test_captioner_greedy_and_beam(dummy_setup):
    model, tokenizer, img = dummy_setup
    captioner = ImageCaptioner(model=model, tokenizer=tokenizer)

    greedy_cap = captioner.generate_caption(img, decoding_strategy="greedy", max_length=10)
    assert isinstance(greedy_cap, str)

    beam_cap = captioner.generate_caption(img, decoding_strategy="beam_search", beam_size=3, max_length=10)
    assert isinstance(beam_cap, str)


def test_vqa_engine(dummy_setup):
    model, tokenizer, img = dummy_setup
    vqa_engine = VQAEngine(model=model, tokenizer=tokenizer, answer_vocab=["red", "blue", "yes"])

    res = vqa_engine.predict_answer(img, question="What color is this?", top_k=2)
    assert "answer" in res
    assert "confidence" in res
    assert len(res["top_k_answers"]) == 2


def test_retrieval_engine(dummy_setup):
    model, tokenizer, img = dummy_setup
    retriever = CrossModalRetriever(model=model, tokenizer=tokenizer)

    images = [img, img]
    texts = ["A brown dog in a park.", "A red sports car on the road."]
    retriever.index_dataset(images, texts)

    txt_results = retriever.search_texts_by_image(img, top_k=2)
    assert len(txt_results) == 2
    assert "text" in txt_results[0]

    img_results = retriever.search_images_by_text("dog park", top_k=2)
    assert len(img_results) == 2
    assert "metadata" in img_results[0]
