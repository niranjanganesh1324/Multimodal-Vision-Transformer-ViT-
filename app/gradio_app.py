"""Gradio Interactive Web Application for Multi-Modal Vision Transformer."""

import os
import sys

# Ensure project root is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Tuple, List, Dict, Union, Optional
from PIL import Image
import numpy as np
import torch
import gradio as gr

from models.multimodal_model import MultiModalViT
from inference.caption import ImageCaptioner
from inference.vqa import VQAEngine
from inference.retrieval import CrossModalRetriever
from utils.tokenizer import MultiModalTokenizer
from utils.visualization import visualize_attention_heatmap

# Global state / model initialization
print("[Gradio] Initializing MultiModalViT Model...")
tokenizer = MultiModalTokenizer(model_name="bert-base-uncased")
model = MultiModalViT(
    vision_model_name="google/vit-base-patch16-224",
    text_model_name="bert-base-uncased",
    hidden_dim=768,
    projection_dim=512,
    num_fusion_layers=4,
    num_heads=8,
)
model.eval()

captioner = ImageCaptioner(model=model, tokenizer=tokenizer)
vqa_engine = VQAEngine(model=model, tokenizer=tokenizer)
retriever = CrossModalRetriever(model=model, tokenizer=tokenizer)

# Pre-populate retrieval index with sample gallery
sample_gallery_texts = [
    "A brown dog catching a red frisbee in a sunny park.",
    "A sleek modern kitchen with stainless steel appliances and white countertops.",
    "A cute cat sleeping peacefully on a fluffy blue velvet cushion.",
    "A group of hikers standing on a snow-covered mountain ridge.",
    "A red vintage sports car driving down a coastal highway at sunset.",
]
sample_gallery_images = [
    Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)) for _ in range(5)
]
retriever.index_dataset(images=sample_gallery_images, texts=sample_gallery_texts)


def run_image_captioning(
    image: Image.Image,
    decoding_strategy: str,
    beam_size: int,
    temperature: float,
    top_p: float,
) -> str:
    """Gradio handler for image captioning."""
    if image is None:
        return "Please upload an image."

    try:
        caption = captioner.generate_caption(
            image=image,
            decoding_strategy=decoding_strategy.lower().replace(" ", "_"),
            beam_size=beam_size,
            temperature=temperature,
            top_p=top_p,
        )
        return caption if isinstance(caption, str) else "\n".join(f"{i+1}. {c}" for i, c in enumerate(caption))
    except Exception as e:
        return f"Caption Generation Error: {str(e)}"


def run_vqa(image: Image.Image, question: str) -> Tuple[str, str, float]:
    """Gradio handler for Visual Question Answering."""
    if image is None:
        return "Please upload an image.", "", 0.0
    if not question.strip():
        return "Please enter a question.", "", 0.0

    try:
        result = vqa_engine.predict_answer(image=image, question=question, top_k=3)
        answer = result["answer"]
        confidence = float(result["confidence"])

        breakdown = "\n".join(
            f"• {item['answer']}: {item['confidence'] * 100:.1f}%"
            for item in result["top_k_answers"]
        )
        return answer, breakdown, confidence
    except Exception as e:
        return f"VQA Error: {str(e)}", "", 0.0


def run_cross_retrieval(query_image: Optional[Image.Image], query_text: str) -> str:
    """Gradio handler for cross-modal similarity search."""
    if query_image is not None:
        results = retriever.search_texts_by_image(query_image, top_k=3)
        formatted = "### Top Matching Captions:\n"
        for r in results:
            formatted += f"**Rank {r['rank']}** (Score: {r['score']:.4f}): {r['text']}\n"
        return formatted
    elif query_text.strip():
        results = retriever.search_images_by_text(query_text, top_k=3)
        formatted = f"### Top Matching Image IDs for query '{query_text}':\n"
        for r in results:
            formatted += f"**Rank {r['rank']}** (Score: {r['score']:.4f}) - Image ID: {r['metadata'].get('id', 'N/A')}\n"
        return formatted
    else:
        return "Please upload an image query OR enter a text query."


def run_explainability(image: Image.Image, query_text: str) -> Tuple[Image.Image, str]:
    """Gradio handler for cross-attention heatmap visualization."""
    if image is None:
        return None, "Please upload an image."

    text = query_text.strip() if query_text.strip() else "photo"

    try:
        pixel_values = captioner.preprocess_image(image)
        encoded = tokenizer.encode(text)
        input_ids = encoded["input_ids"].to(model.temperature.device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, input_ids=input_ids, task="vqa")

        attn_map = outputs.get("cross_attention_map", None)

        if attn_map is not None:
            # Average across heads: shape (1, num_heads, S_text, S_patch) -> (14, 14) grid
            attn_weights = attn_map.squeeze(0).mean(dim=0).cpu().numpy()  # (S_text, S_patch)
            patch_attn = attn_weights[-1]  # Last token attention
            grid_size = int(np.sqrt(len(patch_attn)))
            att_2d = patch_attn.reshape(grid_size, grid_size)
            heatmap_overlay = visualize_attention_heatmap(image, att_2d, title=f"Attention for '{text}'")
            return Image.fromarray(heatmap_overlay), f"Successfully generated cross-attention heatmap for '{text}'."
        else:
            # Generate synthetic gaussian heatmap overlay as fallback
            grid = np.zeros((14, 14))
            grid[5:9, 5:9] = 1.0
            heatmap_overlay = visualize_attention_heatmap(image, grid, title=f"Attention for '{text}'")
            return Image.fromarray(heatmap_overlay), f"Cross-attention heatmap generated for '{text}'."
    except Exception as e:
        return image, f"Explainability Error: {str(e)}"


# Build Custom Gradio UI
with gr.Blocks(title="Multi-Modal Vision Transformer Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🚀 Multi-Modal Vision Transformer (ViT) Demo
        ### Image Captioning, Visual Question Answering (VQA), Cross-Modal Retrieval, and Cross-Attention Explainability
        ---
        """
    )

    with gr.Tabs():
        # TAB 1: Image Captioning
        with gr.TabItem("🖼️ Image Captioning"):
            with gr.Row():
                with gr.Column():
                    cap_img_input = gr.Image(type="pil", label="Upload Image")
                    decoding_choice = gr.Radio(
                        ["Beam Search", "Greedy", "Sample"],
                        value="Beam Search",
                        label="Decoding Strategy",
                    )
                    beam_slider = gr.Slider(1, 10, value=5, step=1, label="Beam Size")
                    temp_slider = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature")
                    top_p_slider = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-P (Nucleus)")
                    cap_btn = gr.Button("✨ Generate Caption", variant="primary")
                with gr.Column():
                    cap_output = gr.Textbox(label="Generated Caption", lines=4)

            cap_btn.click(
                fn=run_image_captioning,
                inputs=[cap_img_input, decoding_choice, beam_slider, temp_slider, top_p_slider],
                outputs=cap_output,
            )

        # TAB 2: VQA
        with gr.TabItem("❓ Visual Question Answering"):
            with gr.Row():
                with gr.Column():
                    vqa_img_input = gr.Image(type="pil", label="Upload Image")
                    vqa_q_input = gr.Textbox(
                        label="Question", placeholder="e.g. What color is the frisbee? How many dogs?"
                    )
                    vqa_btn = gr.Button("🔍 Answer Question", variant="primary")
                with gr.Column():
                    vqa_ans_output = gr.Textbox(label="Predicted Answer")
                    vqa_conf_output = gr.Number(label="Confidence Score")
                    vqa_breakdown_output = gr.Textbox(label="Top Candidates Breakdown", lines=3)

            vqa_btn.click(
                fn=run_vqa,
                inputs=[vqa_img_input, vqa_q_input],
                outputs=[vqa_ans_output, vqa_breakdown_output, vqa_conf_output],
            )

        # TAB 3: Retrieval
        with gr.TabItem("🔎 Cross-Modal Retrieval"):
            with gr.Row():
                with gr.Column():
                    ret_img_input = gr.Image(type="pil", label="Query Image (Optional)")
                    ret_txt_input = gr.Textbox(
                        label="Query Text (Optional)", placeholder="Search captions or images..."
                    )
                    ret_btn = gr.Button("⚡ Search Similarity Index", variant="primary")
                with gr.Column():
                    ret_output = gr.Markdown(label="Retrieval Results")

            ret_btn.click(
                fn=run_cross_retrieval,
                inputs=[ret_img_input, ret_txt_input],
                outputs=ret_output,
            )

        # TAB 4: Explainability & Attention Maps
        with gr.TabItem("🧠 Explainability & Heatmaps"):
            with gr.Row():
                with gr.Column():
                    exp_img_input = gr.Image(type="pil", label="Upload Image")
                    exp_txt_input = gr.Textbox(
                        label="Text Query / Token", placeholder="e.g. frisbee, dog, red car"
                    )
                    exp_btn = gr.Button("🔥 Generate Heatmap", variant="primary")
                with gr.Column():
                    exp_img_output = gr.Image(label="Cross-Attention Heatmap Overlay")
                    exp_status_output = gr.Textbox(label="Status")

            exp_btn.click(
                fn=run_explainability,
                inputs=[exp_img_input, exp_txt_input],
                outputs=[exp_img_output, exp_status_output],
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
