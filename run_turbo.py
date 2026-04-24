import random
import numpy as np
import torch
import gradio as gr
from chatterbox.tts_turbo import ChatterboxTurboTTS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EVENT_TAGS = [
    "[clear throat]", "[sigh]", "[shush]", "[cough]", "[groan]",
    "[sniff]", "[gasp]", "[chuckle]", "[laugh]"
]


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


print(f"Loading Chatterbox-Turbo on {DEVICE}...")
MODEL = ChatterboxTurboTTS.from_pretrained(DEVICE)
print("Model loaded!")


def generate(
        text,
        audio_prompt_path,
        temperature,
        seed_num,
        min_p,
        top_p,
        top_k,
        repetition_penalty,
        norm_loudness
):
    if seed_num != 0:
        set_seed(int(seed_num))

    wav = MODEL.generate(
        text,
        audio_prompt_path=audio_prompt_path,
        temperature=temperature,
        min_p=min_p,
        top_p=top_p,
        top_k=int(top_k),
        repetition_penalty=repetition_penalty,
        norm_loudness=norm_loudness,
    )
    return (MODEL.sr, wav.squeeze(0).numpy())


with gr.Blocks(title="Chatterbox Turbo") as demo:
    gr.Markdown("# ⚡ Chatterbox Turbo")

    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                value="Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. It's the SkyNet T-800 series and it's got basically everything. Including AI integration with ChatGPT and um all that jazz. Would you like me to get some prices for you?",
                label="Text to synthesize (max chars 300)",
                max_lines=5,
                elem_id="main_textbox"
            )

            gr.Markdown("**Event Tags** (click to insert):")
            with gr.Row():
                for tag in EVENT_TAGS:
                    btn = gr.Button(tag, size="sm")
                    btn.click(
                        fn=lambda t, curr_text: curr_text + " " + t,
                        inputs=[btn, text],
                        outputs=text,
                    )

            ref_wav = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Reference Audio File (optional - for voice cloning)",
            )

            run_btn = gr.Button("Generate ⚡", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")

            with gr.Accordion("Advanced Options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 2.0, step=.05, label="Temperature", value=0.8)
                top_p = gr.Slider(0.00, 1.00, step=0.01, label="Top P", value=0.95)
                top_k = gr.Slider(0, 1000, step=10, label="Top K", value=1000)
                repetition_penalty = gr.Slider(1.00, 2.00, step=0.05, label="Repetition Penalty", value=1.2)
                min_p = gr.Slider(0.00, 1.00, step=0.01, label="Min P (0 to disable)", value=0.00)
                norm_loudness = gr.Checkbox(value=True, label="Normalize Loudness (-27 LUFS)")

    run_btn.click(
        fn=generate,
        inputs=[
            text,
            ref_wav,
            temp,
            seed_num,
            min_p,
            top_p,
            top_k,
            repetition_penalty,
            norm_loudness,
        ],
        outputs=audio_output,
    )

if __name__ == "__main__":
    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(server_name="127.0.0.1", server_port=7862)
