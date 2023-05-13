from dataclasses import dataclass
import functools
import os
import argparse

import gradio as gr

# from src_files.modules import shared
from src_files.helper_functions.bn_fusion import fuse_bn_recursively

import warnings

from src_files.interface import Infer, MODELS_NAME_WITH_FILE_PATH

warnings.filterwarnings("ignore")

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def make_args():
    parser = argparse.ArgumentParser(description='ML-Danbooru Demo')
    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--class_map', type=str, default='./class.json')
    parser.add_argument('--model_name', default='caformer_m36')
    parser.add_argument('--num_classes', default=12547)
    parser.add_argument('--image_size', default=448, type=int,
                        metavar='N', help='input image size')
    parser.add_argument('--thr', default=0.75, type=float,
                        metavar='N', help='threshold value')
    parser.add_argument('--keep_ratio', type=str2bool, default=False)

    # ML-Decoder
    parser.add_argument('--use_ml_decoder', default=0, type=int)
    parser.add_argument('--fp16', action="store_true", default=False)
    parser.add_argument('--ema', action="store_true", default=False)

    parser.add_argument('--frelu', type=str2bool, default=True)
    parser.add_argument('--xformers', type=str2bool, default=False)

    # CAFormer
    parser.add_argument('--decoder_embedding', default=384, type=int)
    parser.add_argument('--num_layers_decoder', default=4, type=int)
    parser.add_argument('--num_head_decoder', default=8, type=int)
    parser.add_argument('--num_queries', default=80, type=int)
    parser.add_argument('--scale_skip', default=1, type=int)

    parser.add_argument('--out_type', type=str, default='json')

    args = parser.parse_args()

    # python demo_ca.py --data imgs/t1.jpg --model_name caformer_m36 --ckpt ckpt/caformer_m36-2-20000.ckpt --thr 0.7 --image_size 448

    args.data = "imgs/girl.jpg"
    args.data = "imgs/"
    args.model_name = "caformer_m36"
    args.ckpt = "ckpt/ml_caformer_m36_fp16_dec-5-97527.ckpt"
    # args.fp16=True
    return args


@dataclass
class ModelArgsStruct:
    model_name: str = "caformer_m36"  # string, name of the model. e.g. caformer_m36 ,tresnet_d. e.g. tresnet_d_fp
    # string, path to ckpt. e.g. ckpt/ml_caformer_m36_fp16_dec-5-97527.ckpt
    ckpt: str = MODELS_NAME_WITH_FILE_PATH['caformer_m36']
    input_path: str = ""  # string, path to input image or directory. e.g. img/t1.jpg, img/t2.jpg,...,
    input_img_data: bytes = None  # bytes, input image data. e.g. b'\xff\xd8\xff\xe0\x00\x10
    image_size: int = 448
    class_map_file_path: str = "./class.json"  # string, path to ./class.json
    threshold: float = 0.75  # float, objectness score threshold. e.g. 0.7, 0.8,..., 1.0.
    keep_ratio: bool = False  # bool, whether to keep aspect ratio. e.g. True, False,
    fp16: bool = False  # bool, whether to use fp16. e.g. True, False,
    ema: bool = False  # bool, whether to use ema. e.g. True, False,
    use_ml_decoder: bool = False  # bool, whether to use ml decoder. e.g. True, False,
    space: bool = True  # 'Use Space Instead Of _'
    escape: bool = True  # 'Use Text Escape'
    conf: bool = False  # 'With confidence'
    #    gr_space = gr.Checkbox(value=True, label='Use Space Instead Of _')
    #                     gr_escape = gr.Checkbox(value=True, label='Use Text Escape')
    #                     gr_conf = gr.Checkbox(value=False, label='With confidence')
    share: bool = False


DESCRIPTION = """
ML-Danbooru测试
from https://github.com/7eu7d7/ML-Danbooru-webui

"""
TITLE = "ML-Danbooru-web demo "


def sample_gui_start(model_args: ModelArgsStruct):
    infer = Infer()
    func = functools.partial(
        # predict,
        infer.infer_one,

        image_size=model_args.image_size,
        keep_ratio=model_args.keep_ratio,
        model_name=model_args.model_name,
        space=model_args.space,
        escape=model_args.escape,
        conf=model_args.conf,

    )
    #  def infer_one(self, img: Image.Image, threshold: float, image_size: int,
    #                 keep_ratio: bool, model_name: str, space: bool, escape: bool, conf: bool):

    score_slider_step = 0.01
    gr.Interface(
        fn=func,
        inputs=[
            gr.Image(type="pil", label="Input"),
            # gr.Radio(["SwinV2", "ConvNext", "ViT"], value="SwinV2", label="Model"),
            gr.Slider(
                0,
                1,
                step=score_slider_step,
                value=model_args.threshold,
                label="  Threshold",
            ),

        ],
        # outputs=[gr_output_text, gr_tags],
        outputs=[
            gr.Textbox(label="Output output_text"),
            gr.Textbox(label="Output tags"),
            # gr.Label(label="output_text"),
            # gr.Label(label="tags"),

        ],
        # examples=[["power.jpg", "SwinV2", 0.35, 0.85]],
        title=TITLE,
        description=DESCRIPTION,
        allow_flagging="never",
    ).launch(
        enable_queue=True,
        #  server_name="0.0.0.0",
        share=model_args.share,
        # server_port=server_port,
        # #    auth=("username", "password"),
    )


def gui_start(model_args: ModelArgsStruct):
    infer = Infer()
    func = functools.partial(
        # predict,
        infer.infer_one,

        # image_size=model_args.image_size,
        keep_ratio=model_args.keep_ratio,
        model_name=model_args.model_name,
        space=model_args.space,
        escape=model_args.escape,
        conf=model_args.conf,

    )
    score_slider_step = 0.01
    #  def infer_one(self, img: Image.Image, threshold: float, image_size: int,
    #                 keep_ratio: bool, model_name: str, space: bool, escape: bool, conf: bool):
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                gr_input_image = gr.Image(type='pil', label='Original Image')
                with gr.Row():
                    gr_threshold = gr.Slider(0.0, 1.0, 0.7, label='Tagging Confidence Threshold')
                    gr_image_size = gr.Slider(128, 960, 448, step=32, label='Image for Recognition')
                    # gr_keep_ratio = gr.Checkbox(value=False, label='Keep the Ratio')
                # with gr.Row():
                #     gr_model = gr.Dropdown(infer.MODELS, value=infer.DEFAULT_MODEL, label='Model')
                # with gr.Row():
                #     gr_space = gr.Checkbox(value=True, label='Use Space Instead Of _')
                #     gr_escape = gr.Checkbox(value=True, label='Use Text Escape')
                #     gr_conf = gr.Checkbox(value=False, label='With confidence')

                with gr.Row():
                    gr_btn_submit = gr.Button(value='submit', variant='primary')
                    # gr_btn_unload = gr.Button(value='Unload')

            with gr.Column():
                # with gr.Tabs():
                with gr.Tab("Exported Text"):
                    gr_output_text = gr.TextArea(label='Exported Text', lines=10)
                with gr.Tab("Tags"):
                    gr_tags = gr.Label(label='Tags')

                    gr_info = gr.Text(value="", show_label=False)

                # gr_btn_submit.click(
                #     infer.infer_one,
                #     inputs=[
                #         gr_input_image, gr_threshold, gr_image_size,
                #         gr_keep_ratio, gr_model,
                #         gr_space, gr_escape, gr_conf
                #     ],
                #     outputs=[gr_output_text, gr_tags],
                # )
                #  gr.Interface(
                # fn = func,
                # inputs=[
                #     gr.Image(type="pil", label="Input"),
                #     gr.Slider(
                #         0,
                #         1,
                #         step=score_slider_step,
                #         value=model_args.threshold,
                #         label="  Threshold",
                #     ),
                # ],
                # # outputs=[gr_output_text, gr_tags],
                # outputs=[
                #     gr.Textbox(label="Output output_text"),
                #     gr.Textbox(label="Output tags"),
                #     # gr.Label(label="output_text"),
                #     # gr.Label(label="tags"),
                #
                # ],
        gr_btn_submit.click(
            func,
            inputs=[
                gr_input_image, gr_threshold, gr_image_size,
                # gr_keep_ratio, gr_model,
                # gr_space, gr_escape, gr_conf
            ],
            outputs=[gr_output_text, gr_tags],
            api_name="run_pre"
        )

    demo.launch().launch(
        enable_queue=True,
        #  server_name="0.0.0.0",
        share=model_args.share,
        # server_port=server_port,
        # #    auth=("username", "password"),
        # share=args.share,
    )


if __name__ == '__main__':
    # args = make_args()
    model_args = ModelArgsStruct()
    gui_start(model_args)
    # return 0
