import os
import json
import datetime

import gradio as gr
import torch
import numpy as np
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import cv2
from torchvision.io import read_image

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_sam():
    sam = sam_model_registry["vit_b"]()
    transformers = ResizeLongestSide(sam.image_encoder.img_size)
    state_dict = torch.load(
        "./pytorch_model.bin", map_location="cpu", weights_only=True
    )
    # remove the prefix "model." from state_dict keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_key = k[6:]
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    sam.load_state_dict(new_state_dict)
    sam.to(device=device)
    return sam.eval(), transformers


@torch.inference_mode()
def get_img_embeddings(sam, transformers, image):
    original_size = image.shape[1], image.shape[2]
    input_image = transformers.apply_image(image.permute(1, 2, 0).numpy())
    input_image = (
        torch.as_tensor(input_image).permute(2, 0, 1).unsqueeze(0).to(device=device)
    )
    input_size = input_image.shape[2:]
    with torch.no_grad():
        input_image = sam.preprocess(input_image)
        image_embeddings = sam.image_encoder(input_image)
    return image_embeddings, input_size, original_size


@torch.inference_mode()
def get_predit(
    sam, transformers, image_embeddings, input_size, original_size, box=None
):
    if box is not None:
        input_box = transformers.apply_boxes(np.array(box), original_size)
        input_box = torch.as_tensor(input_box, device=sam.device).unsqueeze(0)

    sparese_embeddings, dense_embeddings = sam.prompt_encoder(
        points=None, masks=None, boxes=input_box if box is not None else None
    )
    mask, _ = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparese_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    mask = sam.postprocess_masks(mask, input_size, original_size)[0, 0]
    mask = torch.sigmoid(mask).cpu()

    if box is None:
        hist = torch.histc(mask, bins=100, min=0.1, max=1.0)
    else:
        hist = torch.histc(
            mask[box[1] : box[3], box[0] : box[2]], bins=100, min=0.1, max=1.0
        )
    return mask, hist


def mask2poly(mask):
    counters, _ = cv2.findContours(
        mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
    )
    if counters is None or len(counters) == 0:
        return None
    else:
        max_counter = max(counters, key=lambda x: cv2.contourArea(x))
        perimeter = cv2.arcLength(max_counter, True)
        epsilon = 0.001 * perimeter
        max_counter = cv2.approxPolyDP(max_counter, epsilon, True)
        return max_counter


def scan_images(data_dir="data"):
    exts = [".jpg", ".jpeg", ".png", ".bmp"]
    result = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in exts:
                img_path = os.path.join(root, file)
                result.append(img_path)
    return result


sam, transformers = get_sam()
img_paths = scan_images()
selected_img_path = None
current_img_path = None
image = None
image_embedding, input_size, original_size = (None, None, None)
box = None
polygons = None
selected_polygons = None
info_top_choices = None
predit = None
hist = None
threshold = None
first_point = None
sencond_point = None


def draw():
    global image, predit, box, polygons, selected_polygons, threshold
    if image is None:
        return None
    img = image.permute(1, 2, 0).numpy().copy()
    if selected_polygons is not None and len(selected_polygons) > 0:
        for idx in selected_polygons:
            label, points = polygons[idx]
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            if label == "2":
                cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            elif label == "3":
                cv2.polylines(
                    img, [pts], isClosed=True, color=(0, 255, 255), thickness=2
                )
            elif label == "4":
                cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
    elif box is not None:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 255, 0), thickness=2)
        if threshold is None:
            heatmap = (predit.numpy() * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            img = cv2.addWeighted(img.astype(np.uint8), 0.7, heatmap, 0.3, 0)
        else:
            mask = (predit > threshold).numpy().astype(np.uint8) * 255
            counters = mask2poly(mask)
            if counters is not None:
                cv2.polylines(
                    img, [counters], isClosed=True, color=(255, 255, 0), thickness=2
                )
    else:
        heatmap = (predit.numpy() * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        img = cv2.addWeighted(img.astype(np.uint8), 0.7, heatmap, 0.3, 0)
        if first_point is not None:
            x, y = first_point
            cv2.circle(img, (x, y), radius=5, color=(255, 255, 0), thickness=-1)
    return img


def get_info_bottom_ui():
    choices = []
    for path in img_paths:
        label = f"{path[5:]}"
        choices.append(label)
    radio = gr.Radio(label="Select Image", choices=choices, interactive=True)
    return radio


with gr.Blocks(
    css="""
.main-row {height: calc(100vh - 32px);}
.main-area {height: 100%;}
.side-area {height: 100%; display: flex; flex-direction: column;}
.side-top, .side-bottom {flex: 1 1 0; min-height: 0;}
.gr-radio .gr-form {display: flex; flex-direction: column;}
"""
) as demo:
    with gr.Row(elem_classes="main-row"):
        with gr.Column(scale=10, elem_classes="main-area"):
            image_display = gr.Image(interactive=False, label="Image Display")

        with gr.Column(scale=2, elem_classes="side-area"):
            with gr.Column(elem_classes="side-top"):
                with gr.Group():
                    select_all = gr.Button("Select/Deselect All")
                    del_btn = gr.Button(
                        "Delete Selected", elem_id="del-btn", variant="stop"
                    )
                    info_top = gr.CheckboxGroup(label="Annotations", interactive=True)
                with gr.Group():
                    hist_display = gr.Image(
                        interactive=False, label="Hist & Threshold", height=100
                    )
                    with gr.Row():
                        label_selector = gr.Number(
                            label="Label",
                            value=2,
                            minimum=2,
                            maximum=4,
                            step=1,
                            interactive=True,
                            show_label=True,
                        )
                        add_btn = gr.Button("Add Annotation")
                save_btn = gr.Button(
                    "Save Annotations", elem_id="save-btn", variant="primary"
                )
            with gr.Column(elem_classes="side-bottom"):
                with gr.Group():
                    info_bottom = get_info_bottom_ui()
                    refresh_btn = gr.Button("Refresh Image List", elem_id="refresh-btn")

    def on_select_image_change(radio_label):
        if radio_label is None:
            return (
                None,
                gr.CheckboxGroup([], label="Annotations", value=[], interactive=True),
                None,
            )
        global selected_img_path, current_img_path, image, image_embedding, input_size, original_size, predit, hist, box, polygons, selected_polygons, info_top_choices
        selected_img_path = radio_label
        full_path = os.path.join("data", selected_img_path)
        current_img_path = full_path
        image = read_image(full_path)
        image_embedding, input_size, original_size = get_img_embeddings(
            sam, transformers, image
        )
        predit, hist = get_predit(
            sam, transformers, image_embedding, input_size, original_size, box=None
        )
        box = None
        json_path = os.path.splitext(full_path)[0] + ".json"
        polygons = []
        selected_polygons = []
        info_top_choices = []
        info_top_selected = []
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                annotations = json.load(f)
                i = 0
                for shape in annotations["shapes"]:
                    if shape["shape_type"] == "polygon":
                        label_str = str(i) + ": " + shape["label"]
                        polygons.append([shape["label"], shape["points"]])
                        info_top_choices.append(label_str)
                        info_top_selected.append(label_str)
                        i += 1
            selected_polygons = list(range(len(info_top_choices)))
        if not info_top_choices:
            info_top_choices = []
            info_top_selected = []
            selected_polygons = []
        image_display_val = draw()
        hist_img = hist_to_img()
        return (
            image_display_val,
            gr.CheckboxGroup(
                info_top_choices,
                label="Annotations",
                value=info_top_selected,
                interactive=True,
            ),
            None if selected_polygons else hist_img,
        )

    info_bottom.change(
        on_select_image_change,
        inputs=[info_bottom],
        outputs=[image_display, info_top, hist_display],
    )

    def on_select_label_change(selected_labels):
        global selected_polygons, hist
        selected_polygons = []
        for label in selected_labels:
            idx = int(label.split(": ")[0])
            selected_polygons.append(idx)
        if len(selected_polygons) == 0:
            return draw(), hist_to_img()
        else:
            return draw(), None

    info_top.change(
        on_select_label_change, inputs=[info_top], outputs=[image_display, hist_display]
    )

    def on_image_click(event: gr.SelectData):
        global first_point, sencond_point, box, predit, hist
        if first_point is None:
            first_point = event.index
            box = None
            predit, hist = get_predit(
                sam, transformers, image_embedding, input_size, original_size, box=None
            )
            return draw(), None
        else:
            sencond_point = event.index
            if (
                first_point[0] <= sencond_point[0]
                and first_point[1] <= sencond_point[1]
            ):
                x1, y1 = first_point
                x2, y2 = sencond_point
            else:
                x2, y2 = first_point
                x1, y1 = sencond_point
            box = [x1, y1, x2, y2]
            predit, hist = get_predit(
                sam, transformers, image_embedding, input_size, original_size, box=box
            )
            first_point = None
            sencond_point = None
            return draw(), hist_to_img()

    image_display.select(on_image_click, outputs=[image_display, hist_display])

    def on_select_all():
        global info_top_choices, selected_polygons, hist
        if len(selected_polygons) < len(info_top_choices):
            selected_polygons = list(range(len(info_top_choices)))
            on_select_label_change(info_top_choices)
            return gr.update(value=info_top_choices), None
        else:
            selected_polygons = []
            on_select_label_change([])
            return gr.update(value=[]), hist_to_img()

    select_all.click(on_select_all, outputs=[info_top, hist_display])

    def hist_to_img():
        global hist, threshold
        hist_img = np.zeros((50, 300, 3), dtype=np.uint8)
        hmax = hist.max().item() if hist is not None else 1
        for i in range(100):
            v = int(hist[i].item() / hmax * 50) if hmax > 0 else 0
            cv2.line(hist_img, (i * 3, 50), (i * 3, 50 - v), (255, 255, 255), 3)
        hist_img = np.zeros((50, 300, 3), dtype=np.uint8)
        hmax = hist.max().item() if hist is not None else 1
        for i in range(100):
            v = int(hist[i].item() / hmax * 50) if hmax > 0 else 0
            cv2.line(hist_img, (i * 3, 50), (i * 3, 50 - v), (255, 255, 255), 3)
        if threshold is not None:
            x = int(threshold * 300)
            cv2.rectangle(hist_img, (x - 1, 0), (300, 49), (0, 128, 255), 2)
        return hist_img

    def on_hist_click(event: gr.SelectData):
        global threshold
        x = event.index[0]
        t = round(x / 300, 2)
        if t == threshold:
            threshold = None
        else:
            threshold = t
        return hist_to_img(), draw()

    hist_display.select(on_hist_click, outputs=[hist_display, image_display])

    def on_add_poly(label_value):
        global predit, threshold, polygons, info_top_choices, selected_polygons, box, first_point, sencond_point
        if threshold is None or predit is None or box is None:
            return gr.update(), draw()
        mask = (predit > threshold).numpy().astype(np.uint8) * 255
        counters = mask2poly(mask)
        if counters is not None:
            label_str = str(len(polygons)) + ": " + str(int(label_value))
            polygons.append([str(int(label_value)), counters])
            info_top_choices.append(label_str)
            selected_polygons = [len(info_top_choices) - 1]

            first_point = None
            sencond_point = None
            box = None
            predit, hist = get_predit(
                sam, transformers, image_embedding, input_size, original_size, box=None
            )
            return gr.update(choices=info_top_choices, value=info_top_choices), draw()
        return gr.update(), draw()

    add_btn.click(
        on_add_poly, inputs=[label_selector], outputs=[info_top, image_display]
    )

    def on_del_selected():
        global polygons, info_top_choices, selected_polygons
        if selected_polygons:
            for idx in sorted(selected_polygons, reverse=True):
                if idx < len(polygons):
                    polygons.pop(idx)
                if idx < len(info_top_choices):
                    info_top_choices.pop(idx)
            info_top_choices = [f"{i}: {p[0]}" for i, p in enumerate(polygons)]
            selected_polygons = []
            return gr.update(choices=info_top_choices, value=[]), draw()
        return gr.update(), draw()

    del_btn.click(on_del_selected, outputs=[info_top, image_display])

    def on_refresh_img_list():
        global img_paths, selected_img_path, current_img_path, image, image_embedding, input_size, original_size, box, polygons, selected_polygons, info_top_choices, predit, hist, threshold, first_point, sencond_point
        img_paths = scan_images()
        choices = []
        for idx, path in enumerate(img_paths):
            label = f"{path[5:]}"
            choices.append(label)
        still_exists = False
        if current_img_path is not None:
            for path in img_paths:
                if path == current_img_path:
                    still_exists = True
                    break
        if not still_exists:
            selected_img_path = None
            current_img_path = None
            image = None
            image_embedding, input_size, original_size = (None, None, None)
            box = None
            polygons = None
            selected_polygons = None
            info_top_choices = None
            predit = None
            hist = None
            threshold = None
            first_point = None
            sencond_point = None
            return gr.update(choices=choices, value=None)
        else:
            label = current_img_path[5:] if current_img_path else None
            return gr.update(choices=choices, value=label)

    refresh_btn.click(on_refresh_img_list, outputs=[info_bottom])

    def on_save_annotations():
        global current_img_path, polygons, info_top_choices
        if current_img_path is None:
            return
        json_path = os.path.splitext(current_img_path)[0] + ".json"
        if os.path.exists(json_path):
            ts = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
            backup_path = json_path.replace(".json", f".json.{ts}")
            os.rename(json_path, backup_path)
        shapes = []
        if polygons is not None and info_top_choices is not None:
            for _, (label, points) in enumerate(polygons):
                shape = {
                    "text": "",
                    "kie_linking": [],
                    "label": str(label),
                    "score": None,
                    "points": points[:, 0, :].tolist(),
                    "group_id": None,
                    "difficult": False,
                    "shape_type": "polygon",
                    "flags": {},
                    "attributes": {},
                }
                shapes.append(shape)
        json_data = {
            "version": "0.0.0",
            "flags": {},
            "shapes": shapes,
            "imagePath": os.path.basename(current_img_path),
            "imageData": None,
            "imageHeight": int(image.shape[1]) if image is not None else None,
            "imageWidth": int(image.shape[2]) if image is not None else None,
            "text": "",
            "description": "",
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

    save_btn.click(on_save_annotations)

demo.launch()
