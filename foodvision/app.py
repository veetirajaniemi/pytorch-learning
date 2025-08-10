
import gradio as gr
import os
import torch
from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

with open("class_names.txt", "r") as f:
  class_names = [food_name.strip() for food_name in f.readlines()]

effnetb2, effnetb2_transforms = create_effnetb2_model(
    num_classes=101
)

effnetb2.load_state_dict(
    torch.load(
        f="09_pretrained_effnetb2_food101.pth",
        map_location=torch.device("cpu") # load model to the CPU
    )
)

def predict(img) -> Tuple[Dict, float]:
  start_time = timer()

  img = effnetb2_transforms(img).unsqueeze(0) # add batch dimension
  effnetb2.eval()
  with torch.inference_mode():
    pred_probs = torch.softmax(effnetb2(img), dim=1) # get prediction probs

  pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

  end_time = timer()
  pred_time = round(end_time - start_time, 4)

  return pred_labels_and_probs, pred_time

title = "FoodVision"
description = "An EfficientNetB2 feature extractor model classifying between 101 classes of food."
article = "Created at [09. PyTorch Model Deployment](https://github.com/veetirajaniemi/pytorch-learning/blob/main/09_pytorch_model_deployment.ipynb)."

example_list = [["examples/" + example] for example in os.listdir("examples")]

demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=10, label="Predictions"),
                             gr.Number(label="Prediction time (s)")],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)

if __name__ == "__main__":
  demo.launch(share=True)
