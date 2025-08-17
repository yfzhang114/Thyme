import torch
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoConfig
from .utils import run_evaluation

# --- Configuration ---
# Replace with your actual model path (local or HF Hub) and image/video folders
MODEL_PATH = "yifanzhang114/Thyme"

# --- 1. Load Model and Processor ---
print("Loading model and processor...")
config = AutoConfig.from_pretrained(MODEL_PATH)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto", # Automatically distributes model across available GPUs
    attn_implementation="sdpa",
    config=config
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("Model and processor loaded.")


# --- 3. Prepare Input Data (Example) ---
# Example question and responses (replace with your actual data)
# Example case (sampled from MME-RealWorld-Lite bench)
question_text = ("Question: What is the plate number of the blue car in the picture?\nOptions:\n"
                 "A. S OT 911\n"
                 "B. S TQ 119\n"
                 "C. S QT 911\n"
                 "D. B QT 119\n"
                 "E. This image doesn't feature the plate number.\n"
                 "Please select the correct answer from the options above.")
image_path = "eval/17127.jpg"

# # # --- 4. Generate Model Output ---
print("Generating model output...")
final_assistant_response, final_answer = run_evaluation(question_text, image_path, model, processor)
    
