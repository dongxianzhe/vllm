import soundfile as sf

from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

MODEL_PATH = "/mnt/cephfs/user_xianzhedong/models/Qwen3-Omni-30B-A3B-Captioner"
DATASET_PATH = "/mnt/cephfs/user_xianzhedong/datasets/caption2.mp3"

model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)

processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": DATASET_PATH},
        ],
    },
]

# Preparation for inference
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, _, _ = process_mm_info(conversation, use_audio_in_video=False)
inputs = processor(text=text, 
                   audio=audios,
                   return_tensors="pt", 
                   padding=True, 
                   use_audio_in_video=False)
inputs = inputs.to(model.device).to(model.dtype)

# Inference: Generation of the output text and audio
text_ids, audio = model.generate(**inputs, 
                                 thinker_return_dict_in_generate=True)

text = processor.batch_decode(text_ids.sequences[:, inputs["input_ids"].shape[1] :],
                              skip_special_tokens=True,
                              clean_up_tokenization_spaces=False)
print(text)
