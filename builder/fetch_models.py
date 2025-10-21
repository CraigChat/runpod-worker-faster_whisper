from faster_whisper.utils import download_model
import deepmultilingualpunctuation

model_names = [
    "tiny",
    "base",
    "small",
    "medium",
    # "large-v1",
    # "large-v2",
    # "large-v3",
    # "distil-large-v2",
    # "distil-large-v3",
    "turbo",
]


def download_model_weights(selected_model):
    """
    Download model weights.
    """
    print(f"Downloading {selected_model}...")
    download_model(selected_model, cache_dir=None)
    print(f"Finished downloading {selected_model}.")


print("Downloading punctuation model...")
model = deepmultilingualpunctuation.PunctuationModel()
result = model.restore_punctuation("das , ist fies ")
print("Finished downloading punctuation model.")

# Loop through models sequentially
for model_name in model_names:
    download_model_weights(model_name)

print("Finished downloading all models.")
