import torch
import clip
from PIL import Image

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load and preprocess the image
image_path = "path_to_your_image.jpg"  # Change this to the path of your image
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

# Prepare text descriptions
text_descriptions = ["a photo of a tiger", "a photo of a random object"]
text_tokens = clip.tokenize(text_descriptions).to(device)

# Compute image and text features
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text_tokens)

# Compare embeddings
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
probabilities = similarity[0].cpu().numpy()

# Determine if there is a tiger in the image
print(f"Probability of being a tiger: {probabilities[0]:.2f}")
print(f"Probability of being a random object: {probabilities[1]:.2f}")

if probabilities[0] > probabilities[1]:
    print("The image likely contains a tiger.")
else:
    print("The image likely does not contain a tiger.")

