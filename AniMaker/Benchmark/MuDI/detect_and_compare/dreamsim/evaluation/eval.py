import os

from dreamsim import dreamsim
from PIL import Image
import pathlib

device = "cuda"
model, preprocess = dreamsim(pretrained=True, device=device)

img1 = preprocess(Image.open("Benchmark/MuDI/detect_and_compare/examples/images/mixed1.jpg")).to(device)
img2 = preprocess(Image.open("Benchmark/MuDI/detect_and_compare/examples/images/mixed2.jpg")).to(device)
distance = model(img1, img2) # The model takes an RGB image from [0, 1], size batch_sizex3x224x224

print(distance)

# Save the result to the specified file
result_path = "Benchmark/results/ds.txt"
# Create directory if it doesn't exist
os.makedirs(os.path.dirname(result_path), exist_ok=True)
# Write the result to the file
with open(result_path, 'w') as f:
    f.write(f"Distance: {distance.item()}\n")

print(f"Result saved to {result_path}")