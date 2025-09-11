import os
import torch
from IPython.display import clear_output
import numpy as np
from PIL import Image
from owl_dreamsim_utils import eval_with_dreamsim, eval_with_dinov2

cache_dir = 'Benchmark/MuDI/detect_and_compare/models'

evalulator = eval_with_dreamsim(cache_dir=cache_dir, device='cuda')
# evalulator = eval_with_dinov2(cache_dir=None, device='cuda')
clear_output()

# functions for D&C
def get_gt_matrix(query_dict):
    embs = query_dict['query_emb']
    n = len(embs)
    gt_matrix = torch.zeros(n, n)
    # Fill the matrix
    for i in range(n):
        for j in range(i, n):  # Only calculate for i <= j
            if i == j:
                # Diagonal: Mean of self-similarity
                similarity = (embs[i] / embs[i].norm(dim=-1, keepdim=True)).matmul((embs[i] / embs[i].norm(dim=-1, keepdim=True)).t())
                gt_matrix[i, j] = similarity.mean()
            else:
                # Off-diagonal: Mean of inter-group similarity
                inter_similarity = (embs[i] / embs[i].norm(dim=-1, keepdim=True)).matmul((embs[j] / embs[j].norm(dim=-1, keepdim=True)).t())
                mean_similarity = inter_similarity.mean()
                gt_matrix[i, j] = mean_similarity
                gt_matrix[j, i] = mean_similarity  # Assign to A[j, i] without recalculating
    gt_matrix = np.array(gt_matrix)
    return gt_matrix


def sort_by_max(A):
    sorted_rows = np.zeros_like(A)
    used_rows = []

    # Iterate over each column
    for i in range(A.shape[1]):
        # Find the maximum value in the i-th column that hasn't been used yet
        max_value = -np.inf
        max_index = -1
        for j in range(A.shape[0]):
            if j not in used_rows and A[j, i] > max_value:
                max_value = A[j, i]
                max_index = j

        # Add the row with the maximum value to the sorted array
        sorted_rows[i] = A[max_index]
        used_rows.append(max_index)
    return sorted_rows


def gt_distance(scores, gt_matrix, ord=None):
    if len(scores) != len(gt_matrix):
        print(f'Count:{len(scores)}')
        return 1.
    # scores = [x.mean(-1).tolist() for x in np.array(scores)] # 2x2
    # scores = np.array(scores).mean(-1)
    tmp = []
    for bbox_score in scores:
        per_bbox = []
        for ref in bbox_score:
            per_bbox.append(np.array(ref).mean())
        tmp.append(per_bbox)
    
    # scores = [np.array(x).mean(-1) for x in scores]
    scores = np.array(tmp)
    scores = sort_by_max(scores)
    return np.linalg.norm(scores - gt_matrix, ord=ord)


# Create function to save results
def save_results(results, filepath):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        for image_name, score in results:
            f.write(f"{image_name}: {score}\n")


query_dict = {
    'query_name': ["monster toy", "robot toy"],
    'query_path': ["examples/references/monster_toy",
                    "examples/references/robot_toy"]
}

# Collect results in a list
results = []

# Test images and collect results
test_images = [
    ("examples/images/good.jpg", "good.jpg"),
    ("examples/images/mixed1.jpg", "mixed1.jpg"),
    ("examples/images/single.jpg", "single.jpg"),
    ("examples/images/mixed2.jpg", "mixed2.jpg"),
    ("examples/images/count.jpg", "count.jpg"),
]

for img_path, img_name in test_images:
    image = Image.open(img_path)
    scores = evalulator.score(image, query_dict, threshold=0.4, return_round=False)
    gt_matrix = get_gt_matrix(query_dict)
    #image.resize((256, 256)).show()
    gt_score = 1 - gt_distance(scores, gt_matrix, ord=2)
    print(f"GT score: {gt_score:.2f}")
    results.append((img_name, f"{gt_score:.2f}"))

# Save all results to file
results_path = "Benchmark/results/dc-ds.txt"
save_results(results, results_path)
print(f"Results saved to {results_path}")