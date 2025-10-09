from cleanvision import Imagelab
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

# Cleanvision result directory
save_path = "./results"

imagelab = Imagelab.load(save_path)

exact_duplicate_sets = imagelab.info["exact_duplicates"]["sets"]

print("Loaded the following number of exact duplicate sets:")
print(len(exact_duplicate_sets))

def check_duplicate_images(image_pairs):
    """
    Check if pairs of images are identical.
    
    Parameters:
    image_pairs (list): A list of [N, 2] image path pairs to compare
                       e.g., [['path1.jpg', 'path2.jpg'], ['path3.jpg', 'path4.jpg']]
    
    Returns:
    list: A list of dictionaries containing comparison results for each pair
    """
    transform = transforms.ToTensor()
    results = []
    
    for idx, (path1, path2) in tqdm(enumerate(image_pairs), total=len(image_pairs), desc="Checking image pairs"):
        result = {
            'pair_index': idx,
            'image1': path1,
            'image2': path2,
            'identical': False,
            'error': None
        }
        
        try:
            # Load images
            img1 = Image.open(path1)
            img2 = Image.open(path2)
            
            # Convert to tensors
            tensor1 = transform(img1)
            tensor2 = transform(img2)
            
            result['shape1'] = tuple(tensor1.shape)
            result['shape2'] = tuple(tensor2.shape)
            
            # Check if identical
            if tensor1.shape != tensor2.shape:
                result['identical'] = False
                result['error'] = 'Different dimensions'
            else:
                result['identical'] = torch.equal(tensor1, tensor2)
                
                # Calculate differences if not identical
                if not result['identical']:
                    diff = torch.abs(tensor1 - tensor2)
                    result['max_diff'] = torch.max(diff).item()
                    result['mean_diff'] = torch.mean(diff).item()
                    
        except Exception as e:
            result['error'] = str(e)
        
        results.append(result)
    
    return results

image_pairs = exact_duplicate_sets[10000:20000]

results = check_duplicate_images(image_pairs)

# Print results
for result in results:
    # print(f"\nPair {result['pair_index']}: {result['image1']} vs {result['image2']}")
    if result['error']:
        print(f"  Error: {result['error']}")
    else:
        # print(f"  Shapes: {result['shape1']} vs {result['shape2']}")
        # print(f"  Identical: {result['identical']}")
        if not result['identical'] and 'max_diff' in result:
            print(f"  Max difference: {result['max_diff']:.6f}")
            print(f"  Mean difference: {result['mean_diff']:.6f}")

# Count duplicates
duplicates = sum(1 for r in results if r['identical'])
print(f"\n{duplicates} out of {len(results)} pairs are identical")