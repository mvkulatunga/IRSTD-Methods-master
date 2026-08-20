import argparse
import numpy as np
from tqdm import tqdm

# NOTE: Change 'dataset' to the actual name of your Python file containing MOCIDDataset
from mocid import MOCIDDataset


def calculate_dataset_stats(annotations_file, img_size=(512, 512)):
    """
    Iterates through the dataset to calculate the average bounding box size.
    This calculates the 'C' constant required for the NWDLoss function.
    """
    print(f"Initializing dataset from: {annotations_file}")
    print(f"Target image size: {img_size}")

    # We set is_train=False to disable random flips, ensuring consistent evaluation.
    # (Though flips don't change width/height, it's best practice).
    try:
        dataset = MOCIDDataset(
            annotations_file=annotations_file, img_size=img_size, is_train=False
        )
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    if len(dataset) == 0:
        print("Dataset is empty. Please check your annotations file.")
        return

    widths = []
    heights = []
    absolute_sizes = []

    print("Analyzing bounding boxes...")
    for i in tqdm(range(len(dataset)), desc="Processing samples"):
        # We only need the target, so we ignore the images tensor
        _, target = dataset[i]

        # target["boxes"] is shape (1, 4) -> [xmin, ymin, xmax, ymax]
        box = target["boxes"][0].numpy()
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]

        # Calculate dimensions
        w = xmax - xmin
        h = ymax - ymin

        # The paper defines 'absolute size' as the square root of the area
        abs_size = np.sqrt(w * h)

        widths.append(w)
        heights.append(h)
        absolute_sizes.append(abs_size)

    avg_w = np.mean(widths)
    avg_h = np.mean(heights)
    avg_abs_size = np.mean(absolute_sizes)

    median_abs_size = np.median(absolute_sizes)
    min_size = np.min(absolute_sizes)
    max_size = np.max(absolute_sizes)

    print("\n" + "=" * 40)
    print(" 📊 DATASET BOUNDING BOX STATISTICS")
    print("=" * 40)
    print(f"Total Samples Analyzed : {len(dataset)}")
    print(f"Average Width (pixels) : {avg_w:.2f}")
    print(f"Average Height (pixels): {avg_h:.2f}")
    print("-" * 40)
    print(f"Minimum Absolute Size  : {min_size:.2f}")
    print(f"Maximum Absolute Size  : {max_size:.2f}")
    print(f"Median Absolute Size   : {median_abs_size:.2f}")
    print("=" * 40)
    print(f"🎯 RECOMMENDED NWD CONSTANT (C): {avg_abs_size:.2f}")
    print("=" * 40)
    print(
        f"\nUsage in YOLOLoss:\nself.nwd_loss = NWDLoss(reduction='none', C={avg_abs_size:.2f})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate NWD 'C' constant from MOCIDDataset"
    )
    parser.add_argument(
        "--annotations", type=str, required=True, help="Path to your annotations file"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=512,
        help="Image size used in the dataset (default: 512)",
    )

    args = parser.parse_args()

    calculate_dataset_stats(
        annotations_file=args.annotations, img_size=(args.img_size, args.img_size)
    )
