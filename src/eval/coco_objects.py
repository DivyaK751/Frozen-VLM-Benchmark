"""COCO's 80 object categories grouped into its 12 supercategories (from the COCO annotations).
POPE asks "Is there a <object> in the image?" with COCO category names, so this lets us break
hallucination down by object type."""

SUPERCATEGORIES = {
    "person": ["person"],
    "vehicle": ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
    "outdoor": ["traffic light", "fire hydrant", "stop sign", "parking meter", "bench"],
    "animal": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
    "accessory": ["backpack", "umbrella", "handbag", "tie", "suitcase"],
    "sports": ["frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
               "skateboard", "surfboard", "tennis racket"],
    "kitchen": ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl"],
    "food": ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"],
    "furniture": ["chair", "couch", "potted plant", "bed", "dining table", "toilet"],
    "electronic": ["tv", "laptop", "mouse", "remote", "keyboard", "cell phone"],
    "appliance": ["microwave", "oven", "toaster", "sink", "refrigerator"],
    "indoor": ["book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"],
}

OBJECT_TO_SUPERCATEGORY = {obj: sup for sup, objs in SUPERCATEGORIES.items() for obj in objs}


def supercategory(obj):
    return OBJECT_TO_SUPERCATEGORY.get(obj, "unknown")
