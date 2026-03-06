# test_hybrid_real.py
from inference.predictor import DrawingPredictor
import os
import random

predictor = DrawingPredictor('checkpoints/best_model.pt')

# All 29 TRAINED classes
trained_classes = [
    'apple', 'banana', 'bicycle', 'bird', 'bus', 'cake', 'car', 'cat',
    'chair', 'clock', 'cloud', 'diamond', 'dog', 'door', 'eyeglasses',
    'fish', 'flower', 'hexagon', 'horse', 'key', 'ladder', 'lightning',
    'mountain', 'pizza', 'scissors', 'square', 'table', 'triangle', 'truck'
]

# NEW classes (not in training)
new_classes = ['airplane', 'book', 'tree', 'house', 'umbrella', 'guitar', 'moon', 'star']

print("\n" + "="*60)
print("HYBRID LEARNING TEST - Real QuickDraw Data")
print("="*60)

print(f"\nTrained classes ({len(trained_classes)}): {trained_classes}")
print(f"New classes ({len(new_classes)}): {new_classes}\n")

# Test 1: Base classes still work (sample 8 to get broader coverage)
print("-"*60)
print("TEST 1: Base classes (should work perfectly)")
print("-"*60)

correct_base = 0
total_base = 0
sample_size = min(8, len(trained_classes))
for class_name in random.sample(trained_classes, sample_size):
    test_img = f'data/quickdraw/test/{class_name}/0001.png'
    if os.path.exists(test_img):
        result = predictor.predict(test_img)
        is_correct = result['class'] == class_name
        correct_base += int(is_correct)
        total_base += 1
        symbol = "+" if is_correct else "-"
        print(f"{symbol} {class_name:12} -> {result['class']:12} ({result['confidence']:.1%})")

if total_base > 0:
    print(f"\nBase class accuracy: {correct_base}/{total_base} = {correct_base/total_base:.1%}")

# Test 2: NEW classes (model has NEVER seen these!)
print("\n" + "-"*60)
print("TEST 2: NEW classes BEFORE teaching (should fail/guess)")
print("-"*60)

for class_name in random.sample(new_classes, 3):
    test_img = f'custom_drawings/{class_name}/0010.png'
    if os.path.exists(test_img):
        result = predictor.predict(test_img)
        print(f"? {class_name:10} -> {result['class']:10} ({result['confidence']:.1%})")
        print(f"  Top-3: {[p['class'] for p in result['top_k']]}")

# Test 3: Teach the model with just 5 examples!
print("\n" + "-"*60)
print("TEST 3: Teaching new classes (5-shot learning)")
print("-"*60)

n_examples = 5  # Only 5 examples — the whole point of few-shot learning!
for class_name in new_classes:
    # Use images 0-4 for teaching
    examples = [
        f'custom_drawings/{class_name}/{i:04d}.png'
        for i in range(n_examples)
        if os.path.exists(f'custom_drawings/{class_name}/{i:04d}.png')
    ]
    
    if len(examples) >= 3:
        success = predictor.add_custom_class(class_name, examples)
        status = "OK" if success else "FAIL"
        print(f"[{status}] Taught '{class_name}' with {len(examples)} examples")

# Test 4: NEW classes AFTER teaching (should work!)
print("\n" + "-"*60)
print("TEST 4: NEW classes AFTER teaching (should work!)")
print("-"*60)

correct_count = 0
total_count = 0

for class_name in new_classes:
    # Test on images the model has NEVER seen (images 20-29, well separated from 0-4)
    for i in range(20, 30):
        test_img = f'custom_drawings/{class_name}/{i:04d}.png'
        if os.path.exists(test_img):
            result = predictor.predict(test_img)
            correct = result['class'] == class_name
            if correct:
                correct_count += 1
            total_count += 1
            
            symbol = "+" if correct else "-"
            print(f"{symbol} {class_name:10} -> {result['class']:10} ({result['confidence']:.1%})")

# Summary
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

classes = predictor.get_available_classes()
print(f"Total classes available: {classes['total']}")
print(f"  Base (trained): {len(classes['base'])}")
print(f"  Custom (few-shot): {len(classes['custom'])}")

if total_base > 0:
    print(f"\nBase class accuracy:     {correct_base}/{total_base} = {correct_base/total_base:.1%}")
if total_count > 0:
    print(f"Few-shot accuracy:       {correct_count}/{total_count} = {correct_count/total_count:.1%}")
    print(f"\nThe model learned {len(new_classes)} new classes with just {n_examples} examples each.")
