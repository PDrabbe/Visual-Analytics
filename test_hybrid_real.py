# test_hybrid_real.py
from inference.predictor import DrawingPredictor
import os
import random

predictor = DrawingPredictor('checkpoints/best_model.pt')

# Your 9 TRAINED classes
trained_classes = ['cat', 'dog', 'bird', 'fish', 'horse', 'apple', 'banana', 'cake', 'pizza']

# NEW classes (not in training)
new_classes = ['airplane', 'book', 'tree', 'house', 'umbrella', 'guitar', 'moon', 'star']

print("\n" + "="*60)
print("HYBRID LEARNING TEST - Real QuickDraw Data")
print("="*60)

print(f"\nTrained classes: {trained_classes}")
print(f"New classes: {new_classes}\n")

# Test 1: Base classes still work
print("-"*60)
print("TEST 1: Base classes (should work perfectly)")
print("-"*60)

for class_name in random.sample(trained_classes, 3):
    test_img = f'data/quickdraw/test/{class_name}/0001.png'
    if os.path.exists(test_img):
        result = predictor.predict(test_img)
        correct = "+" if result['class'] == class_name else "-"
        print(f"{correct} {class_name:10} -> {result['class']:10} ({result['confidence']:.1%})")

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

for class_name in new_classes:
    # Use first 5 images as examples
    examples = [
        f'custom_drawings/{class_name}/{i:04d}.png'
        for i in range(5)
        if os.path.exists(f'custom_drawings/{class_name}/{i:04d}.png')
    ]
    
    if len(examples) >= 3:
        success = predictor.add_custom_class(class_name, examples[:5])
        status = "OK" if success else "FAIL"
        print(f"[{status}] Taught '{class_name}' with {len(examples[:5])} examples")

# Test 4: NEW classes AFTER teaching (should work!)
print("\n" + "-"*60)
print("TEST 4: NEW classes AFTER teaching (should work!)")
print("-"*60)

correct_count = 0
total_count = 0

for class_name in new_classes:
    # Test on images the model has NEVER seen (images 10-15)
    for i in range(10, 15):
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

if total_count > 0:
    print(f"\nFew-shot accuracy: {correct_count}/{total_count} = {correct_count/total_count:.1%}")
    print(f"The model learned {len(new_classes)} new classes with just 5 examples each.")
