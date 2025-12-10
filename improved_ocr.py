import cv2
import easyocr
import numpy as np
import spacy
import re
import os

INPUT_FOLDER = "inputs"
OUTPUT_FOLDER = "outputs"
filename = "page_14.jpg"

print(f"\n{'='*80}")
print(f"IMPROVED OCR PIPELINE - Quality Filtering Enabled")
print(f"{'='*80}\n")

print("[INIT] Loading EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False)

print("[INIT] Loading SpaCy...")
try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None
input_path = os.path.join(INPUT_FOLDER, filename)
image = cv2.imread(input_path)

print(f"\n[1/5] Pre-processing {filename}...")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_inv = cv2.bitwise_not(gray)
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
detected_lines = cv2.morphologyEx(gray_inv, cv2.MORPH_OPEN, horizontal_kernel)
clean_inv = cv2.subtract(gray_inv, detected_lines)
clean = cv2.bitwise_not(clean_inv)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
processed_image = clahe.apply(clean)

print("[2/5] Running OCR with confidence scoring...")
results = reader.readtext(processed_image, detail=1, paragraph=False)

print(f"   Total detections: {len(results)}")

print("\n[3/5] Filtering low-quality results...")

def is_readable(text, confidence):
    if confidence < 0.3:
        return False, "Low confidence"
    if len(text.strip()) < 2:
        return False, "Too short"
    special_chars = sum(1 for c in text if not c.isalnum() and c not in [' ', '-', '&', '/', '.'])
    if special_chars > len(text) // 2:
        return False, "Too many special chars"
    clean_text = re.sub(r'[^\w]', '', text).lower()
    if re.search(r'[bcdfghjklmnpqrstvwxyz]{6,}', clean_text):
        return False, "Too many consonants"
    if len(clean_text) > 20:
        return False, "Too long"
    if len(clean_text) > 5:
        num_count = sum(1 for c in clean_text if c.isdigit())
        if 0 < num_count < len(clean_text) and num_count > len(clean_text) * 0.6:
            return False, "Weird number/letter mix"
    return True, "OK"
filtered_results = []
rejected_results = []

for (bbox, text, confidence) in results:
    readable, reason = is_readable(text, confidence)
    
    if readable:
        filtered_results.append((bbox, text, confidence))
    else:
        rejected_results.append((bbox, text, confidence, reason))

print(f"   Accepted: {len(filtered_results)}")
print(f"   Rejected: {len(rejected_results)}")


if rejected_results:
    print(f"\n   Examples of rejected text:")
    for bbox, text, conf, reason in rejected_results[:10]:
        print(f"      '{text}' (conf: {conf:.2f}) - {reason}")

print("\n[4/5] Building cleaned text...")
filtered_results.sort(key=lambda x: (x[0][0][1], x[0][0][0]))

cleaned_text = " ".join([text for (bbox, text, conf) in filtered_results])

print(f"   Original length: {sum(len(text) for _, text, _ in results)} chars")
print(f"   Cleaned length: {len(cleaned_text)} chars")

print("\n[5/5] Detecting PII in cleaned text...")

detected_pii = []
patterns = {
    'PHONE': r'(\+91[\-\s]?)?[6-9]\d{9}|\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}',
    'EMAIL': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    'DATE': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
}

for p_type, pattern in patterns.items():
    for match in re.finditer(pattern, cleaned_text):
        detected_pii.append({'type': p_type, 'value': match.group()})


if nlp:
    doc = nlp(cleaned_text)
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE"]:
            if len(ent.text) > 2:
                detected_pii.append({'type': ent.label_, 'value': ent.text})

print(f"   PII detected: {len(detected_pii)}")



output_file = os.path.join(OUTPUT_FOLDER, f"extracted_{filename}_CLEANED.txt")

with open(output_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("IMPROVED OCR EXTRACTION (Quality Filtered)\n")
    f.write("="*80 + "\n\n")
    f.write(f"Source: {filename}\n")
    f.write(f"Total detections: {len(results)}\n")
    f.write(f"Accepted (readable): {len(filtered_results)}\n")
    f.write(f"Rejected (unreadable): {len(rejected_results)}\n")
    f.write(f"Acceptance rate: {len(filtered_results)/len(results)*100:.1f}%\n")
    f.write("\n" + "="*80 + "\n")
    f.write("CLEANED TEXT\n")
    f.write("="*80 + "\n\n")
    f.write(cleaned_text)
    f.write("\n\n" + "="*80 + "\n")
    f.write(f"DETECTED PII ({len(detected_pii)} items)\n")
    f.write("="*80 + "\n")
    for pii in detected_pii:
        f.write(f"{pii['type']}: {pii['value']}\n")
    

    f.write("\n\n" + "="*80 + "\n")
    f.write("REJECTED TEXT (Low Quality/Unreadable)\n")
    f.write("="*80 + "\n")
    for bbox, text, conf, reason in rejected_results:
        f.write(f"'{text}' (confidence: {conf:.2f}) - Reason: {reason}\n")

print(f"\n✓ Cleaned results saved to: {output_file}")



print(f"\n{'='*80}")
print("QUALITY IMPROVEMENT SUMMARY")
print(f"{'='*80}\n")

original_file = os.path.join(OUTPUT_FOLDER, f"extracted_{filename}.txt")
if os.path.exists(original_file):
    with open(original_file, 'r', encoding='utf-8') as f:
        original_text = f.read().split('--- DETECTED PII ---')[0].strip()
    
    print(f"BEFORE (Original OCR):")
    print(f"  - Characters: {len(original_text):,}")
    print(f"  - Words: {len(original_text.split()):,}")
    print(f"  - Sample: {original_text[:150]}...")
    
    print(f"\nAFTER (Quality Filtered):")
    print(f"  - Characters: {len(cleaned_text):,}")
    print(f"  - Words: {len(cleaned_text.split()):,}")
    print(f"  - Sample: {cleaned_text[:150]}...")
    
    print(f"\nIMPROVEMENT:")
    print(f"  - Removed {len(rejected_results)} low-quality detections")
    print(f"  - Acceptance rate: {len(filtered_results)/len(results)*100:.1f}%")
    print(f"  - Text reduction: {(1 - len(cleaned_text)/len(original_text))*100:.1f}%")

print(f"\n{'='*80}")
print("DONE - Check the cleaned output file for improved results")
print(f"{'='*80}\n")
