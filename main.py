import cv2
import easyocr
import numpy as np
import spacy
import re
import os

INPUT_FOLDER = "inputs"
OUTPUT_FOLDER = "outputs"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

print("[INIT] Loading EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False) 

print("[INIT] Loading SpaCy...")
try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None

class ProPipeline:
    def __init__(self, filename):
        self.filename = filename
        self.input_path = os.path.join(INPUT_FOLDER, filename)
        self.image = cv2.imread(self.input_path)
        
        if self.image is None:
            raise ValueError(f"Could not load image: {self.input_path}")
            
        self.processed_image = None
        self.ocr_results = []
        self.full_text = ""
        self.pii_matches = [] 

    def preprocess(self):
        img = self.image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_inv = cv2.bitwise_not(gray)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        detected_lines = cv2.morphologyEx(gray_inv, cv2.MORPH_OPEN, horizontal_kernel)
        clean_inv = cv2.subtract(gray_inv, detected_lines)
        clean = cv2.bitwise_not(clean_inv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.processed_image = clahe.apply(clean)
        debug_path = os.path.join(OUTPUT_FOLDER, f"debug_clean_{self.filename}")
        cv2.imwrite(debug_path, self.processed_image)
        print(f"   [DEBUG] Preprocessed image saved to {debug_path}")
        return self.processed_image

    def run_ocr(self):
        print(f"   [OCR] Scanning {self.filename}...")
        results = reader.readtext(self.processed_image, detail=1, paragraph=False)
        self.ocr_results = results
        self.full_text = " ".join([res[1] for res in results])
        return self.full_text

    def detect_pii(self):
        text = self.full_text
        detected = []
        patterns = {
            'PHONE': r'(\+91[\-\s]?)?[6-9]\d{9}|\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}',
            'EMAIL': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'DATE': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        }
        for p_type, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                detected.append({'type': p_type, 'value': match.group()})
        if nlp:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE"]:
                    if len(ent.text) > 2: 
                        detected.append({'type': ent.label_, 'value': ent.text})
        self.pii_matches = detected
        return detected

    def redact_image(self):
        redacted_img = self.image.copy()
        for (bbox, text, prob) in self.ocr_results:
            is_sensitive = False
            for pii in self.pii_matches:
                clean_text = re.sub(r'[^\w]', '', text).lower()
                clean_pii = re.sub(r'[^\w]', '', pii['value']).lower()
                if len(clean_text) > 2 and (clean_text in clean_pii or clean_pii in clean_text):
                    is_sensitive = True
                    break
            if is_sensitive:
                (tl, tr, br, bl) = bbox
                tl = (int(tl[0]), int(tl[1]))
                br = (int(br[0]), int(br[1]))
                cv2.rectangle(redacted_img, tl, br, (0, 0, 0), -1)
        return redacted_img

    def save_results(self, final_image):
        text_filename = os.path.join(OUTPUT_FOLDER, f"extracted_{self.filename}.txt")
        with open(text_filename, "w", encoding='utf-8') as f:
            f.write(self.full_text)
            f.write("\n\n--- DETECTED PII ---\n")
            for pii in self.pii_matches:
                f.write(f"{pii['type']}: {pii['value']}\n")
        img_filename = os.path.join(OUTPUT_FOLDER, f"redacted_{self.filename}")
        cv2.imwrite(img_filename, final_image)
        print(f"   [DONE] Saved to {OUTPUT_FOLDER}")

if __name__ == "__main__":
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not files:
        print(f"Please put images in '{INPUT_FOLDER}'")
    else:
        for file in files:
            print(f"\n--- Processing {file} ---")
            try:
                pipeline = ProPipeline(file)
                pipeline.preprocess()
                text = pipeline.run_ocr()
                print(f"   [PREVIEW] {text[:100]}...") 
                pipeline.detect_pii()
                final_img = pipeline.redact_image()
                pipeline.save_results(final_img)
            except Exception as e:
                print(f"   [ERROR] {e}")