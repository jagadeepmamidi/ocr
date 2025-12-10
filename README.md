# OCR Pipeline Assignment - Handwritten Document PII Extraction

##  Overview

This project implements a complete OCR + PII-extraction pipeline for handwritten documents in JPEG format.

**Pipeline Flow:**
```
Input (handwritten JPEG) → Pre-processing → OCR → Text Cleaning → PII Detection → (Optional) Redacted Image
```

---

##  Objective

Build a simple OCR + PII-extraction pipeline for handwritten documents that can:
- Handle slightly tilted images
- Work with different handwriting styles
- Process basic doctor/clinic-style notes or forms
- Detect and redact sensitive information (PII)

---

##  Deliverables

###  1. Python Notebook File
- **File:** `OCR_PII_Pipeline.ipynb`
- Complete pipeline implementation with detailed documentation
- Includes visualization and analysis cells
- Ready to run end-to-end

###  2. Dependency Document
- **File:** `requirements.txt`
- All required Python packages listed
- Installation instructions included

###  3. Results Screenshot
- **Files:** 
  - `outputs/test_results_screenshot.png` - Comprehensive results with pipeline flow
  - `outputs/comparison_screenshot.png` - Side-by-side comparison
- Shows original, preprocessed, and redacted images
- Displays extracted text and detected PII

###  4. Benchmarking Ready
- Pipeline can process multiple documents in batch
- Simply add images to `inputs/` folder and run

---

##  Quick Start

### Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Download SpaCy language model:**
```bash
python -m spacy download en_core_web_sm
```

### Usage

#### Option 1: Run Python Script
```bash
# Place your handwritten images in the 'inputs' folder
python main.py
```

#### Option 2: Run Jupyter Notebook
```bash
jupyter notebook OCR_PII_Pipeline.ipynb
```

Results will be saved in the `outputs/` folder:
- `extracted_[filename].txt` - Extracted text and detected PII
- `redacted_[filename]` - Image with PII redacted (blacked out)
- `debug_clean_[filename]` - Preprocessed image (for debugging)

---

##  Pipeline Stages

### 1. Pre-processing
**Purpose:** Clean the image to improve OCR accuracy

**Steps:**
- Convert to grayscale
- Detect and remove horizontal lines (common in forms)
- Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Enhance text clarity

**Why it matters:** Handwritten documents often have ruled lines that confuse OCR engines. Removing them significantly improves accuracy.

### 2. OCR (Optical Character Recognition)
**Engine:** EasyOCR

**Features:**
- Works well with handwritten text
- No GPU required (CPU mode)
- Returns bounding boxes for each detected text region

### 3. Text Cleaning
**Process:** Extract and concatenate all detected text regions

### 4. PII Detection
**Methods:**

#### A. Regex Patterns
- **Phone Numbers:** Indian (+91) and US formats
- **Email Addresses:** Standard email pattern
- **Dates:** Multiple date formats (DD/MM/YYYY, MM-DD-YYYY, etc.)

#### B. Named Entity Recognition (NLP)
**Engine:** SpaCy (`en_core_web_sm`)

**Detects:**
- `PERSON` - Person names
- `ORG` - Organizations
- `GPE` - Geopolitical entities (cities, countries)

### 5. Redaction (Optional)
**Process:** 
- Match detected PII with OCR bounding boxes
- Draw black rectangles over sensitive information
- Preserve original image structure

---

##  Test Results

### Sample Document Processing

**Input:** Handwritten medical/clinical note (`sample1.jpg`)

**Detected PII:**
- Names (PERSON entities)
- Organizations (ORG entities)
- Locations (GPE entities)

**Output Files:**
- `outputs/extracted_sample1.jpg.txt` - Full text + PII list
- `outputs/redacted_sample1.jpg` - Redacted image
- `outputs/test_results_screenshot.png` - Visual results

---

##  Features

###  Handles Multiple Challenges
- Slightly tilted images
- Different handwriting styles
- Ruled/lined paper
- Varying ink intensity
- Form-based documents

###  Comprehensive PII Detection
- Phone numbers (multiple formats)
- Email addresses
- Dates
- Person names
- Organizations
- Locations

###  Flexible Output
- Extracted text files
- Redacted images
- Debug/preprocessed images
- Visualization support

---

##  Project Structure

```
redact/
├── inputs/                      # Place input images here
│   └── sample1.jpg
├── outputs/                     # Generated results
│   ├── extracted_sample1.jpg.txt
│   ├── redacted_sample1.jpg
│   ├── debug_clean_sample1.jpg
│   ├── test_results_screenshot.png
│   └── comparison_screenshot.png
├── main.py                      # Main pipeline script
├── OCR_PII_Pipeline.ipynb      # Jupyter notebook (Deliverable #1)
├── requirements.txt             # Dependencies (Deliverable #2)
├── generate_results.py          # Results visualization script
└── README.md                    # This file
```

---

##  How It Works

### Pre-processing Deep Dive

The pre-processing stage is critical for handwritten documents:

1. **Line Detection:**
   - Uses morphological operations with a horizontal kernel (40x1 pixels)
   - Detects horizontal lines that are common in forms and notebooks

2. **Line Removal:**
   - Subtracts detected lines from the image
   - Preserves vertical and curved strokes (actual handwriting)

3. **Contrast Enhancement:**
   - CLAHE algorithm enhances local contrast
   - Makes faint ink darker and more uniform
   - Improves OCR accuracy significantly

### PII Detection Strategy

**Two-pronged approach:**

1. **Pattern Matching (Regex):**
   - Fast and accurate for structured data (phones, emails, dates)
   - No ML model required
   - Works even with OCR errors (flexible patterns)

2. **Named Entity Recognition (SpaCy):**
   - Detects contextual entities (names, organizations)
   - Handles variations in formatting
   - More robust than simple keyword matching

---

## 🧪 Testing with New Documents

To test with your own documents:

1. **Prepare images:**
   - Format: JPEG or PNG
   - Content: Handwritten text
   - Quality: Clear, well-lit photos

2. **Add to inputs:**
   ```bash
   # Copy your images to the inputs folder
   cp your_document.jpg inputs/
   ```

3. **Run pipeline:**
   ```bash
   python main.py
   ```

4. **Check results:**
   ```bash
   # View extracted text
   cat outputs/extracted_your_document.jpg.txt
   
   # View redacted image
   # Open outputs/redacted_your_document.jpg
   ```

---

##  Troubleshooting

### Issue: OCR accuracy is poor
**Solutions:**
- Check image quality (should be clear, well-lit)
- Adjust CLAHE parameters in `preprocess()` method
- Try different EasyOCR parameters

### Issue: PII not detected
**Solutions:**
- Check regex patterns match your data format
- Ensure SpaCy model is installed: `python -m spacy download en_core_web_sm`
- Review extracted text for OCR errors

### Issue: Too many false positives
**Solutions:**
- Adjust confidence thresholds
- Refine regex patterns
- Filter by entity length (currently: `len(ent.text) > 2`)

---

## 📚 Dependencies

- **opencv-python** - Image processing and manipulation
- **easyocr** - OCR engine for text extraction
- **numpy** - Numerical operations
- **spacy** - Natural Language Processing for NER
- **matplotlib** (optional) - For visualization in notebook

---

##  Assignment Compliance

### Requirements Met:

 **Pipeline Flow:** Input → Pre-processing → OCR → Text Cleaning → PII Detection → Redacted Image

 **Input Support:** 2-3 handwritten document images (JPEG)

 **Robustness:**
- Handles slightly tilted images
- Works with different handwriting styles
- Processes doctor/clinic-style notes or forms

 **Deliverables:**
1. Python Notebook file (`OCR_PII_Pipeline.ipynb`)
2. Dependency document (`requirements.txt`)
3. Results screenshot (`outputs/test_results_screenshot.png`)
4. Ready for benchmarking with additional documents

---

##  Future Enhancements

Potential improvements for production use:

1. **Advanced Pre-processing:**
   - Deskewing for heavily tilted images
   - Noise reduction for low-quality scans
   - Adaptive thresholding

2. **Better OCR:**
   - Ensemble multiple OCR engines
   - Fine-tune EasyOCR on medical handwriting
   - Post-processing with spell-checking

3. **Enhanced PII Detection:**
   - Medical-specific entities (diagnosis codes, medications)
   - Address detection
   - Social Security Numbers
   - Custom entity training

4. **Performance:**
   - GPU acceleration for faster processing
   - Batch processing optimization
   - Parallel processing for multiple documents

5. **Output Options:**
   - PDF generation
   - Searchable PDF with OCR layer
   - JSON/CSV export for structured data

---

##  License

This project is created for educational purposes as part of an OCR Pipeline Assignment.

---

##  Author

Created for the OCR Pipeline Assignment - Handwritten Document PII Extraction

---

##  Support

For issues or questions:
1. Check the Troubleshooting section
2. Review the Jupyter notebook for detailed examples
3. Examine the debug output images in `outputs/`

---

**Last Updated:** December 2025
