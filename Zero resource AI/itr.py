import os
os.environ["TESSDATA_PREFIX"] = r"C:\Users\amins\OneDrive\Pictures\Documents\ITR\tessdata"
import logging
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\amins\OneDrive\Pictures\Documents\ITR\tesseract.exe"

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mapping for Tesseract language code to NLLB model language code
SOURCE_LANGS = {
    'kannada':   {'tess': 'kan', 'nllb': 'kan_Knda'},
    'tamil':     {'tess': 'tam', 'nllb': 'tam_Taml'},
    'marathi':   {'tess': 'mar', 'nllb': 'mar_Deva'},
    'gujarati':  {'tess': 'guj', 'nllb': 'guj_Gujr'},
    'hindi':     {'tess': 'hin', 'nllb': 'hin_Deva'},
    'french':    {'tess': 'fra', 'nllb': 'fra_Latn'},   
}

# Output/Target languages supported by NLLB (abbreviated set)
TARGET_LANGS = {
    'english':    'eng_Latn',
    'hindi':      'hin_Deva',
    'kannada':    'kan_Knda',
    'tamil':      'tam_Taml',
    'marathi':    'mar_Deva',
    'french':     'fra_Latn',
}

def extract_text_from_image(image_path, tess_lang):
    logger.info(f"Extracting text from image: {image_path} (language: {tess_lang})")
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang=tess_lang)
    logger.info(f"Extracted text:\n{text.strip()}")
    return text.strip()

def translate_text_nllb(text, src_nllb, tgt_nllb, trans_path="facebook/nllb-200-distilled-600M"):
    logger.info(f"Translating from {src_nllb} to {tgt_nllb} using NLLB-200...")

    try:
        # First try offline
        tokenizer = AutoTokenizer.from_pretrained(trans_path, local_files_only=True, src_lang=src_nllb, tgt_lang=tgt_nllb)
        model = AutoModelForSeq2SeqLM.from_pretrained(trans_path, local_files_only=True)
    except Exception as e:
        logger.warning("Model not found locally. Downloading from HuggingFace...")
        tokenizer = AutoTokenizer.from_pretrained(trans_path, src_lang=src_nllb, tgt_lang=tgt_nllb)
        model = AutoModelForSeq2SeqLM.from_pretrained(trans_path)

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_nllb)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=200
        )
    translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    logger.info("Translation completed.")
    return translation

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multilingual Image Text Translator")
    parser.add_argument("image_file", help="Path to image file containing text")
    args = parser.parse_args()

    # Show user language options
    print("\nAvailable source languages:")
    for i, lang in enumerate(SOURCE_LANGS):
        print(f"  {i+1}. {lang.title()}")
    src_choice = int(input("Select source language (number): "))
    src_key = list(SOURCE_LANGS.keys())[src_choice - 1]
    src_lang = SOURCE_LANGS[src_key]

    print("\nAvailable target languages:")
    for i, lang in enumerate(TARGET_LANGS):
        print(f"  {i+1}. {lang.title()}")
    tgt_choice = int(input("Select target language (number): "))
    tgt_key = list(TARGET_LANGS.keys())[tgt_choice - 1]
    tgt_lang = TARGET_LANGS[tgt_key]

    # Step 1: OCR extraction
    text = extract_text_from_image(args.image_file, src_lang['tess'])
    if not text.strip():
        print("No text detected! Check image quality/language.")
        return

    # Step 2: Translation
    translation = translate_text_nllb(text, src_lang['nllb'], tgt_lang)
    print("\n========== Translation Result ==========")
    print("Extracted Text:\n", text)
    print("\nTranslated Text:\n", translation)
    print("========================================\n")

if __name__ == "__main__":
    main()
