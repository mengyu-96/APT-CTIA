
import sys
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        # Read first 10 pages or so to get the gist (Abstract, Intro, Methodology, Experiments)
        # Reading all might be too much for context window, but let's try reading a significant portion.
        # Usually Intro and Experiments are key.
        for i, page in enumerate(doc):
            text += f"--- Page {i+1} ---\n"
            text += page.get_text()
            if i >= 15: # Limit to 15 pages to avoid excessive output if it's a book
                break
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

if __name__ == "__main__":
    pdf_path = r"d:\git\APT归因\Computer Networks--APT-ATT.pdf"
    content = extract_text_from_pdf(pdf_path)
    print(f"Extraction result length: {len(content)}")
    if content.startswith("Error"):
        print(content)
    else:
        try:
            with open(r"d:\git\APT归因\paper_content.txt", "w", encoding="utf-8") as f:
                f.write(content)
            print("Successfully wrote to paper_content.txt")
        except Exception as e:
            print(f"Error writing file: {e}")
