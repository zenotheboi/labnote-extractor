"""
pipeline.perception — page-agnostic OCR/CV stages.

All stages are content-agnostic: they run identically on any notebook page.
Each stage returns or updates the shared ``fields`` dict (see pipeline.__init__
for the Field envelope contract).

Typical call order (orchestrated by main.py):

    preprocessed = preprocess(image_path)
    regions      = layout.segment(preprocessed)
    fields       = {}
    fields.update(ocr_text.extract(text_regions))
    fields.update(ocr_math.extract(math_regions))
    fields.update(ocsr.extract(drawing_regions))
    fields       = normalize.run(fields)
"""
