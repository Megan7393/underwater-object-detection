from datetime import datetime
from pathlib import Path

progress_file = Path("reports/progress_notes.md")

note = """
### {date}

Task completed:
- Created and tested `src/check_images.py`.
- Confirmed the script can count images and report image dimensions from `data/test_images`.

Summary:
A reusable image inspection script has been added. This will be useful later for checking underwater dataset image files before training object detection models.

---
""".format(date=datetime.now().strftime("%d %B %Y"))

with progress_file.open("a", encoding="utf-8") as file:
    file.write(note)

print("Progress note added to reports/progress_notes.md")