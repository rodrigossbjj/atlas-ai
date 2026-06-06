
def cleaner_text(text: str) -> str:
    if text is None or text.strip() == "":
        return ""
    return " ".join(text.replace("\n", " ").strip().split())
