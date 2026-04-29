import re

def normalise_merchant(description: str) -> str:
    """
    Cleans an individual transaction description by removing
    common bank prefixes, special characters, and reference codes.
    """
    text = description.lower()

    # remove common bank prefixes
    prefixes = [
        "pos txn ", "pos ", "bacs ", "sto ",
        "card payment ", "bank transfer ",
        "fps ", "emp "
    ]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break

    # remove special characters
    text = re.sub(r'[*#/]', '', text)

    # remove pure reference codes
    # matches tokens that are mix of letters+numbers like "58dr", "87wx", "6t4ufe"
    text = re.sub(r'\b[a-z]*\d+[a-z0-9]*\b', '', text)

    # collapse spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text
