"""
javus.posix_paths

"""


def detect_posix_path(text_to_search: str) -> str:
    """
    detect_posix_path:

    Detects valid POSIX paths within a string and transforms them into a tts-readable format.

    Args: text_to_search (str): The text to search for POSIX paths.

    Appologies: I attempted this function with regiex with AI assistance, but chatGPT was not able to provide a soltion.


    Known issues:
    - fractions like "`1/2`" will be transformed into "one slash two" which would be betters as "one half"
    - there are times when the TTS will stumble over letters - not much we can do about that.

    """

    words = text_to_search.split(" ")
    paths = []
    new_paths = []
    for word in words:
        if "/" in word or word.startswith("."):
            paths.append(word)

    for path in paths:
        new_parts = []
        parts = path.split("/")
        for part in parts:
            new_part = ""
            if part != "." and part != "..":
                for letter in part:
                    new_part += letter + " "
            new_parts.append(new_part or part)
        new_path = (
            (
                "/".join(new_parts)
                .replace("/", " slash ")
                .replace(".", " dot ")
                .replace("_", " underscore ")
                .replace("-", " hyphen ")
            )
            .strip()
            .replace("  ", " ")
        )
        new_paths.append(new_path)

    for idx, new_path in enumerate(new_paths):
        text_to_search = text_to_search.replace(paths[idx], new_path)

    return text_to_search
