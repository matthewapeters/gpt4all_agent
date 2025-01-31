import re


def detect_posix_path(text_to_search: str) -> str:
    """Detects valid POSIX paths within a string."""

    # This regex pattern matches:
    # - Starts with / (root) or ./ (current directory) or ../ (parent directory)
    # - Followed by any number of directory names and/or filenames
    # - Directory names and filenames can contain letters, numbers, underscores, hyphens, and dots
    # - Path can end with a filename or a slash

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


# Test cases
def test_transform_posix_paths():
    test_cases = [
        (
            "this is my home path /home/dudely and it has many files",
            "this is my home path slash h o m e slash d u d e l y and it has many files",
        ),
        (
            "the current directory is ./documents/stuff and it is empty",
            "the current directory is dot slash d o c u m e n t s slash s t u f f and it is empty",
        ),
        (
            "/usr/local/bin is a common directory",
            "slash u s r slash l o c a l slash b i n is a common directory",
        ),
        (
            "relative path like project/src is valid",
            "relative path like p r o j e c t slash s r c is valid",
        ),
        (
            "a trailing slash /var/log/ should be handled",
            "a trailing slash slash v a r slash l o g slash should be handled",
        ),
        (
            "hidden files like .config are common",
            "hidden files like dot c o n f i g are common",
        ),
        (
            "/this/path/ends/with/slash/",
            "slash t h i s slash p a t h slash e n d s slash w i t h slash s l a s h slash",
        ),
        (
            "a path.with.dots/in.the.middle",
            "a p a t h dot w i t h dot d o t s slash i n dot t h e dot m i d d l e",
        ),
        (
            "checking multiple paths /var/tmp and /opt/bin",
            "checking multiple paths slash v a r slash t m p and slash o p t slash b i n",
        ),
        (
            "a single file /filename should be converted",
            "a single file slash f i l e n a m e should be converted",
        ),
    ]

    for input_text, expected in test_cases:
        output = detect_posix_path(input_text)
        assert (
            output == expected
        ), f"For input: '{input_text}', expected '{expected}' but got '{output}'"

    print("All tests passed.")


test_transform_posix_paths()
