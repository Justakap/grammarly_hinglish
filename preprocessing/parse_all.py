
from pathlib import Path

PROJECT_DIR = Path(
    "/content/drive/MyDrive/hinglish-next-word-predictor"
)

INPUT_FILE = PROJECT_DIR / "data" / "raw" / "all.txt"
OUTPUT_FILE = PROJECT_DIR / "data" / "processed" / "sentences.txt"
TAGGED_OUTPUT_FILE = PROJECT_DIR / "data" / "processed" / "sentences_with_tags.txt"


def parse_all_txt():
    sentences = []
    tagged_sentences = []

    current_tokens = []
    current_tags = []

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            # Blank line = end of sentence
            if not line:
                if current_tokens:
                    sentences.append(" ".join(current_tokens))
                    tagged_sentences.append(
                        " ".join(
                            f"{token}/{tag}"
                            for token, tag in zip(
                                current_tokens, current_tags
                            )
                        )
                    )

                    current_tokens = []
                    current_tags = []

                continue

            parts = line.split("\t")

            # Ignore malformed lines safely
            if len(parts) != 2:
                print(
                    f"Warning: malformed line {line_number}: "
                    f"{raw_line!r}"
                )
                continue

            token, tag = parts

            token = token.strip()
            tag = tag.strip()

            if not token:
                continue

            current_tokens.append(token)
            current_tags.append(tag)

    # Handle a final sentence without a trailing blank line
    if current_tokens:
        sentences.append(" ".join(current_tokens))
        tagged_sentences.append(
            " ".join(
                f"{token}/{tag}"
                for token, tag in zip(current_tokens, current_tags)
            )
        )

    OUTPUT_FILE.write_text(
        "\n".join(sentences),
        encoding="utf-8"
    )

    TAGGED_OUTPUT_FILE.write_text(
        "\n".join(tagged_sentences),
        encoding="utf-8"
    )

    print(f"Sentences parsed: {len(sentences)}")
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Saved: {TAGGED_OUTPUT_FILE}")


if __name__ == "__main__":
    parse_all_txt()
