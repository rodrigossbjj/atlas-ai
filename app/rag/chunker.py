CHUNK_SIZE = 300
OVERLAP = 50


def create_chunks(document: dict) -> list[dict]:
    chunks = []
    chunk_id = 1

    for page in document["pages"]:
        words = page["content"].split()
        start = 0

        while start < len(words):
            end = min(start + CHUNK_SIZE, len(words))

            chunks.append({
                "chunk_id": chunk_id,
                "page": page["page"],
                "source": document["filename"],
                "content": " ".join(words[start:end])
            })

            chunk_id += 1
            start += CHUNK_SIZE - OVERLAP

    return chunks