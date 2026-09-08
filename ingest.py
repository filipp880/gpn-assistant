import logging
import os
import chromadb
from core import adaptive_chunk
from retrieval import get_bge_m3, auto_tune_if_data_changed

logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if hasattr(chromadb, "PersistentClient"):
        client = chromadb.PersistentClient(path="./chromadb")
    elif hasattr(chromadb, "Client"):
        client = chromadb.Client()
    else:
        raise ImportError("Не удалось инициализировать ChromaDB. Проверьте отсутствие файла chromadb.py в директории.")

    try:
        client.delete_collection('kbase')
    except Exception:
        pass

    collection = client.create_collection(
        name='kbase',
        metadata={"hnsw:space": "cosine"}
    )

    data_dir = 'data'
    if not os.path.isdir(data_dir):
        logger.warning("Не найдена директория data/ — база знаний не построена.")
        return

    model = get_bge_m3()

    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        if os.path.isfile(fpath) and fname.endswith('.txt'):
            with open(fpath, 'r', encoding='utf-8') as f:
                text = f.read()

            chunks = adaptive_chunk(text)

            if not chunks:
                continue

            dense = model.encode(chunks, return_dense=True, return_sparse=False)["dense_vecs"]

            collection.add(
                documents=chunks,
                embeddings=dense.tolist(),
                ids=[f"{fname}_{i}" for i in range(len(chunks))],
                metadatas=[{'source': fname} for _ in chunks]
            )
            logger.info("%s: %d чанков", fname, len(chunks))

    logger.info("Проверка фильтрации: %d чанков из transformer_notes.txt",
                len(collection.get(where={'source': 'transformer_notes.txt'})['ids']))

    logger.info("Авто-подбор параметров retrieval...")
    auto_tune_if_data_changed()

if __name__ == "__main__":
    main()