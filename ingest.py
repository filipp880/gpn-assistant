import json
import logging
import os
import chromadb
from core import parent_child_chunk
from retrieval import get_bge_m3, auto_tune_if_data_changed

logger = logging.getLogger(__name__)

PARENT_MAP_FILE = os.path.join("chromadb", "parent_map.json")


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
    parent_map = {}  # parent_id → parent_text

    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        if os.path.isfile(fpath) and fname.endswith('.txt'):
            with open(fpath, 'r', encoding='utf-8') as f:
                text = f.read()

            families = parent_child_chunk(text)

            if not families:
                continue

            child_docs = []
            child_ids = []
            child_metadatas = []

            for parent_idx, (parent_text, children) in enumerate(families):
                parent_id = f"{fname}_p{parent_idx}"
                parent_map[parent_id] = parent_text

                for child_idx, child_text in enumerate(children):
                    child_id = f"{parent_id}_c{child_idx}"
                    child_docs.append(child_text)
                    child_ids.append(child_id)
                    child_metadatas.append({
                        'source': fname,
                        'parent_id': parent_id,
                    })

            dense = model.encode(child_docs, return_dense=True, return_sparse=False)["dense_vecs"]

            collection.add(
                documents=child_docs,
                embeddings=dense.tolist(),
                ids=child_ids,
                metadatas=child_metadatas,
            )
            logger.info("%s: %d parents, %d children", fname, len(families), len(child_docs))

    # Сохраняем parent_map на диск (для загрузки при старте сервиса)
    os.makedirs(os.path.dirname(PARENT_MAP_FILE) or ".", exist_ok=True)
    with open(PARENT_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(parent_map, f, ensure_ascii=False)
    logger.info("Parent map сохранён: %d записей → %s", len(parent_map), PARENT_MAP_FILE)

    logger.info("Авто-подбор параметров retrieval...")
    auto_tune_if_data_changed()

if __name__ == "__main__":
    main()