import json
from app.database.connection import get_connection

class Repository:
    def save_chunk(self,chunk_id,page,source,content,embedding):
        try:
            conn = get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO chunks
                (
                    chunk_id,
                    page,
                    source,
                    content,
                    embedding
                )
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    chunk_id,
                    page,
                    source,
                    content,
                    json.dumps(embedding)
                )
            )

            conn.commit()
            return True

        except Exception as e:
            print(f"Erro ao salvar chunk: {e}")
            return False
        finally:
            cursor.close()
            conn.close()
    
    
    def get_all_chunks(self): 
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                chunk_id,
                page,
                source,
                content,
                embedding
            FROM chunks
        """)

        rows = cursor.fetchall()

        cursor.close()
        conn.close()

        chunks = []

        for row in rows:
            chunks.append({
                "chunk_id": row[0],
                "page": row[1],
                "source": row[2],
                "content": row[3],
                "embedding": (
                    json.loads(row[4])
                    if isinstance(row[4], str)
                    else row[4]
                )
            })

        return chunks

    def count(self) -> int:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT
                count(*)
            FROM chunks
        """)

        total = cursor.fetchone()[0]

        cursor.close()
        conn.close()

        return total
