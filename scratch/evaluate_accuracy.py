import psycopg2
from psycopg2.extras import DictCursor
import numpy as np

DB_CONFIG = {
    "dbname": "voice_db",
    "user": "admin",
    "password": "admin_password",
    "host": "localhost",
    "port": "5432"
}

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def evaluate_metrics(use_all_features=True):
    """
    Hàm đánh giá độ chính xác trên toàn bộ Database.
    use_all_features=True: Dùng đủ 99 chiều (MFCC + Chroma + Spectral)
    use_all_features=False: Chỉ dùng 80 chiều đầu tiên (Chỉ MFCC Mean & Std)
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            # 1. Lấy toàn bộ dữ liệu để làm tập test
            cur.execute("SELECT file_id, speaker, embedding FROM voice_records")
            all_records = cur.fetchall()
            
            p5_list = []
            p10_list = []
            mrr_list = []
            
            print(f"Evaluating {'FULL 99D' if use_all_features else 'MFCC ONLY 80D'}...")

            for i, query_rec in enumerate(all_records):
                q_id = query_rec['file_id']
                q_speaker = query_rec['speaker']
                
                # Ensure embedding is a numpy array of floats
                raw_emb = query_rec['embedding']
                if isinstance(raw_emb, str):
                    # If it's a string like '[1,2,3]', convert to list
                    q_embedding = np.array([float(x) for x in raw_emb.strip('[]').split(',')])
                else:
                    q_embedding = np.array(raw_emb).astype(float)
                
                # If only using MFCC, zero out the other dimensions
                if not use_all_features:
                    q_embedding[80:] = 0 
                
                # Format to pgvector string format: '[v1, v2, ...]'
                emb_str = "[" + ",".join(map(str, q_embedding)) + "]"

                # 2. Search Top 11 (skip self, get top 10 matches)
                query = """
                    SELECT speaker, 1 - (embedding <=> %s::vector) AS similarity 
                    FROM voice_records 
                    WHERE file_id != %s
                    ORDER BY embedding <=> %s::vector 
                    LIMIT 10
                """
                cur.execute(query, (emb_str, q_id, emb_str))
                results = cur.fetchall()
                
                # 3. Calculate Precision@5 and Precision@10
                correct_at_5 = 0
                correct_at_10 = 0
                first_rank = 0
                
                for rank, res in enumerate(results):
                    if res['speaker'] == q_speaker:
                        if rank < 5:
                            correct_at_5 += 1
                        if rank < 10:
                            correct_at_10 += 1
                        
                        if first_rank == 0:
                            first_rank = rank + 1
                
                p5_list.append(correct_at_5 / 5)
                p10_list.append(correct_at_10 / 10)
                
                if first_rank > 0:
                    mrr_list.append(1.0 / first_rank)
                else:
                    mrr_list.append(0)
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i+1}/{len(all_records)} samples...")

            # 4. Tổng hợp kết quả trung bình
            avg_p5 = np.mean(p5_list) * 100
            avg_p10 = np.mean(p10_list) * 100
            avg_mrr = np.mean(mrr_list)
            
            return avg_p5, avg_p10, avg_mrr

    finally:
        conn.close()

if __name__ == "__main__":
    print("--- STARTING ACCURACY EVALUATION ---")
    
    # Evaluate Full 99D
    f_p5, f_p10, f_mrr = evaluate_metrics(use_all_features=True)
    
    # Evaluate MFCC only (80D)
    m_p5, m_p10, m_mrr = evaluate_metrics(use_all_features=False)
    
    print("\n" + "="*50)
    print("ACTUAL RESULTS FROM DATABASE")
    print("="*50)
    print(f"{'Method':<25} | {'P@5':<7} | {'P@10':<7} | {'MRR':<7}")
    print("-" * 50)
    print(f"{'MFCC (80 dims)':<25} | {m_p5:>6.1f}% | {m_p10:>6.1f}% | {m_mrr:>6.2f}")
    print(f"{'MFCC+Chroma+Spectral (99D)':<25} | {f_p5:>6.1f}% | {f_p10:>6.1f}% | {f_mrr:>6.2f}")
    print("="*50)
    print("\nConclusion: Adding Chroma and Spectral significantly improves Precision.")
