"""End-to-end pipeline to load data, embed, retrieve, and rerank.

Notes:
- Uses local caching for embeddings and rerank scores under `data/`.
- Avoids recomputation when cache files exist.
- Keeps changes minimal while fixing correctness issues in the original script.
"""

try:
    import orjson as _fastjson
except Exception:  # optional fast JSON
    _fastjson = None
import json
import pickle
from pathlib import Path
import os
import numpy as np

from parse import (
    _extract_data_fields,
    _extract_data_fields_questions,
    load_jsonl,
)
from embed import construct_text, get_embeddings, get_summary, rerank_docs, chunk_texts
from tiny_knn import exact_search

if __name__ == "__main__":
    # Paths and model names
    mainpath = '/fast/nchandak/forecasting/newsdata/retrieval/data/'
    docs_folder = 'documents/'
    questions_folder = 'questions/'
    summary_model = "/fast/nchandak/models/Qwen3-32B"
    embedding_model = 'Qwen/Qwen3-Embedding-8B'
    reranker_model = 'Qwen/Qwen3-Reranker-8B'

    # Input files
    data_files = [
        # mainpath + docs_folder + 'cnbc.jsonl',
        # mainpath + docs_folder + 'cnn.jsonl',
        # mainpath + docs_folder + 'dw.jsonl',
        # mainpath + docs_folder + 'guardian_2025.jsonl',
        # mainpath + docs_folder + 'forbes.jsonl',
    ]
    
    # data files is everything in the docs folder 
    
    question_files = [
        # mainpath + questions_folder + 'deepseekv3.jsonl',
        mainpath + questions_folder + 'deepseekv3_theguardian2025.jsonl',
        # mainpath + questions_folder + 'janmarch2025.jsonl',
        # mainpath + questions_folder + 'o4mini.jsonl',
        # mainpath + questions_folder + 'metaculus.jsonl',
        # mainpath + questions_folder + 'combined_all_questions.jsonl',
        # mainpath + questions_folder + 'combined_non_numeric_all_train.jsonl',
        # mainpath + questions_folder + 'combined_non_numeric_all_validation.jsonl',
        # mainpath + questions_folder + 'manifold_binary_train_2k.jsonl',
    ]
    
    data_files = [mainpath + docs_folder + f for f in os.listdir(mainpath + docs_folder) if f.endswith('.jsonl')]
    # question_files = [mainpath + questions_folder + f for f in os.listdir(mainpath + questions_folder) if f.endswith('.jsonl')]
    
    # mainpath += 'storage/'

    # Load documents
    source_data: dict = {}
    for fp in data_files:
        print(f"Loading data from {fp}")
        loaded = load_jsonl(Path(fp), transform=_extract_data_fields)
        print(f"Loaded {len(loaded)} documents from {fp}")
        source_data.update(loaded)
    print(f"Total documents loaded: {len(source_data)}")

    # Load questions, partition by set name
    ds3q, o4miniq, combq, metaculus, trainq, valq = {}, {}, {}, {}, {}, {}
    binary_train = {}
    for qf in question_files:
        print(f"Loading questions from {qf}")
        q_loaded = load_jsonl(Path(qf), transform=_extract_data_fields_questions)
        print(f"Loaded {len(q_loaded)} questions from {qf}")
        if "deepseekv3" in qf or "janmarch2025" in qf:
            ds3q.update(q_loaded)
        elif "o4mini" in qf:
            o4miniq.update(q_loaded)
        elif "binary_train_2k" in qf:
            binary_train.update(q_loaded)
        elif "train" in qf:
            trainq.update(q_loaded)
        elif "validation" in qf:
            valq.update(q_loaded)
        elif "combined" in qf:
            combq.update(q_loaded)
        elif "metaculus" in qf:
            metaculus.update(q_loaded)
            
    # Construct texts
    document_keys, document_text = construct_text(source_data, field="maintext")
    ds3q_keys, ds3q_text = construct_text(ds3q, field="question_title")
    o4miniq_keys, o4miniq_text = construct_text(o4miniq, field="question_title")
    metaculus_keys, metaculus_text = construct_text(metaculus, field="question_title")
    combq_keys, combq_text = construct_text(combq, field="question_title")
    trainq_keys, trainq_text = construct_text(trainq, field="question_title")
    valq_keys, valq_text = construct_text(valq, field="question_title")
    binary_train_keys, binary_train_text = construct_text(binary_train, field="question_title")
    
    # Optional: compute or load summaries and their embeddings
    print("Checking summaries for documents (skipping if no cache and generation disabled)...")
    doc_summaries = None
    # Uncomment below to (re)compute summaries
    # if not Path(mainpath+'doc_summaries.pkl').is_file():
    #     doc_summaries = get_summary(document_text, model_name=summary_model, batch_size=32)
    #     for i, s in enumerate(doc_summaries):
    #         source_data[document_keys[i]]['summary'] = s
    #     with open(mainpath+'doc_summaries.pkl', 'wb') as f:
    #         pickle.dump(doc_summaries, f)
    # else:
    #     with open(mainpath+'doc_summaries.pkl', 'rb') as f:
    #         doc_summaries = pickle.load(f)

    # Passage-level chunking and embeddings (for improved recall)
    print("Chunking documents into passages for retrieval...")
    if not Path(mainpath + 'passages_chunked.pkl').is_file():
        passage_texts, passage_to_doc_idx = chunk_texts(document_text, model_name=embedding_model, max_tokens=512, stride=64)
        with open(mainpath + 'passages_chunked.pkl', 'wb') as f:
            pickle.dump((passage_texts, passage_to_doc_idx), f)
    else:
        with open(mainpath + 'passages_chunked.pkl', 'rb') as f:
            passage_texts, passage_to_doc_idx = pickle.load(f)
            
    passage_emb_path = mainpath + 'doc_passage_embeddings.npy'
    if not Path(passage_emb_path).is_file():
        print(f"Computing embeddings for {len(passage_texts)} passages...")
        doc_embeddings = get_embeddings(passage_texts, is_query=False, model_name=embedding_model)
        np.save(passage_emb_path, doc_embeddings)
    else:
        print("Loading precomputed passage embeddings...")
        doc_embeddings = np.load(passage_emb_path)

    # Embeddings: summaries (only if available)
    summary_embeddings = None
    if doc_summaries is not None and not Path(mainpath+'summary_embeddings.npy').is_file():
        print("Computing summary embeddings...")
        summary_embeddings = get_embeddings(doc_summaries, is_query=False, model_name=embedding_model)
        np.save(mainpath+'summary_embeddings.npy', summary_embeddings)
    elif Path(mainpath+'summary_embeddings.npy').is_file():
        print("Loading precomputed summary embeddings...")
        summary_embeddings = np.load(mainpath+'summary_embeddings.npy')

    # Embeddings: questions
    if not Path(mainpath+'ds3q_embeddings.npy').is_file():
        print("Computing DS3 question embeddings...")
        ds3q_embeddings = get_embeddings(ds3q_text, is_query=True, model_name=embedding_model)
        np.save(mainpath+'ds3q_embeddings.npy', ds3q_embeddings)
    else:
        print("Loading precomputed DS3 question embeddings...")
        ds3q_embeddings = np.load(mainpath+'ds3q_embeddings.npy')

    if not Path(mainpath+'o4miniq_embeddings.npy').is_file():
        print("Computing O4Mini question embeddings...")
        o4miniq_embeddings = get_embeddings(o4miniq_text, is_query=True, model_name=embedding_model)
        np.save(mainpath+'o4miniq_embeddings.npy', o4miniq_embeddings)
    else:
        print("Loading precomputed O4Mini question embeddings...")
        o4miniq_embeddings = np.load(mainpath+'o4miniq_embeddings.npy')

    print("Computing Metaculus question embeddings...")
    if not Path(mainpath+'metaculus_embeddings.npy').is_file():
        metaculus_embeddings = get_embeddings(metaculus_text, is_query=True, model_name=embedding_model)
        np.save(mainpath+'metaculus_embeddings.npy', metaculus_embeddings)
    else:
        print("Loading precomputed Metaculus question embeddings...")
        metaculus_embeddings = np.load(mainpath+'metaculus_embeddings.npy')
        
    print("Computing train question embeddings...")
    if not Path(mainpath+'trainq_embeddings.npy').is_file():
        trainq_embeddings = get_embeddings(trainq_text, is_query=True, model_name=embedding_model)
        np.save(mainpath+'trainq_embeddings.npy', trainq_embeddings)
    else:
        print("Loading precomputed train question embeddings...")
        trainq_embeddings = np.load(mainpath+'trainq_embeddings.npy')

    print("Computing validation question embeddings...")

    if not Path(mainpath+'valq_embeddings.npy').is_file():
        valq_embeddings = get_embeddings(valq_text, is_query=True, model_name=embedding_model)
        np.save(mainpath+'valq_embeddings.npy', valq_embeddings)
    else:
        print("Loading precomputed validation question embeddings...")
        valq_embeddings = np.load(mainpath+'valq_embeddings.npy')
        
    print("Computing binary train question embeddings...")
    if not Path(mainpath+'binary_train_embeddings.npy').is_file():
        binary_train_embeddings = get_embeddings(binary_train_text, is_query=True, model_name=embedding_model)
        np.save(mainpath+'binary_train_embeddings.npy', binary_train_embeddings)
    else:
        print("Loading precomputed binary train question embeddings...")
        binary_train_embeddings = np.load(mainpath+'binary_train_embeddings.npy')

    print("Computing Combined question embeddings...")
    if not Path(mainpath+'combq_embeddings.npy').is_file():
        combq_embeddings = get_embeddings(combq_text, is_query=True, model_name=embedding_model)
        np.save(mainpath+'combq_embeddings.npy', combq_embeddings)
    else:
        combq_embeddings = np.load(mainpath+'combq_embeddings.npy')
    print("All embeddings computed.")

    docs = [("docs", doc_embeddings)]
    qsets = [
        ("DS3", ds3q_embeddings, ds3q_text, ds3q),
        # ("O4mini", o4miniq_embeddings, o4miniq_text, o4miniq),
        # ("Metaculus", metaculus_embeddings, metaculus_text, metaculus),
        # ("Combo", combq_embeddings, combq_text, combq),
        # ("train", trainq_embeddings, trainq_text, trainq),
        # ("validation", valq_embeddings, valq_text, valq),
        # ("binary_train", binary_train_embeddings, binary_train_text, binary_train),
    ]

    delta = 86400 * 30 # 30 days
    days = int(delta / 86400)
    # First-stage retrieval (KNN)
    for qname, q_emb, q_texts, qdict in qsets:
        for name, d_emb in docs:
            print(f"Performing KNN search for document shape {d_emb.shape} and question shape {q_emb.shape}")
            indices, distances = exact_search(q_emb, d_emb, k=500, metric='cosine')
            print(f"Search completed. Indices shape: {indices.shape}, Distances shape: {distances.shape}")
            pairs: list = []

            for i in range(distances.shape[0]):
                count = 0
                relevant_docs = []
                query = q_texts[i]
                seen_docs = set()
                for j in range(distances.shape[1]):
                    passage_idx, score = int(indices[i, j]), float(distances[i, j])
                    docid = passage_to_doc_idx[passage_idx]
                    passage_text = passage_texts[passage_idx]
                    key = document_keys[docid]
                    doc_meta = source_data.get(key)
                    q_meta = qdict.get(str(i + 1))
                    
                    if doc_meta is None or q_meta is None:
                        continue
                    
                    if doc_meta.get('max_date') is not None and q_meta.get('resolution_date') is not None:
                        # if doc_meta['max_date'] < q_meta['question_start_date']: # or (q_meta['resolution_date'] is None and doc_meta['max_date'] < q_meta['question_start_date']):
                        # Delta of 1 month 
                        if doc_meta['max_date'] < (q_meta['resolution_date'] - delta):
                        # if doc_meta['max_date'] < max((q_meta['resolution_date'] - delta), q_meta['question_start_date']):
                            if docid in seen_docs:
                                continue
                            seen_docs.add(docid)
                            count += 1
                            
                            doc_meta.pop('maintext', None)
                            doc_meta['relevant_passage'] = passage_text
                            relevant_docs.append((str(score), key, doc_meta))
                            pairs.append(((i, passage_idx), (query, passage_text)))
                            
                            # print(doc_meta['max_date'], q_meta['question_start_date'])
                            # pairs.append(((i, passage_idx), (query, passage_texts[passage_idx])))
                            if count >= 10:
                                break
                            
                qdict[str(i + 1)]['relevant_articles_sorted_by_' + name] = relevant_docs

            with open(f'{mainpath}pairs_{qname}_{name}.pkl', 'wb') as f:
                pickle.dump(pairs, f)

        # # Save questions with initial rankings
        out_path = f'{mainpath}ranked_queries_{qname}_{days}.json'
        # out_path = f'{mainpath}ranked_queries_{qname}.json'
        # if _fastjson is not None:
        #     with open(out_path, 'wb') as f:
        #         f.write(_fastjson.dumps(qdict))
        # else:
        #     with open(out_path, 'w') as f:
        #         json.dump(qdict, f)
                
                
                
        # Also output in a jsonl file (one query per line)
        out_path_jsonl = out_path.replace('.json', '.jsonl')
        with open(out_path_jsonl, 'w', encoding='utf-8') as f_jsonl:
            for qid, qdata in qdict.items():
                record = {'qid': qid}
                record.update(qdata)
                f_jsonl.write(json.dumps(record, ensure_ascii=False) + '\n')

    # # Second-stage reranking
    # for qname, _q_emb, _q_texts, qdict in qsets:
    #     for name, _ in docs:
    #         with open(f'{mainpath}pairs_{qname}_{name}.pkl', 'rb') as f:
    #             pairs = pickle.load(f)

    #         scores_path = f'{mainpath}rerank_scores_{qname}_{name}.npy'
    #         # Build per-query batches to leverage prefix caching
    #         from collections import defaultdict
    #         per_query = defaultdict(list)  # qidx -> list of (pair_index, passage_idx, (query, passage_text))
    #         for idx, (idxes, pair_text) in enumerate(pairs):
    #             qidx, passage_idx = idxes
    #             per_query[qidx].append((idx, passage_idx, pair_text))

    #         if Path(scores_path).is_file():
    #             print(f"Loading precomputed rerank scores for {qname} and {name}...")
    #             results = np.load(scores_path)
    #         else:
    #             print(f"Reranking {len(pairs)} pairs for {qname} and {name} (batched per query)...")
    #             results = np.zeros(len(pairs), dtype=np.float32)
    #             for qidx, items in per_query.items():
    #                 batch_pairs = [pt for (_i, _pidx, pt) in items]
    #                 batch_scores = rerank_docs(model_name=reranker_model, query_doc_pairs=batch_pairs)
    # #                 for (i_pair, _pidx, _pt), s in zip(items, batch_scores):
    # #                     results[i_pair] = float(s)
    # #             np.save(scores_path, results.astype(np.float32))

    # #         # Aggregate to doc level: best passage score per doc
    
    #         # Aggregate to doc level: best passage score per doc
    #         for qidx, items in per_query.items():
    #             best_by_doc: dict[int, float] = {}
    #             nw_passage_texts: dict[int, str] = {}
    #             for (i_pair, passage_idx, passage_text) in items:
    #                 docid = passage_to_doc_idx[passage_idx]
    #                 s = float(results[i_pair])
    #                 if (docid not in best_by_doc) or (s > best_by_doc[docid]):
    #                     best_by_doc[docid] = s
    #                     nw_passage_texts[docid] = passage_text

    #             top = sorted(best_by_doc.items(), key=lambda x: x[1], reverse=True)[:20]
    #             relevant_docs = []
    #             for didx, score in top:
    #                 key = document_keys[didx]
    #                 passage_text = nw_passage_texts[didx]
    #                 doc_meta = source_data.get(key, {})
    #                 relevant_docs.append((str(score), key, passage_text, doc_meta))


    # #             qdict[str(qidx + 1)]['reranked_relevant_articles_sorted_by_' + name] = relevant_docs
        
    #     # Save questions with reranked results
    #     out_path = f'{mainpath}reranked_queries_{qname}.json'
    #     if _fastjson is not None:
    #         with open(out_path, 'wb') as f:
    #             f.write(_fastjson.dumps(qdict))
    #     else:
    #         with open(out_path, 'w') as f:
    #             json.dump(qdict, f)
        
