# Retrieval Pipeline

## News Download and Extraction
First, we need to obtain CCNews data and extract articles from the WARC files with HTML. For this, we use news-please.
```git clone https://github.com/fhamborg/news-please.git```

Specifically, we use news-please/news_please/examples/commoncrawl.py. Here, we edit this file to filter to the list of ~150 domains provided in domains.txt as by default CCNews has 10k+ domains. This list was created using Claude by providing it a seed list of important domains and then asking it to filter the full list of domains. 

To launch the above news extraction on MPI Cluster, we use ```jobs_news.py``` which calls ```launch_news_crawl_job.sh```. This streams warcs in a random-seeming order from the common crawl bucket, with a file (```fast/sgoel/forecasting/news/filtered_cc_warc/fullyextractedwarcs.list```) tracking previously extracted warcs (in case of resume) and also the date, domain filters being sequentially checked for each streamed warc before extracting it. The warc extraction can become a bottleneck, so while it may be tempting to increase number of extraction processes in ```jobs_news.py``` to a large number (like 96), this leads to the job hanging unexplicably. So I stuck to num_extractors=1 which is slow but reliable. 

This outputs each article in a separate json file in the directory specified by download_dir_article which was set to ```fast/sgoel/forecasting/news/filtered_cc_articles```. The large number of files can soon lead to cluster file quota limits. Since SQLite, LMDB don't work on cluster due to Lustre/NFS and not supporting flock (file locking mechanism), we simply combine articles from a single domain into a single jsonl file using ```to_jsonl.py``` launched using ```src/scripts/launch_jsonl_conversion.py```, which were saved at ```fast/sgoel/forecasting/news/filtered_cc_articles/jsonl/```. For continuing download from the same domains, make sure to remove the ```processed_dirs.txt``` file in the jsonl directory before launching the job again. Many downstream search/retrieval indexers work with json anyway so this is not too bad. Potential alternatives (not tried yet) are Parquet or Arrow (column query/filtering).

We stopped at 27M+ articles in 150+ domains, totalling 150GB+ of data. The extractor is technically going month by month to find relevant warcs, but for some reason some later warcs get downloaded earlier too. That's why in first complete run (27M articles) we saw many articles from [2017, 2021], and much lesser articles in 2022-2025. 2016 2nd half is when CCNews starts, and it has much fewer articles. This variance in number of retrieved articles can be a confounder in model performance "dropping for more recent questions even with retrieval" as shown in the Are LLMs Prescient paper if they stopped earlier (they had an arbitrary-ish 1.6M articles which seemed a bit too less for their chosen 4 domains). I have now re-launched pipeline by hardcoding start date as 2021-03-01 in examples/commoncrawl.py. [TODO: remove this later]

## News Processing
For BM25, tokenization is important. It is currently done using ```src/tokenize_for_rag.py``` launched using ```src/launch_tokenize_for_rag.py``` which calls ```src/launch_tokenize_for_rag_job.sh```. This adds a "tokenized_news" column to the old json structure, stores jsonl files in ```fast/sgoel/forecasting/news/tokenized_data/``` and deletes the original jsonls.

Similarly, deduplication can be done using ```src/deduplicate_news_jsonl.py``` launched using ```src/launch_dedup_news.py``` which calls ```src/launch_dedup_news.sh```. This stores the deduplicated jsonl files in the passed folder without dedup by creating a new folder there called ```deduped/```.

## BM25 Retrieval

Code in ```src/bm25_jsonl.py``` which can be launched using ```src/launch_bm25_retrieval.py``` which calls ```src/launch_bm25_retrieval.sh```. Needs the question dataset to have the tokenized_query and the articles jsonls to have the tokenized_news column. Dataset with a column for retrieved articles is stored in ```fast/sgoel/forecasting/news/retrieval/```.
