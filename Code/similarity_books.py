from pyspark.sql import SparkSession
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml import Pipeline
from pyspark.sql.functions import (
    col, udf, regexp_extract, lower, regexp_replace, input_file_name, 
    explode, split, count, collect_list, size, log, lit, array, when, format_number, countDistinct,
    avg, trim, sum as _sum
)
from pyspark.sql.types import DoubleType, ArrayType, StringType, FloatType, StructType, StructField
from pyspark.ml.linalg import Vectors, VectorUDT
import numpy as np
from datetime import datetime
import logging
import os
import glob
import re


log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, f"book_similarity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_spark_session():
    """Create and return a Spark session"""
    spark = SparkSession.builder \
        .appName("BookSimilarity") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def load_books(spark, input_path):
    """Load books from the input directory."""
    
    base_dir = os.path.dirname(input_path.replace("*.txt", ""))
    
    
    txt_files = []
    for file in os.listdir(base_dir):
        if file.endswith(".txt") and not file.endswith("Zone.Identifier"):
            txt_files.append(os.path.join(base_dir, file))
    
    if not txt_files:
        raise ValueError(f"No valid .txt files found in {base_dir}")
    
    
    df = spark.read.text(txt_files)
    
    
    return df.withColumn("file_name", regexp_extract(input_file_name(), r"([^/]+)\.txt$", 1)) \
             .withColumnRenamed("value", "text")

def processing_text(books_df):
    """Clean and preprocess the text data"""
    
    print("\nSample of Original Text:")
    print("-" * 100)
    books_df.select("text").limit(1).show(truncate=100)
    
    
    def remove_gutenberg_text(text):
        
        start_markers = [
            r'\*\*\* START OF .+?\*\*\*',
            r'Project Gutenberg\'s.+?produced by.+?\n\n',
            r'This eBook is for the use of anyone.+?\n\n'
        ]
        end_markers = [
            r'\*\*\* END OF .+?\*\*\*.*',
            r'End of Project Gutenberg.+',
            r'End of the Project Gutenberg.+'
        ]
        
        
        for pattern in start_markers:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        
        
        for pattern in end_markers:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        
        return text.strip()
    
    remove_gutenberg_udf = udf(remove_gutenberg_text, StringType())
    
    
    df_steps = books_df \
        .withColumn("after_gutenberg", remove_gutenberg_udf(col("text"))) \
        .withColumn("after_lowercase", lower(col("after_gutenberg"))) \
        .withColumn("after_newlines", regexp_replace(col("after_lowercase"), "\\n+", " ")) \
        .withColumn("after_spaces", regexp_replace(col("after_newlines"), "\\s+", " ")) \
        .withColumn("after_nonalpha", regexp_replace(col("after_spaces"), "[^a-z\\s]", " ")) \
        .withColumn("after_shortwords", regexp_replace(col("after_nonalpha"), "\\b\\w{1,2}\\b", "")) \
        .withColumn("cleaned_text", trim(regexp_replace(col("after_shortwords"), "\\s+", " ")))
    
    
    print("\nSample After Removing Gutenberg Headers/Footers:")
    print("-" * 100)
    df_steps.select("after_gutenberg").limit(1).show(truncate=100)
    
    print("\nSample After Converting to Lowercase:")
    print("-" * 100)
    df_steps.select("after_lowercase").limit(1).show(truncate=100)
    
    print("\nSample After Removing Special Characters:")
    print("-" * 100)
    df_steps.select("after_nonalpha").limit(1).show(truncate=100)
    
    print("\nSample After Removing Short Words:")
    print("-" * 100)
    df_steps.select("after_shortwords").limit(1).show(truncate=100)
    
    print("\nFinal Cleaned Text:")
    print("-" * 100)
    df_steps.select("cleaned_text").limit(1).show(truncate=100)
    
    
    tokenizer = RegexTokenizer(
        inputCol="cleaned_text",
        outputCol="words",
        pattern="\\s+",
        minTokenLength=3  
    )
    words_df = tokenizer.transform(df_steps)
    
    
    print("\nSample of Tokenized Words:")
    print("-" * 100)
    words_df.select("words").limit(1).show(truncate=100)
    
    
    remover = StopWordsRemover(
        inputCol="words",
        outputCol="filtered_words",
        caseSensitive=False
    )
    
    filtered_df = remover.transform(words_df)
    
    
    print("\nSample After Stop Word Removal:")
    print("-" * 100)
    filtered_df.select("filtered_words").limit(1).show(truncate=100)
    
    
    print("\nWord Count Statistics:")
    print("-" * 100)
    filtered_df.select(
        size("words").alias("words_before_filtering"),
        size("filtered_words").alias("words_after_filtering")
    ).agg(
        avg("words_before_filtering").alias("avg_words_before"),
        avg("words_after_filtering").alias("avg_words_after")
    ).show()
    
    
    return filtered_df.select("file_name", "filtered_words")

def calculate_term_frequency(df):
    """
    Calculate Term Frequency (TF) for each word in each book
    TF = (Number of times term t appears in document) / (Total number of terms in document)
    """
    
    words_df = df.select("file_name", explode("filtered_words").alias("word"))
    
    
    word_counts = words_df \
        .groupBy("file_name", "word") \
        .agg(count("word").alias("term_count"))
    
    
    doc_lengths = words_df \
        .groupBy("file_name") \
        .agg(count("word").alias("total_terms"))
    
    
    tf_df = word_counts \
        .join(doc_lengths, "file_name") \
        .withColumn("tf", col("term_count") / col("total_terms")) \
        .select(
            "file_name",
            "word",
            "term_count",
            "total_terms",
            format_number(col("tf"), 6).alias("term_frequency")
        )
    
    
    print("\nSample Term Frequency (TF) Calculations:")
    print("-" * 100)
    print("Formula: TF = (Number of times term appears in document) / (Total number of terms in document)")
    print("-" * 100)
    
    
    sample_docs = tf_df.select("file_name").distinct().limit(3)
    
    for doc in sample_docs.collect():
        file_name = doc.file_name
        print(f"\nTop 10 frequent terms in {file_name}:")
        print("-" * 80)
        tf_df \
            .filter(col("file_name") == file_name) \
            .orderBy(col("tf").desc()) \
            .select(
                "word",
                "term_count",
                "total_terms",
                "term_frequency"
            ) \
            .show(10, truncate=False)
    
    return tf_df

def calculate_tf_idf(df):
  
    
    words_df = df.select("file_name", explode("filtered_words").alias("word"))
    
    
    tf_df = words_df \
        .groupBy("file_name", "word") \
        .agg(count("word").alias("term_count")) \
        .join(
            words_df.groupBy("file_name")
            .agg(count("word").alias("total_terms")),
            "file_name"
        ) \
        .withColumn("tf", col("term_count") / col("total_terms"))
    
    
    total_docs = df.select("file_name").distinct().count()
    
    idf_df = words_df \
        .groupBy("word") \
        .agg(countDistinct("file_name").alias("doc_freq")) \
        .withColumn(
            "idf",
            log(lit(total_docs) / col("doc_freq"))
        )
    
    
    tfidf_df = tf_df \
        .join(idf_df, "word") \
        .withColumn("tfidf", col("tf") * col("idf")) \
        .select(
            "file_name",
            "word",
            format_number(col("tf"), 4).alias("tf"),
            format_number(col("idf"), 4).alias("idf"),
            format_number(col("tfidf"), 4).alias("tfidf")
        )
    
    
    print("\nSample TF-IDF Calculations:")
    print("-" * 80)
    tfidf_df \
        .orderBy(col("tfidf").desc()) \
        .select("file_name", "word", "tf", "idf", "tfidf") \
        .show(10, truncate=False)
    
    return tfidf_df.select("file_name", "word", "tfidf")

def calculate_inverse_document_frequency(df):
    
    words_df = df.select("file_name", explode("filtered_words").alias("word"))
    
    total_docs = df.select("file_name").distinct().count()
    
    doc_freq_df = words_df \
        .groupBy("word") \
        .agg(
            countDistinct("file_name").alias("doc_freq"),
            count("word").alias("total_occurrences")
        ) \
        .withColumn(
            "idf",
            log(lit(total_docs) / col("doc_freq"))
        ) \
        .withColumn(
            "percent_docs",
            format_number(col("doc_freq") / lit(total_docs) * 100, 2)
        ) \
        .select(
            "word",
            "doc_freq",
            col("total_occurrences").alias("total_times_used"),
            col("percent_docs").alias("docs_containing_percent"),
            format_number("idf", 4).alias("idf_score")
        )
    
    
    print("\nInverse Document Frequency (IDF) Statistics:")
    print("-" * 100)
    print(f"Total number of documents analyzed: {total_docs}")
    print("\nFormula: IDF = log(Total_Documents / Documents_Containing_Word)")
    print("-" * 100)
    
    
    print("\nTop 10 Most Common Words (Appear in most documents):")
    print("-" * 100)
    doc_freq_df \
        .orderBy(col("doc_freq").desc()) \
        .show(10, truncate=False)
    
   
    print("\nTop 10 Most Rare Words (Appear in fewest documents):")
    print("-" * 100)
    doc_freq_df \
        .filter(col("doc_freq")>1)  \
        .orderBy(col("doc_freq").asc()) \
        .show(10, truncate=False)
    
    return doc_freq_df

def book_vectors_creation(tfidf_df):
    """
    Create vector representations of books using TF-IDF scores.
    Each book is represented as a sparse vector where each dimension
    corresponds to a word and the value is the TF-IDF score.
    """
    
    vocabulary = tfidf_df \
        .select("word") \
        .distinct() \
        .orderBy("word") \
        .collect()
    
    word_list = [row.word for row in vocabulary]
    word_to_index = {word: idx for idx, word in enumerate(word_list)}
    vocab_size = len(word_list)
    
    print(f"\nVocabulary Statistics:")
    print("-" * 80)
    print(f"Total unique words (vector dimensions): {vocab_size}")
    
    
    def create_sparse_vector(words, scores):
        
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        
        pairs = [(word_to_idx[word], float(score)) for word, score in zip(words, scores) if word in word_to_idx]
        
        pairs.sort(key=lambda x: x[0])
        
        indices = [p[0] for p in pairs]
        values = [p[1] for p in pairs]
        return Vectors.sparse(vocab_size, indices, values)

    create_vector_udf = udf(lambda w, s: create_sparse_vector(w, s), VectorUDT())
    
    
    book_vectors = tfidf_df \
        .groupBy("file_name") \
        .agg(
            collect_list("word").alias("words"),
            collect_list("tfidf").alias("scores")
        )
    
    print("\n=== Book TF-IDF Vector Representations ===")
    
    sample_vectors = book_vectors.limit(3).collect()
    for vector in sample_vectors:
        print(f"\nBook: {vector.file_name}")
        print("Top 10 words with highest TF-IDF scores:")
        word_scores = list(zip(vector.words, [float(s) for s in vector.scores]))
        word_scores.sort(key=lambda x: x[1], reverse=True)
        for word, score in word_scores[:10]:
            print(f"  {word}: {score:.4f}")

    
    print("\n=== Cosine Similarities Between Books ===")
    
    
    def cosine_similarity(vec1, vec2):
        
        dot_product = float(vec1.dot(vec2))
        
        norm1 = float(vec1.norm(2))
        norm2 = float(vec2.norm(2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    
    cosine_similarity_udf = udf(cosine_similarity, FloatType())

    
    vector_df = book_vectors.select(
        col("file_name"),
        create_vector_udf(col("words"), col("scores")).alias("features")
    )
    
    
    pairs_df = vector_df.crossJoin(vector_df.withColumnRenamed("file_name", "file_name2")
                                          .withColumnRenamed("features", "features2"))
    
    
    pairs_df = pairs_df.filter(col("file_name") < col("file_name2"))
    
    
    similarity_df = pairs_df.select(
        col("file_name"),
        col("file_name2"),
        cosine_similarity_udf(col("features"), col("features2")).alias("similarity")
    ).orderBy(col("similarity").desc())
    
    print("\nTop 10 Most Similar Book Pairs:")
    top_pairs = similarity_df.limit(10).collect()
    for pair in top_pairs:
        print(f"Books: {pair.file_name} - {pair.file_name2}")
        print(f"Similarity Score: {pair.similarity:.4f}\n")
    
    
    def vector_stats(vector):
        array = vector.toArray()
        nonzero = array[array != 0]
        return {
            'nonzero_count': str(len(nonzero)),
            'mean_tfidf': f"{float(nonzero.mean()):.4f}" if len(nonzero) > 0 else "0",
            'max_tfidf': f"{float(nonzero.max()):.4f}" if len(nonzero) > 0 else "0",
            'sparsity': f"{(1 - len(nonzero)/len(array))*100:.2f}%"
        }
    
    stats_schema = StructType([
        StructField("nonzero_count", StringType()),
        StructField("mean_tfidf", StringType()),
        StructField("max_tfidf", StringType()),
        StructField("sparsity", StringType())
    ])
    
    vector_stats_udf = udf(vector_stats, stats_schema)
    
   
    book_vectors = book_vectors \
        .withColumn("stats", vector_stats_udf(col("vector")))
    
    
    print("\nBook Vector Statistics:")
    print("-" * 80)
    book_vectors \
        .select(
            "file_name",
            col("stats.nonzero_count").alias("unique_terms"),
            col("stats.mean_tfidf").alias("mean_tfidf"),
            col("stats.max_tfidf").alias("max_tfidf"),
            col("stats.sparsity").alias("sparsity")
        ) \
        .show(truncate=False)
    
   
    print("\nMost Important Terms by TF-IDF Score:")
    print("-" * 80)
    sample_books = book_vectors.select("file_name").limit(3)
    
    for book in sample_books.collect():
        file_name = book.file_name
        print(f"\nTop terms for {file_name}:")
        print("-" * 60)
        
        tfidf_df \
            .filter(col("file_name") == file_name) \
            .orderBy(col("tfidf").desc()) \
            .select(
                "word",
                format_number("tfidf", 4).alias("tfidf_score")
            ) \
            .show(10, truncate=False)
    
    return book_vectors

def find_similar_books(tfidf_df, target_book_id, top_n=5):
    
    book_vectors = tfidf_df \
        .groupBy("file_name") \
        .agg(
            collect_list("word").alias("words"),
            collect_list("tfidf").alias("scores")
        )
    
    
    target_vector = book_vectors \
        .filter(col("file_name") == target_book_id) \
        .collect()
    
    if not target_vector:
        raise ValueError(f"Book {target_book_id} not found")
    
    target_vector = target_vector[0]
    target_words = target_vector.words
    target_scores = [float(s) for s in target_vector.scores]
    
    
    def cosine_similarity(words1, scores1, words2, scores2):
        
        scores_dict1 = dict(zip(words1, scores1))
        scores_dict2 = dict(zip(words2, scores2))
        
        
        all_words = set(words1) | set(words2)
        
     
        vector1 = []
        vector2 = []
        
        
        for word in all_words:
            vector1.append(float(scores_dict1.get(word, 0.0)))
            vector2.append(float(scores_dict2.get(word, 0.0)))
        
        
        dot_product = sum(v1 * v2 for v1, v2 in zip(vector1, vector2))
        norm1 = sum(v * v for v in vector1) ** 0.5
        norm2 = sum(v * v for v in vector2) ** 0.5
        
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    
    cosine_sim_udf = udf(
        lambda words, scores: cosine_similarity(target_words, target_scores, words, scores),
        FloatType()
    )
    
    
    similarities = book_vectors \
        .filter(col("file_name") != target_book_id) \
        .withColumn("similarity", cosine_sim_udf(col("words"), col("scores"))) \
        .select(
            "file_name",
            format_number("similarity", 4).alias("similarity")
        ) \
        .orderBy(col("similarity").desc()) \
        .limit(top_n)
    
    
    print(f"\nTop {top_n} Books Most Similar to {target_book_id}:")
    print("-" * 80)
    similarities.show(truncate=False)
    
    return similarities

def calculate_cosine_similarities(book_vectors_df):
    
    print("\nCalculating Pairwise Cosine Similarities:")
    print("-" * 80)
    
    
    pairs = book_vectors_df.crossJoin(book_vectors_df.select(
        col("file_name").alias("book2"),
        col("vector").alias("vector2")
    ))
    
   
    pairs = pairs.filter(col("file_name") < col("book2"))
    
    
    def cosine_similarity(v1, v2):
        
        a1 = v1.toArray()
        a2 = v2.toArray()
        
       
        dot_product = float(a1.dot(a2))
        
        
        norm1 = float(np.sqrt(a1.dot(a1)))
        norm2 = float(np.sqrt(a2.dot(a2)))
        
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    
    similarity_udf = udf(cosine_similarity, FloatType())
    
    
    similarities = pairs.select(
        col("file_name").alias("book1"),
        col("book2"),
        format_number(
            similarity_udf(col("vector"), col("vector2")),
            4
        ).alias("similarity")
    )
    
    print("\nSimilarity Score Statistics:")
    print("-" * 80)
    similarities \
        .select(
            format_number(avg("similarity"), 4).alias("average_similarity"),
            format_number(min("similarity"), 4).alias("min_similarity"),
            format_number(max("similarity"), 4).alias("max_similarity")
        ) \
        .show()
    
   
    print("\nMost Similar Book Pairs:")
    print("-" * 80)
    similarities \
        .orderBy(col("similarity").desc()) \
        .show(10, truncate=False)
    
    
    print("\nLeast Similar Book Pairs:")
    print("-" * 80)
    similarities \
        .orderBy(col("similarity").asc()) \
        .show(10, truncate=False)
    
    return similarities

def main():
    """Main function to run the book similarity analysis"""
    logger.info("Starting book similarity analysis...")
    
    try:
        
        spark = create_spark_session()
        logger.info("Spark session created")
        
        
        input_path = "/home/g23ai2032_ass1/D184MB/D184MB/*.txt"
        books_df = load_books(spark, input_path)
        logger.info(f"Loaded {books_df.count()} books")
        
       
        processed_df = processing_text(books_df)
        logger.info("Text preprocessing completed")
        
       
        tfidf_df = calculate_tf_idf(processed_df)
        logger.info("TF-IDF calculation completed")
        
        
        book_vectors = book_vectors_creation(tfidf_df)
        logger.info("Book vectors created")
        
        
        similarities = calculate_cosine_similarities(book_vectors)
        logger.info("Pairwise similarities calculated")
        
        processed_df.write.mode("overwrite").parquet("preprocessed_books")
        similarities.write.mode("overwrite").parquet("book_similarities")
        logger.info("Saved processed data to parquet files")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise
    finally:
        spark.stop()
        logger.info("Spark session stopped")

if __name__ == "__main__":
    main()