from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, split, lower, regexp_replace, count, regexp_extract, year, length, avg, desc, when, sum, min, max
from pyspark.sql.types import StructType, StructField, StringType
import re
import logging
import os
import glob
from datetime import datetime

# Configure logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, f"gutenberg_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Also print to console but with less detail
    ]
)
logger = logging.getLogger(__name__)

def create_spark_session():
    """Create and return a Spark session"""
    return SparkSession.builder \
        .appName("Gutenberg Analysis") \
        .getOrCreate()

def load_books(spark, input_path):
    """Load books from text files into a DataFrame"""
    # Get list of text files, excluding Zone.Identifier files
    valid_files = []
    for file in glob.glob(os.path.join(input_path, "*.txt")):
        if not file.endswith(":Zone.Identifier"):
            valid_files.append(file)
    
    if not valid_files:
        raise ValueError(f"No valid text files found in {input_path}")
    
    # Read the filtered list of files
    books_df = spark.read.text(valid_files)
    return books_df

def extract_metadata(books_df):
    """Extract metadata from book text using regular expressions"""
    
    title_pattern = r"Title:\s*([^\n]+)"
    release_date_pattern = r"Release Date:\s*(\w+\s+\d{1,2},\s*\d{4})"
    language_pattern = r"Language:\s*(\w+)"
    encoding_pattern = r"Character set encoding:\s*(\w+[-\w]*)"
    
    metadata_df = books_df.select(
        regexp_extract(col("value"), title_pattern, 1).alias("title"),
        regexp_extract(col("value"), release_date_pattern, 1).alias("release_date"),
        regexp_extract(col("value"), language_pattern, 1).alias("language"),
        regexp_extract(col("value"), encoding_pattern, 1).alias("encoding")
    )
    
    metadata_df = metadata_df.filter(
        (col("title") != "") | 
        (col("release_date") != "") | 
        (col("language") != "") | 
        (col("encoding") != "")
    )
    
    return metadata_df

def save_analysis_results(df, analysis_name, output_path):
    """Save DataFrame results to CSV and log summary"""
    df.write.mode("overwrite").csv(output_path)
    logger.info(f"Saved {analysis_name} results to {output_path}")
    logger.info(f"\n=== {analysis_name} Summary ===")
    df.show(20, truncate=False)

def analyze_metadata(metadata_df):
    """Analyze the extracted metadata"""
    logger.info("\n=== Gutenberg Dataset Analysis ===")
    
    # 1. Books Released Each Year
    logger.info("\n1. Books Released Per Year Analysis")
    year_analysis = metadata_df.filter(col("release_date").isNotNull()) \
        .select(regexp_extract(col("release_date"), r"\d{4}", 0).alias("year")) \
        .filter(col("year") != "") \
        .filter(col("year").rlike("^[12][0-9]{3}$"))
    
    year_counts = year_analysis \
        .groupBy("year") \
        .count() \
        .orderBy("year")
    
    # Year statistics
    year_stats = year_counts.agg(
        sum("count").alias("total_books"),
        min("year").alias("earliest_year"),
        max("year").alias("latest_year")
    ).collect()[0]
    
    logger.info(f"Book Release Statistics:")
    logger.info(f"- Total Books with Valid Release Dates: {year_stats['total_books']}")
    logger.info(f"- Publication Period: {year_stats['earliest_year']} to {year_stats['latest_year']}")
    
    # Show all years with their book counts
    logger.info("\nNumber of Books Released Each Year:")
    year_counts_list = year_counts.orderBy("year").collect()
    for row in year_counts_list:
        logger.info(f"{row['year']}: {row['count']} books")
    
    logger.info("\nTop 20 Years by Number of Books:")
    top_years = year_counts.orderBy(desc("count")).limit(20).collect()
    for row in top_years:
        logger.info(f"{row['year']}: {row['count']} books")
    
    # 2. Language Analysis
    logger.info("\n2. Language Analysis")
    language_analysis = metadata_df.filter(col("language").isNotNull()) \
        .filter(col("language") != "") \
        .groupBy("language") \
        .count() \
        .orderBy(desc("count"))
    
    # Get the most common language
    top_language = language_analysis.first()
    total_books = language_analysis.agg(sum("count")).collect()[0][0]
    
    logger.info(f"Language Statistics:")
    logger.info(f"- Most Common Language: {top_language['language']} ({top_language['count']} books)")
    logger.info(f"- Total Books with Language Info: {total_books}")
    logger.info("\nLanguage Distribution:")
    language_analysis.show(truncate=False)
    
    # 3. Title Length Analysis
    logger.info("\n3. Title Length Analysis")
    title_analysis = metadata_df.filter(col("title").isNotNull()) \
        .filter(col("title") != "") \
        .select(
            length(col("title")).alias("title_length")
        )
    
    title_stats = title_analysis.agg(
        avg("title_length").alias("avg_length"),
        min("title_length").alias("min_length"),
        max("title_length").alias("max_length"),
        count("title_length").alias("total_titles")
    ).collect()[0]
    
    logger.info(f"Title Length Statistics:")
    logger.info(f"- Average Title Length: {title_stats['avg_length']:.2f} characters")
    logger.info(f"- Shortest Title: {title_stats['min_length']} characters")
    logger.info(f"- Longest Title: {title_stats['max_length']} characters")
    logger.info(f"- Total Titles Analyzed: {title_stats['total_titles']}")
    
    # Distribution of title lengths in ranges
    logger.info("\nTitle Length Distribution:")
    title_length_dist = title_analysis \
        .withColumn("length_range", (col("title_length") / 10).cast("int") * 10) \
        .groupBy("length_range") \
        .count() \
        .orderBy("length_range")
    
    title_length_dist.show(truncate=False)
    
    # 4. Data quality report
    logger.info("\n=== Data Quality Report ===")
    total_books = metadata_df.count()
    missing_stats = metadata_df.select(
        [count(when(col(c).isNull() | (col(c) == ""), 1)).alias(f"missing_{c}")
         for c in metadata_df.columns]
    )
    missing_stats.show(truncate=False)

def process_words(books_df):
    """Process text and count words"""
    words_df = books_df \
        .select(explode(split(lower(col("value")), "\\s+")).alias("word")) \
        .filter(col("word") != "") \
        .filter(col("word").rlike("^[a-z']+$")) \
        .filter(col("word").rlike("^(?!.*'')"))\
        .groupBy("word") \
        .agg(count("*").alias("count")) \
        .orderBy(col("count").desc())
    
    return words_df

def main():
    spark = create_spark_session()
    logger.info("Spark session created")
    
    try:
        # Load books
        logger.info("Loading books...")
        books_df = load_books(spark, "/home/g23ai2032_ass1/D184MB/D184MB")
        books_df.cache()
        logger.info(f"Loaded {books_df.count()} books")
        
        # Extract and analyze metadata
        logger.info("Extracting metadata...")
        metadata_df = extract_metadata(books_df)
        metadata_df.cache()
        
        # Display metadata DataFrame summary
        logger.info("\n=== Metadata DataFrame Summary ===")
        logger.info(f"Total number of books: {metadata_df.count()}")
        logger.info("\nSample of metadata records:")
        metadata_df.show(5, truncate=False)
        
        logger.info("\nMetadata DataFrame Schema:")
        metadata_df.printSchema()
        
        # Analyze metadata
        analyze_metadata(metadata_df)
        
        # Save metadata results
        save_analysis_results(
            metadata_df, 
            "Metadata Analysis", 
            "/home/g23ai2032_ass1/metadata_results"
        )
        
        # Process and analyze words
        logger.info("Processing word frequencies...")
        words_df = process_words(books_df)
        
        # Save word count results
        save_analysis_results(
            words_df.limit(1000),  # Save top 1000 words
            "Word Frequency Analysis", 
            "/home/g23ai2032_ass1/word_counts"
        )
        
        logger.info(f"Analysis complete. Total unique words: {words_df.count()}")
        logger.info(f"Results saved to metadata_results and word_counts directories")
        logger.info(f"Full analysis log available at: {log_file}")
        
    finally:
        spark.stop()

if __name__ == "__main__":
    main()