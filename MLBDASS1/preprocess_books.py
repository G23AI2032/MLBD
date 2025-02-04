import os
import re
import logging
from datetime import datetime
import pandas as pd
from tabulate import tabulate

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class BookPreprocessor:
    def __init__(self, directory_path):
        self.directory_path = directory_path
        
    def extract_author_date(self, text):
        # Common patterns for author extraction
        author_patterns = [
            r'by\s+([\w\s\.]+)',  # Matches "by Author Name"
            r'Author:\s*([\w\s\.]+)',  # Matches "Author: Author Name"
            r'written by\s+([\w\s\.]+)'  # Matches "written by Author Name"
        ]
        
        # Common patterns for date extraction
        date_patterns = [
            r'(\d{4})',  # Simple year
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}',
            r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY
            r'\d{4}-\d{2}-\d{2}'  # YYYY-MM-DD
        ]
        
        # Extract author
        author = None
        for pattern in author_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                author = match.group(1).strip()
                break
                
        # Extract date
        release_date = None
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                release_date = match.group(0)
                break
                
        return author, release_date
    
    def process_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                author, release_date = self.extract_author_date(content)
                
                # Get file size in MB
                file_size = os.path.getsize(file_path) / (1024 * 1024)
                
                return {
                    'File Name': os.path.basename(file_path),
                    'Author': author if author else 'Unknown',
                    'Release Date': release_date if release_date else 'Unknown',
                    'File Size (MB)': round(file_size, 2)
                }
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")
            return None
    
    def process_all_files(self):
        results = {}
        print(f"\nSearching for .txt files in: {os.path.abspath(self.directory_path)}")
        
        try:
            files = os.listdir(self.directory_path)
            print(f"Found {len(files)} total files")
            txt_files = [f for f in files if f.endswith('.txt')]
            print(f"Found {len(txt_files)} .txt files")
            
            for file_name in txt_files:
                file_path = os.path.join(self.directory_path, file_name)
                print(f"Processing: {file_name}")
                result = self.process_file(file_path)
                if result:
                    results[file_name] = result
                    logging.info(f"Processed {file_name}")
                else:
                    print(f"Failed to process: {file_name}")
            
            return results
        except Exception as e:
            print(f"Error in process_all_files: {str(e)}")
            return results

def display_results(results, top_n=20):
    if not results:
        print("\nNo results to display. No books were successfully processed.")
        return
        
    # Convert results to pandas DataFrame, properly handling the nested structure
    data = []
    for file_name, info in results.items():
        data.append(info)
    df = pd.DataFrame(data)
    
    # Debug: Print DataFrame info
    print("\nDataFrame Info:")
    print(df.info())
    print("\nDataFrame Columns:")
    print(df.columns.tolist())
    
    if df.empty:
        print("\nNo data to display. DataFrame is empty.")
        return
    
    # Display the table with all columns (no sorting or dropping)
    print(f"\nProcessed Books:")
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    
    # Display some statistics
    print("\nSummary Statistics:")
    print(f"Total number of books processed: {len(df)}")

def create_influence_network(results, years_threshold=5):
    """Create a network of author influences based on publication dates.
    
    An author A is considered to have potentially influenced author B if:
    1. Author A's book was published before Author B's book
    2. The time difference between their publications is within years_threshold
    
    Args:
        results: Dictionary of book information including authors and dates
        years_threshold: Maximum number of years between publications to consider influence
        
    Returns:
        List of tuples (author1, author2, year_diff) where author1 potentially influenced author2
    """
    influence_edges = []
    processed_dates = {}  # Keep track of processed dates for each author
    
    # First pass: collect all valid dates for each author
    for file_name, info in results.items():
        author = info.get('Author')
        date_str = info.get('Release Date')
        
        if not author or not date_str or author == 'Unknown':
            continue
            
        try:
            # Try different date formats
            for date_format in ['%Y', '%B %d, %Y', '%m/%d/%Y', '%Y-%m-%d']:
                try:
                    date = datetime.strptime(date_str, date_format)
                    if author not in processed_dates:
                        processed_dates[author] = []
                    processed_dates[author].append(date)
                    break
                except ValueError:
                    continue
        except Exception as e:
            print(f"Error processing date for {author}: {str(e)}")
            continue
    
    # Sort dates for each author
    for author in processed_dates:
        processed_dates[author] = sorted(set(processed_dates[author]))  # Remove duplicates
    
    # Second pass: create influence relationships
    authors = list(processed_dates.keys())
    for i, author1 in enumerate(authors):
        for author2 in authors[i+1:]:
            if author1 == author2:
                continue
                
            # Compare each pair of dates between the authors
            for date1 in processed_dates[author1]:
                for date2 in processed_dates[author2]:
                    year_diff = date2.year - date1.year
                    
                    # Only create influence edge if author1's book came before author2's
                    # and the time difference is within the threshold
                    if 0 < year_diff <= years_threshold:
                        influence_edges.append((author1, author2, year_diff))
                    elif 0 < -year_diff <= years_threshold:
                        influence_edges.append((author2, author1, -year_diff))
    
    print(f"\nCreated influence network with {len(influence_edges)} edges")
    
    # Convert to pandas DataFrame for better visualization
    if influence_edges:
        df = pd.DataFrame(influence_edges, columns=['Influencer', 'Influenced', 'Years_Between'])
        
        print(f"\nInfluence window in years : {years_threshold} years")
        print("\nSample of influence relationships (first 10 rows):")
        print(tabulate(df.head(5), headers='keys', tablefmt='grid', showindex=False))
        
        # Print statistics grouped by influencer
        print("\nTop 5 Most Influential Authors:")
        influencer_stats = df.groupby('Influencer').agg({
            'Influenced': 'count',
            'Years_Between': ['mean', 'min', 'max']
        }).round(2)
        influencer_stats.columns = ['Influence_Count', 'Avg_Years', 'Min_Years', 'Max_Years']
        influencer_stats = influencer_stats.sort_values('Influence_Count', ascending=False)
        print(tabulate(influencer_stats.head(), headers='keys', tablefmt='grid'))
        
        # Print statistics grouped by influenced author
        print("\nTop 5 Most Influenced Authors:")
        influenced_stats = df.groupby('Influenced').agg({
            'Influencer': 'count',
            'Years_Between': ['mean', 'min', 'max']
        }).round(2)
        influenced_stats.columns = ['Times_Influenced', 'Avg_Years', 'Min_Years', 'Max_Years']
        influenced_stats = influenced_stats.sort_values('Times_Influenced', ascending=False)
        print(tabulate(influenced_stats.head(), headers='keys', tablefmt='grid'))
        
        # Overall time difference statistics
        print(f"\nTime difference statistics:")
        print(f"Average time between influences: {df['Years_Between'].mean():.1f} years")
        print(f"Minimum time difference: {df['Years_Between'].min()} years")
        print(f"Maximum time difference: {df['Years_Between'].max()} years")
    
    return influence_edges

def analyze_influence_network(edge_df):
    """Analyze the influence network"""
    
    print("\nNetwork Analysis:")
    
    # Number of unique authors
    influencers = set(edge_df['influencer'])
    influenced = set(edge_df['influenced'])
    unique_authors = len(influencers.union(influenced))
    print(f"Number of unique authors in the network: {unique_authors}")
    
    # Distribution of year differences
    print("\nYear difference distribution:")
    year_diff_stats = edge_df['year_difference'].describe()
    print(year_diff_stats)

def calculate_degree_metrics(edge_df):
    """Calculate in-degree and out-degree metrics for each author"""
    try:
        # Calculate in-degree (number of influencers)
        in_degree = edge_df.groupby('influenced').size().reset_index()
        in_degree.columns = ['author', 'in_degree']
        
        # Calculate out-degree (number of authors influenced)
        out_degree = edge_df.groupby('influencer').size().reset_index()
        out_degree.columns = ['author', 'out_degree']
        
        # Merge the metrics
        degree_metrics = pd.merge(in_degree, out_degree, on='author', how='outer').fillna(0)
        
        # Calculate total degree
        degree_metrics['total_degree'] = degree_metrics['in_degree'] + degree_metrics['out_degree']
        
        # Display top 5 authors by in-degree
        print("\nTop 5 Most Influenced Authors (Highest In-Degree):")
        print("These authors were influenced by the most other authors")
        top5_in = degree_metrics.nlargest(5, 'in_degree')[['author', 'in_degree']]
        print(tabulate(top5_in, headers=['Author', 'Number of Influencers'], 
                      tablefmt='grid', showindex=False))
        
        # Display top 5 authors by out-degree
        print("\nTop 5 Most Influential Authors (Highest Out-Degree):")
        print("These authors influenced the most other authors")
        top5_out = degree_metrics.nlargest(5, 'out_degree')[['author', 'out_degree']]
        print(tabulate(top5_out, headers=['Author', 'Number of Authors Influenced'], 
                      tablefmt='grid', showindex=False))
        
        # Display top authors by total influence
        print("\nTop 10 Authors by Total Influence (In-degree + Out-degree):")
        top10_total = degree_metrics.nlargest(10, 'total_degree')[['author', 'in_degree', 'out_degree', 'total_degree']]
        print(tabulate(top10_total, 
                      headers=['Author', 'In-Degree', 'Out-Degree', 'Total Influence'],
                      tablefmt='grid', showindex=False))
        
        # Calculate and display some network statistics
        print("\nNetwork Degree Statistics:")
        print(f"Average In-Degree: {degree_metrics['in_degree'].mean():.2f}")
        print(f"Average Out-Degree: {degree_metrics['out_degree'].mean():.2f}")
        print(f"Maximum In-Degree: {degree_metrics['in_degree'].max()}")
        print(f"Maximum Out-Degree: {degree_metrics['out_degree'].max()}")
        
        return degree_metrics
        
    except Exception as e:
        print(f"Error in calculate_degree_metrics: {str(e)}")
        raise

def main():
    try:
        # Directory containing the book text files
        directory_path = "D184MB/D184MB"  # Using relative path
        
        # Print current working directory
        print(f"\nCurrent working directory: {os.getcwd()}")
        print(f"Looking for directory: {directory_path}")
        print(f"Absolute path: {os.path.abspath(directory_path)}")
        
        if not os.path.exists(directory_path):
            print(f"Error: Directory '{directory_path}' does not exist.")
            print("Please make sure the directory exists and contains .txt files.")
            return
            
        # List directory contents
        print("\nDirectory contents:")
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            if os.path.isfile(item_path):
                size = os.path.getsize(item_path) / (1024 * 1024)  # Convert to MB
                print(f"File: {item} ({size:.2f} MB)")
            else:
                print(f"Dir:  {item}")
            
        preprocessor = BookPreprocessor(directory_path)
        results = preprocessor.process_all_files()
        
        if not results:
            print("No text files were processed. Please check if the directory contains .txt files.")
            return
            
        # Display results in a table
        display_results(results, top_n=20)
        
        # Only continue with network analysis if we have results
        if results:
            # Create influence network edges
            influence_edges = create_influence_network(results, years_threshold=5)
            
            if influence_edges:
                # Import here to avoid dependency issues if Spark is not needed
                from influence_network import create_spark_network
                
                # Create and analyze Spark DataFrame
                edge_df = create_spark_network(influence_edges)
                print("\nAnalysis completed successfully!")
            else:
                print("\nNo influence relationships found in the dataset.")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()