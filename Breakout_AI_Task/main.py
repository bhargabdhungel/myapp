from concurrent.futures import ThreadPoolExecutor
import os
import pandas as pd
from typing import List, Dict
from tqdm import tqdm
import logging
from Keyword_extractor import extract_keywords
from WebScrapper import extract_top_website_text
from kg import KG
import pickle
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AllInOne:
    def __init__(self, csv_path: str, column: str, question: str, max_workers: int = 3):
        self.csv = pd.read_csv(csv_path)
        self.column = list(self.csv[column])
        self.question = question
        self.max_workers = max_workers
        self.cache_dir = "cache"
        self._setup_cache()
        
        logging.info("Extracting keywords from question")
        suffix = ' '.join(extract_keywords(self.question))
        self.searches = [f"{text} {suffix}" for text in self.column]

    def _setup_cache(self):
        """Set up cache directory"""
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_filename(self, search: str) -> str:
        """Generate a stable cache filename from search string"""
        # Create a stable hash of the search string
        hash_object = hashlib.md5(search.encode())
        return os.path.join(self.cache_dir, f"search_{hash_object.hexdigest()[:10]}.pkl")

    def _process_single_search(self, search: str) -> str:
        """Process a single search query with caching"""
        cache_file = self._get_cache_filename(search)
        
        if os.path.exists(cache_file):
            logging.info(f"Using cached result for: {search}")
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logging.warning(f"Failed to load cache for {search}: {str(e)}")
        
        try:
            logging.info(f"Processing search: {search}")
            # Extract website text
            extract_top_website_text(search, 3)
            
            # Query knowledge graph
            knowledge_g = KG('data')
            response = knowledge_g.query(self.question)  # This now returns a string
            
            # Clean up temporary files
            data_file = f"data/{search}.md"
            if os.path.exists(data_file):
                os.remove(data_file)
            
            # Cache the result
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(response, f)
            except Exception as e:
                logging.error(f"Failed to cache result for {search}: {str(e)}")
            
            return response
            
        except Exception as e:
            error_msg = f"Error processing {search}: {str(e)}"
            logging.error(error_msg)
            return error_msg

    def __call__(self) -> None:
        """Execute the pipeline with parallel processing"""
        logging.info("Starting parallel processing")
        
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                answers = list(tqdm(
                    executor.map(self._process_single_search, self.searches),
                    total=len(self.searches),
                    desc="Processing searches"
                ))
            
            self.csv['answers'] = answers
            
        except Exception as e:
            logging.error(f"Error in parallel processing: {str(e)}")
            raise

    def save_results(self, output_path: str) -> None:
        """Save results to CSV"""
        try:
            self.csv.to_csv(output_path, index=False)
            logging.info(f"Results saved to {output_path}")
        except Exception as e:
            logging.error(f"Error saving results: {str(e)}")
            raise

# if __name__ == "__main__":
#     try:
#         csv = "testing.csv"
#         col = "company"
#         question = "Wha"
#         ob = AllInOne(csv, column=col, question=question)
#         ob()
#         ob.save_results("Output.csv")
#     except Exception as e:
#         logging.error(f"Main execution failed: {str(e)}")