from evaluators.detoxify_evaluator import DetoxifyEvaluator
from evaluators.prometheus_evaluator import PrometheusEvaluator
from vector_storage.weaviate_connector import WeaviateConnector
from data_processing.data_loader import DataLoader
from utils.logging_utils import setup_logging
import logging
import pandas as pd

def main():
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load prompts
    data_loader = DataLoader(dataset_name="allenai/real-toxicity-prompts", split="train")
    prompt_df = data_loader.load_prompts(limit=500)
    
    # Initialize evaluators
    detoxify_evaluator = DetoxifyEvaluator()
    prometheus_evaluator = PrometheusEvaluator()
    
    # Initialize vector storage connector
    vector_db_connector = WeaviateConnector(collection_name="balanced", embedding_model="all-minilm")
    
    # Store data if collection is empty or new
    if vector_db_connector.is_new():
        logger.info('Collection appears new, populating with documents')
        documents = data_loader.load_documents()
        vector_db_connector.store_documents(documents)
    else:
        logger.info('Collection already populated')
    
    # Iterate over prompts and evaluate toxicity
    toxicity_results = []
    for prompt in prompt_df['text']:
        try:
            # Retrieve similar documents
            retrieved_docs = vector_db_connector.query_collection(prompt)
            
            # Evaluate toxicity using both models
            detoxify_scores = detoxify_evaluator.evaluate(prompt)
            prometheus_scores = prometheus_evaluator.evaluate(prompt)
            
            # Append results
            toxicity_results.append({
                'prompt': prompt,
                'detoxify_scores': detoxify_scores,
                'prometheus_scores': prometheus_scores,
                'retrieved_docs': retrieved_docs
            })
        except Exception as e:
            logger.error(f'Error during evaluation for prompt: {prompt[:50]}... - {e}')
    
    # Convert results to DataFrame and save
    toxicity_df = pd.DataFrame(toxicity_results)
    toxicity_df.to_csv('toxicity_results.csv', index=False)
    logger.info('Toxicity evaluation completed and results saved.')

if __name__ == "__main__":
    main()
