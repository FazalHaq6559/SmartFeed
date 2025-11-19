from fastapi import FastAPI, HTTPException # type: ignore
from fastapi import Response  # type: ignore
from pydantic import BaseModel # type: ignore
from typing import List, Optional
import numpy as np # type: ignore
import pandas as pd # type: ignore
import tensorflow as tf # type: ignore
import pickle
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
import logging
from functools import lru_cache
from pymongo import MongoClient # type: ignore
from bson import ObjectId # type: ignore
from transformers import BertTokenizer, BertModel # type: ignore
import torch # type: ignore
import traceback
from pymongo.errors import ConnectionFailure # type: ignore
import time
import os
from functools import lru_cache
from bson.objectid import ObjectId # type: ignore
from bson.errors import InvalidId # type: ignore
import json
from datetime import datetime
# ---------------------------

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()

# Connect to MongoDB

def get_db_connection(max_retries=3, retry_delay=5):
    # Get the MongoDB URI from environment variables
    mongodb_uri = os.getenv('MONGODB_URI', 'mongodb+srv://Fazalhaq:F1a2z3l4*@smartfeed.rl4ig.mongodb.net/?retryWrites=true&w=majority&appName=SmartFeed')
    
    for attempt in range(max_retries):
        try:
            client = MongoClient(mongodb_uri)
            client.admin.command('ismaster')  # Check if the server is running
            print("MongoDB connection successful")
            return client
        except ConnectionFailure as e:
            print(f"Connection attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Could not connect to MongoDB.")
                raise

# Use this function to get your database connection
try:
    client = get_db_connection()
    db = client.test  # Replace 'test' with your actual database name
    user_collection = db['users']  # Ensure this matches your MongoDB Atlas collection name
    news_collection = db['news']  # Ensure this matches your MongoDB Atlas collection name
except ConnectionFailure:
    # Handle the error, maybe set up a fallback or exit gracefully
    print("Failed to connect to MongoDB after multiple attempts")






# Load BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Function to generate embeddings from BERT
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def generate_news_embeddings():
    news_embeddings_dict = {}
    for news in news_collection.find():
        # Use get() method with a default value for 'abstract'
        title = news.get('title', '')
        abstract = news.get('abstract', '')
        combined_text = f"{title} {abstract}".strip()
        
        # Generate embeddings using your existing function
        embedding = get_embedding(combined_text)
        
        news_id = str(news['_id'])  # Use the MongoDB ObjectId as the key
        news_embeddings_dict[news_id] = embedding
        
        # Update the MongoDB document with the generated embeddings
        news_collection.update_one(
            {'_id': news['_id']},  # Find the document by its _id
            {'$set': {'embeddings': embedding.tolist()}}  # Set the embeddings field
        )
        
    
    # Save the embeddings dictionary to a .pkl file
    with open('hybrid_model/news_embeddings.pkl', 'wb') as f:
        pickle.dump(news_embeddings_dict, f)
    
    logger.info("News embeddings generated and saved.")
    return news_embeddings_dict

# Load or generate news embeddings
try:
    with open('hybrid_model/news_embeddings.pkl', 'rb') as f:
        news_embeddings_dict = pickle.load(f)
    logger.info("News embeddings loaded successfully.")
except FileNotFoundError:
    logger.info("News embeddings not found. Generating new embeddings.")
    news_embeddings_dict = generate_news_embeddings()


def load_news_mapping():
    try:
        with open('hybrid_model/news_id_mapping.pkl', 'rb') as f:
            raw_mapping = pickle.load(f)
            # Keep the mapping as is - don't try to convert to int32
            return raw_mapping
    except Exception as e:
        logger.error(f"Error loading news mapping: {e}")
        raise
    
    
def extract_news_id(news_dict):
    """Extract the numeric ID from a news dictionary or return the value if it's already a number."""
    if isinstance(news_dict, dict):
        # Assuming the dictionary has a numeric ID field - adjust the key based on your actual structure
        return news_dict.get('ncf_id') or news_dict.get('id') or 0
    return news_dict

def recommend_for_user(user_id, top_n=10):
    try:
        user_idx = user_id_mapping.get(user_id)
        if user_idx is None:
            logger.warning(f"User ID {user_id} not found in mapping. Adding new user.")
            update_user_mapping(user_id)
            user_idx = user_id_mapping[user_id]
        
        # Extract numeric IDs from the news mapping
        news_indices = [extract_news_id(v) for v in news_id_mapping.values()]
        
        # Log the first few indices to verify correct extraction
        logger.debug(f"Sample news indices: {news_indices[:5]}")
        
        # Convert to numpy arrays with explicit dtype
        try:
            user_indices = np.array([user_idx] * len(news_indices), dtype=np.int32)
            news_indices_array = np.array(news_indices, dtype=np.int32)
        except Exception as e:
            logger.error(f"Error converting arrays: {e}")
            logger.error(f"Sample user_idx: {user_idx}")
            logger.error(f"Sample news indices: {news_indices[:5]}")
            raise
        
        # Convert to TensorFlow tensors
        user_tensor = tf.convert_to_tensor(user_indices)
        news_tensor = tf.convert_to_tensor(news_indices_array)
        
        # Make predictions
        predicted_probs = ncf_model.predict([user_tensor, news_tensor])
        
        # Create recommendations DataFrame
        recommendations = pd.DataFrame({
            'newsId': list(news_id_mapping.keys()),
            'predicted_prob': predicted_probs.flatten()
        })
        
        return recommendations.sort_values(by='predicted_prob', ascending=False).head(top_n)
    
    except Exception as e:
        logger.error(f"Error in recommend_for_user: {str(e)}", exc_info=True)
        raise ValueError(f"Error generating recommendations: {str(e)}")

# Update the initialization section with more detailed logging
try:
    ncf_model = tf.keras.models.load_model('hybrid_model/ncf_model.keras')
    
    # Load and log user mapping
    with open('hybrid_model/user_id_mapping.pkl', 'rb') as f:
        user_id_mapping = pickle.load(f)
        logger.info(f"User ID mapping type: {type(user_id_mapping)}")
        if user_id_mapping:
            sample_user = next(iter(user_id_mapping.items()))
            logger.info(f"Sample user mapping: {sample_user}")
    
    # Load and log news mapping
    with open('hybrid_model/news_id_mapping.pkl', 'rb') as f:
        news_id_mapping = pickle.load(f)
        logger.info(f"News ID mapping type: {type(news_id_mapping)}")
        if news_id_mapping:
            sample_news = next(iter(news_id_mapping.items()))
            logger.info(f"Sample news mapping: {sample_news}")
            logger.info(f"Sample news value type: {type(sample_news[1])}")
            if isinstance(sample_news[1], dict):
                logger.info(f"Sample news value keys: {sample_news[1].keys()}")
    
    logger.info("Model and mappings loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model or mappings: {e}")
    raise

def find_similar_news(user_embedding, top_n=10):
    similar_news = []
    for news_id, news_embedding in news_embeddings_dict.items():
        similarity = cosine_similarity([user_embedding], [news_embedding])[0][0]
        similar_news.append((news_id, similarity))
    return sorted(similar_news, key=lambda x: x[1], reverse=True)[:top_n]






# Update the news_id_mapping loading to ensure integer types

# Modify the generate_news_embeddings function to update news_id_mapping

# Modify the update_news_mapping function
def update_news_mapping(news_id):
    global news_id_mapping  # Ensure we're modifying the global variable
    if news_id not in news_id_mapping:
        news_id_mapping[news_id] = len(news_id_mapping)
        with open('hybrid_model/news_id_mapping.pkl', 'wb') as f:
            pickle.dump(news_id_mapping, f)
        logger.info(f"Updated news mapping for ID {news_id}")


def get_user_embedding(user_history):
    embeddings = []
    for news_id in user_history:
        if news_id in news_embeddings_dict:
            embeddings.append(news_embeddings_dict[news_id])
    
    if not embeddings:
        logger.warning("No valid news items found in user history for embedding calculation.")
        return None
    
    user_embedding = np.mean(embeddings, axis=0)
    return user_embedding





# Update the hybrid_recommendations function
def is_valid_objectid(id_string):
    try:
        ObjectId(id_string)
        return True
    except InvalidId:
        return False

def fetch_news_details(final_recs, news_collection, news_dataset):
    # Ensure final_recs is a DataFrame
    if not isinstance(final_recs, pd.DataFrame):
        final_recs = pd.DataFrame(final_recs)
    
    # Validate newsId column
    if 'newsId' not in final_recs.columns:
        raise ValueError("No 'newsId' column in recommendations")
    
    news_ids = final_recs['newsId'].tolist()
    
    # Separate IDs by source
    dataset_news_ids = [nid for nid in news_ids if nid.startswith('N')]
    mongodb_news_ids = [nid for nid in news_ids if not nid.startswith('N')]
    
    # Fetch news from dataset
    dataset_news = news_dataset[news_dataset['newsId'].isin(dataset_news_ids)].copy()
    
    # Fetch news from MongoDB
    mongodb_query = {'_id': {'$in': [ObjectId(nid) for nid in mongodb_news_ids if ObjectId.is_valid(nid)]}}
    mongodb_news_details = list(news_collection.find(mongodb_query))
    mongodb_news_df = pd.DataFrame(mongodb_news_details)
    
    # Convert MongoDB '_id' to string if needed
    if '_id' in mongodb_news_df.columns:
        mongodb_news_df['newsId'] = mongodb_news_df['_id'].astype(str)
    
    # Combine dataset and MongoDB news
    combined_news = pd.concat([dataset_news, mongodb_news_df], ignore_index=True)
    
    # Select and clean required columns
    required_columns = ['newsId', 'category', 'title', 'description', 'url', 'entities']
    combined_news = combined_news[required_columns]
    
    # Handle NaN values
    combined_news = combined_news.replace({np.nan: None})
    
    # Reorder to match the original recommendation order
    combined_news = combined_news.set_index('newsId').loc[news_ids].reset_index()
    
    combined_news['publishedAt'] = datetime.now()  # Default to current time if not available
    combined_news['author'] = 'Unknown'  # Default author
    combined_news['urlToImage'] = ''  # Default image URL
    combined_news['source'] = {'name': 'SmartFeed'}  # Default source
    
    return combined_news

def json_safe_formatter(row):
    """
    Safely format news item for frontend display
    """
    # Convert row to dictionary if it's a Series
    if hasattr(row, 'to_dict'):
        row = row.to_dict()
    
    # Handle case where row might be a float or other non-dictionary type
    if not isinstance(row, dict):
        return {
            "title": "",
            "description": "",
            "url": "",
            "urlToImage": "",
            "publishedAt": datetime.now().isoformat(),
            "author": "Unknown",
            "source": {"name": "SmartFeed"}
        }
    
    return {
        "title": row.get('title', ''),
        "description": row.get('description', ''),
        "url": row.get('url', ''),
        "urlToImage": row.get('urlToImage', ''),
        "publishedAt": (row.get('publishedAt') or datetime.now()).isoformat(),
        "author": row.get('author', 'Unknown'),
        "source": {
            "name": row.get('source', {}).get('name', 'SmartFeed') 
                    if isinstance(row.get('source'), dict) 
                    else 'SmartFeed'
        }
    }

    
    
def hybrid_recommendations(user_id: str, preferences: tuple, history: tuple, top_n: int = 10):
    try:
        logger.info(f"Generating hybrid recommendations for User ID: {user_id}")
       

        # Step 1: Get NCF recommendations
        ncf_recs = recommend_for_user(user_id, top_n * 2)  # Get more recommendations initially
        logger.info(f"NCF recommendations shape: {ncf_recs.shape}")
        logger.info(f"NCF recommendations columns: {ncf_recs.columns}")

        # # Ensure 'newsId' column exists
        if 'news_id' in ncf_recs.columns:
            ncf_recs = ncf_recs.rename(columns={'news_id': 'newsId'})
        elif 'newsId' not in ncf_recs.columns:
            raise ValueError("Neither 'newsId' nor 'news_id' column found in NCF recommendations")

        # Step 2: Handle user history and generate content-based recommendations
        if not history:
            logger.info(f"No history for User ID: {user_id}. Using NCF recommendations only.")
            final_recs = ncf_recs
        else:
            user_embedding = get_user_embedding(history)
            if user_embedding is None:
                logger.warning(f"Unable to generate user embedding for User ID: {user_id}")
                final_recs = ncf_recs
            else:
                similar_news = find_similar_news(user_embedding, top_n=top_n * 2)
                content_recs = pd.DataFrame(similar_news, columns=['newsId', 'similarity'])
                final_recs = pd.concat([ncf_recs, content_recs])
                final_recs = final_recs.drop_duplicates('newsId')

        logger.info(f"Recommendations before filtering: {len(final_recs)}")

        # Step 3: Fetch full news details
        news_ids = final_recs['newsId'].tolist()
        logger.info(f"Extracted news IDs: {news_ids}")
       
        dst = pd.read_csv(
            r"C:\Users\USER\Desktop\FYP\SmartFeed\model_service\app\news.tsv",
            sep='\t', 
            quoting=3,  # Disable quoting
            on_bad_lines='skip',
            low_memory=False,
            encoding='utf-8',
            header=None,  # Treat first row as data, not headers
            names=['newsId', 'category', 'tags', 'title', 'description', 'url', 'entities', 'extra']  # Specify column names
            )
        
        print(dst.columns)
        final_recs = fetch_news_details(final_recs, news_collection, dst )

        # Step 4: Apply preference-based filtering
        if preferences:
            filtered_recs = final_recs[final_recs['category'].isin(preferences)]
            if filtered_recs.empty:
                logger.warning("No recommendations match user preferences. Using unfiltered recommendations.")
            else:
                final_recs = filtered_recs
        logger.info(f"Recommendations after category filtering: {len(final_recs)}")

        # Step 5: Filter out articles from the user's history
        if history:
            history_set = set(history[-100:])  # Consider only the last 100 items in history
            final_recs = final_recs[~final_recs['newsId'].isin(history_set)]
        logger.info(f"Recommendations after history filtering: {len(final_recs)}")

        # # Step 6: Ensure minimum number of recommendations
        # min_recommendations = 10
        # if len(final_recs) < min_recommendations:
        #     logger.warning(f"Less than {min_recommendations} recommendations after filtering. Adding more recommendations.")
        #     additional_recs = recommend_for_user(user_id, top_n * 3)
        #     if 'news_id' in additional_recs.columns:
        #         additional_recs = additional_recs.rename(columns={'news_id': 'newsId'})
        #     additional_recs = additional_recs[~additional_recs['newsId'].isin(final_recs['newsId'])]
        #     additional_news = list(news_collection.find({'newsId': {'$in': additional_recs['newsId'].tolist()}}))
        #     additional_df = pd.DataFrame(additional_news)
        #     if '_id' in additional_df.columns:
        #         additional_df['newsId'] = additional_df['_id'].astype(str)
        #     final_recs = pd.concat([final_recs, additional_df])

        # final_recs = final_recs.sort_values('predicted_prob' if 'predicted_prob' in final_recs.columns else 'similarity', ascending=False)
        # final_recs = final_recs.head(top_n)

        # Step 7: Format the recommendations for output
        # Use the new formatter
         # Convert to list of dictionaries, handling potential type issues
        formatted_recs = [
            json_safe_formatter(row) if isinstance(row, (dict, pd.Series)) else 
            json_safe_formatter({'title': str(row)}) 
            for row in final_recs.to_dict('records')
        ]
        
        return formatted_recs

    except Exception as e:
        logger.error(f"Error in hybrid_recommendations: {str(e)}", exc_info=True)
        raise
    
    
    
cached_hybrid_recommendations = lru_cache(maxsize=128)(hybrid_recommendations)

class UserRecommendationRequest(BaseModel):
    user_id: str
    preferences: List[str]
    history: List[str]
    
    
@app.post("/recommend/hybrid")
async def recommend_hybrid(request: UserRecommendationRequest, top_n: int = 10):
    try:
        recommendations = hybrid_recommendations(
            request.user_id,
            tuple(request.preferences),
            tuple(request.history),
            top_n
        )
        
        return {
            "status": "success",
            "user_id": request.user_id,
            "recommendations": recommendations
        }
    
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Recommendation generation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
    
# --------------------User Maping ----------------------------------
# ___________________________________________________________________


class UserData(BaseModel):
    user_id: str

def save_user_mapping():
    try:
        with open('hybrid_model/user_id_mapping.pkl', 'wb') as f:
            pickle.dump(user_id_mapping, f)
        logger.info("User mapping saved successfully")
    except Exception as e:
        logger.error(f"Error saving user mapping: {str(e)}")
        raise
    
def user_exists_in_db(user_id):
    try:
        user = user_collection.find_one({'_id': ObjectId(user_id)})
        logger.info(f"Checking user existence for ID {user_id}: {'Found' if user else 'Not found'}")
        return user is not None
    except Exception as e:
        logger.error(f"Error checking user existence: {str(e)}")
        raise

def update_user_mapping(user_id):
    try:
        if user_id not in user_id_mapping:
            user_id_mapping[user_id] = len(user_id_mapping)
            save_user_mapping()
            logger.info(f"Updated user mapping for ID {user_id}")
        else:
            logger.info(f"User ID {user_id} already in mapping")
    except Exception as e:
        logger.error(f"Error updating user mapping: {str(e)}")
        raise

@app.post("/update_user_mapping")
async def update_user_mapping_endpoint(user_data: UserData):
    try:
        logger.info(f"Received request to update user mapping for ID: {user_data.user_id}")
        if user_exists_in_db(user_data.user_id):
            update_user_mapping(user_data.user_id)
            return {
                "message": "User mapping updated",
                "user_index": user_id_mapping[user_data.user_id]
            }
        else:
            logger.warning(f"User not found in database: {user_data.user_id}")
            raise HTTPException(status_code=404, detail="User not found in database")
    except Exception as e:
        logger.error(f"Error updating user mapping for user_id {user_data.user_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    
if __name__ == "__main__":
    import uvicorn # type: ignore
    uvicorn.run(app, host="0.0.0.0", port=8000)
    