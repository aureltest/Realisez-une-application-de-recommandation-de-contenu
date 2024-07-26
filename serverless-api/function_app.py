import azure.functions as func
import logging
import pickle
from scipy.spatial import distance
import numpy as np
import pandas as pd
import uuid
import threading
import os
from typing import List, Tuple


app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

CACHE_DIR = "/home/cache"
CACHE_FILES = {
    "svd_model": "svd_model.pkl",
    "ratings": "ratings.pkl",
    "articles_emb": "articles_emb.pkl",
}
cache_lock = threading.Lock()
cache = {}
HYBRID_WEIGHT = 0.5
DEFAULT_RECOMMENDATION_COUNT = 5


def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def load_pickle_file(file_stream):
    file_content = file_stream.read()
    return pickle.loads(file_content)


def load_cache(ratingFile, embeddingFile, svdModel):
    ensure_cache_dir()

    data_sources = {
        "svd_model": svdModel,
        "ratings": ratingFile,
        "articles_emb": embeddingFile,
    }

    with cache_lock:
        for key, blob_source in data_sources.items():
            if key not in cache:
                cache_file = os.path.join(CACHE_DIR, CACHE_FILES[key])
                if os.path.exists(cache_file):
                    logging.info(f"Loading {key} from cache")
                    with open(cache_file, "rb") as f:
                        cache[key] = pickle.load(f)
                else:
                    logging.info(f"Loading {key} from blob and caching")
                    cache[key] = load_pickle_file(blob_source)
                    with open(cache_file, "wb") as f:
                        pickle.dump(cache[key], f)
                logging.info(f"{key.capitalize()} loaded into cache")

    logging.info("All data loaded into cache successfully")


@app.function_name(name="httpTrigger")
@app.route(
    route="users/{user_id:int?}",
    trigger_arg_name="req",
    binding_arg_name="$return",
    methods=[func.HttpMethod.GET],
)
@app.blob_input(
    arg_name="ratingFile",
    path="model-articledata/ratings.pkl",
    data_type="binary",
    connection="AzureWebJobsStorage",
)
@app.blob_input(
    arg_name="embeddingFile",
    path="model-articledata/articles_embeddings_pca.pkl",
    data_type="binary",
    connection="AzureWebJobsStorage",
)
@app.blob_input(
    arg_name="svdModel",
    path="model-articledata/svd++_algo.pkl",
    data_type="binary",
    connection="AzureWebJobsStorage",
)
def recommender_function(
    req: func.HttpRequest,
    ratingFile: func.InputStream,
    embeddingFile: func.InputStream,
    svdModel: func.InputStream,
) -> func.HttpResponse:
    request_id = str(uuid.uuid4())
    try:
        # Chargement du cache si nécessaire
        load_cache(ratingFile, embeddingFile, svdModel)
        logging.info(f"Model and files loaded from cache")

        user_id = req.route_params.get("user_id")
        logging.info(f"Received user_id: {user_id} for request ID: {request_id}")
        if not user_id:
            return func.HttpResponse(
                "User ID is required for recommendations", status_code=400
            )
        try:
            user_id_int = int(user_id)
        except ValueError:
            logging.error(f"Invalid user_id format: {user_id}")
            return func.HttpResponse("Invalid user ID format", status_code=400)

        filtered_ratings = cache["ratings"][cache["ratings"]["user_id"] == user_id_int]
        all_article_ids = list(range(cache["articles_emb"].shape[0]))

        top_recommended = hybrid_recommendation(
            user_id_int,
            cache["articles_emb"],
            filtered_ratings,
            all_article_ids,
            cache["svd_model"],
        )

        return func.HttpResponse(
            body=f"For user_id: {user_id}, {top_recommended}",
            status_code=200,
        )

    except Exception as e:
        logging.error(f"Error in processing: {str(e)}")
        return func.HttpResponse(
            "An error occurred in the recommendation process", status_code=500
        )


def get_weighted_average_embeddings(user_articles, user_click_counts, articles_emb):
    """
    Cette fonction prend les IDs des articles, les comptes de clics, et les embeddings des articles,
    et retourne un embedding moyen pondéré basé sur le nombre de fois qu'un utilisateur a interagi
    avec chaque article.

    Parameters:
    - user_articles (list of int): Liste des IDs des articles avec lesquels un utilisateur a interagi.
    - user_click_counts (list of int): Liste du nombre de fois qu'un utilisateur a interagi avec chaque article.
    - articles_emb (array-like): Embeddings des articles.

    Returns:
    - array-like: L'embedding moyen pondéré des articles interactés par l'utilisateur.
    """
    total_weight = sum(user_click_counts)
    weighted_average_embedding = np.zeros_like(articles_emb[0])
    for article_id, click_count in zip(user_articles, user_click_counts):
        weighted_average_embedding += articles_emb[article_id] * (
            click_count / total_weight
        )
    return weighted_average_embedding


def getnArticles(
    user_id: int,
    articles_emb: np.ndarray,
    df_cliks: pd.DataFrame,
    n: int = DEFAULT_RECOMMENDATION_COUNT,
) -> List[Tuple[int, float]]:
    """
    Retourne une liste de n articles recommandés pour un utilisateur spécifié,
    basée sur la similarité des embeddings des articles pondéré par le nombre de click de l'utilisateur.
    Si l'utilisateur n'a lu aucun article, la fonction retourne les n articles les plus populaires.

    Parameters:
    userId (int): ID de l'utilisateur pour lequel les recommandations doivent être générées.
    n (int): Nombre d'articles recommandés à récupérer.
    articles_emb (array-like): Tableau 2D où chaque sous-liste
                                représente l'embedding d'un article.
    df_cliks (DataFrame): Un DataFrame pandas contenant les données de clics des
                          utilisateurs.

    Return:
    list of int: Liste des IDs des n articles recommandés.
    """
    user_articles = df_cliks.loc[df_cliks["user_id"] == user_id]["article_id"].tolist()
    logging.info(print(user_articles))
    user_click_counts = df_cliks.loc[df_cliks["user_id"] == user_id][
        "article_user_clicks"
    ].tolist()

    if not user_articles:
        logging.info(f"User {user_id} has not read any articles yet!")
        return

    weighted_average_embedding = get_weighted_average_embeddings(
        user_articles, user_click_counts, articles_emb
    )
    unread_articles = list(set(range(len(articles_emb))) - set(user_articles))
    cos_distance = distance.cdist(
        [weighted_average_embedding], articles_emb[unread_articles], "cosine"
    )[0]
    similarity_scores = 1 - cos_distance

    recommended_indices = np.argsort(similarity_scores)[::-1][
        :n
    ]  # Tri décroissant pour avoir les scores les plus élevés en premier
    recommended_articles_scores = [
        (unread_articles[i], similarity_scores[i]) for i in recommended_indices
    ]

    return recommended_articles_scores


def svd_function(user_id, all_article_ids, algo, n=5):
    """
    Obtenir les prédictions de score pour un utilisateur et une liste d'articles en utilisant SVD++.

    Parameters:
    - user_id (int): L'identifiant de l'utilisateur pour lequel générer des prédictions.
    - all_article_ids (list of int): Une liste d'identifiants d'articles pour lesquels générer des prédictions.
    - algo (object): L'objet SVD++ entraîné.

    Returns:
    list of tuples: Liste des tuples (article_id, predicted_score).
    """
    predictions = []
    for article_id in all_article_ids:
        prediction = algo.predict(uid=user_id, iid=article_id)
        predictions.append((article_id, prediction.est))

    # Triez les prédictions en ordre décroissant de score
    predictions.sort(key=lambda x: x[1], reverse=True)

    return predictions[:n]


def hybrid_recommendation(
    user_id: int,
    articles_emb: np.ndarray,
    df_clicks: pd.DataFrame,
    all_article_ids: List[int],
    algo: object,
    n: int = DEFAULT_RECOMMENDATION_COUNT,
    w: float = HYBRID_WEIGHT,
) -> List[Tuple[int, float]]:
    """
    Retourne une liste d'articles recommandés basée sur une combinaison linéaire des scores
    de similarité de CBF et SVD++.
    """

    # Obtenir les recommandations de CBF
    cbf_recommendations = getnArticles(user_id, articles_emb, df_clicks)
    logging.info(f"CBF recommandations : \n {cbf_recommendations}")

    # Obtenir les recommandations de SVD++
    svd_recommendations = svd_function(user_id, all_article_ids, algo, n=10)
    logging.info(f"SVD++ recommandations : \n {svd_recommendations}")

    # Créer un dictionnaire pour stocker les scores combinés
    combined_scores = {}
    if cbf_recommendations:
        for article_id, score in cbf_recommendations:
            combined_scores[article_id] = w * score

    for article_id, score in svd_recommendations:
        if article_id in combined_scores:
            combined_scores[article_id] += (1 - w) * score
        else:
            combined_scores[article_id] = (1 - w) * score

    # Trier les articles par score combiné de manière décroissante
    recommended_articles = sorted(
        combined_scores.items(), key=lambda x: x[1], reverse=True
    )
    top_recommended = recommended_articles[:n]
    logging.info(f"Hybrid recommandations : \n {top_recommended}")

    return top_recommended
