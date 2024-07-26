import azure.functions as func
import logging
import pickle
from scipy.spatial import distance
import numpy as np
import pandas as pd
import os
import json
import uuid
import time


app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
with open("all_article_ids.pkl", "rb") as f:
    all_article_ids = pickle.load(f)


@app.function_name(name="httpTrigger")
@app.route(
    route="users/{user_id:int?}",
    trigger_arg_name="req",
    binding_arg_name="$return",
    methods=[func.HttpMethod.GET],
)
# @app.blob_input(
#     arg_name="ratingFile",
#     path="model-articledata/ratings.pkl",
#     data_type="binary",
#     connection="AzureWebJobsStorage",
# )
# @app.blob_input(
#     arg_name="embeddingFile",
#     path="model-articledata/articles_embeddings_pca.pkl",
#     data_type="binary",
#     connection="AzureWebJobsStorage",
# )
@app.blob_input(
    arg_name="svdModel",
    path="model-articledata/svd++_algo.pkl",
    data_type="binary",
    connection="AzureWebJobsStorage",
)
def recommender_function(
    req: func.HttpRequest,
    # ratingFile: func.InputStream,
    # embeddingFile: func.InputStream,
    svdModel: func.InputStream,
) -> func.HttpResponse:
    svd_model = load_pickle_file(svdModel)
    logging.info(f"SVD Model loaded")
    try:
        user_id = req.route_params.get("user_id")
        logging.info(f"Received user_id: {user_id}")
        if not user_id:
            return func.HttpResponse(
                "User ID is required for recommendations", status_code=400
            )
        try:
            user_id_int = int(user_id)
        except ValueError:
            logging.error(f"Invalid user_id format: {user_id}")
            return func.HttpResponse("Invalid user ID format", status_code=400)

        # ratings = load_and_filter_data(ratingFile, user_id_int)
        # articles_emb = load_pickle_file(embeddingFile)

        # # all_article_ids = list(range(articles_emb.shape[0]))

        top_recommended = svd_function(user_id, all_article_ids, svd_model, n=5)

        # top_recommended = hybrid_recommendation(
        #     user_id_int, articles_emb, ratings, all_article_ids, svd_model
        # )

        return func.HttpResponse(
            body=f"For user_id: {user_id}, {top_recommended}",
            status_code=200,
        )

    except Exception as e:
        logging.error(f"Error in processing: {str(e)}")
        return func.HttpResponse(
            "An error occurred in the recommendation process", status_code=500
        )


def load_pickle_file(file_stream):
    file_content = file_stream.read()
    return pickle.loads(file_content)


# def load_and_filter_data(file_stream, user_id):
#     data = load_pickle_file(file_stream)
#     filtered_data = data[data["user_id"] == user_id]
#     return filtered_data


# def get_weighted_average_embeddings(user_articles, user_click_counts, articles_emb):
#     """
#     Cette fonction prend les IDs des articles, les comptes de clics, et les embeddings des articles,
#     et retourne un embedding moyen pondéré basé sur le nombre de fois qu'un utilisateur a interagi
#     avec chaque article.

#     Parameters:
#     - user_articles (list of int): Liste des IDs des articles avec lesquels un utilisateur a interagi.
#     - user_click_counts (list of int): Liste du nombre de fois qu'un utilisateur a interagi avec chaque article.
#     - articles_emb (array-like): Embeddings des articles.

#     Returns:
#     - array-like: L'embedding moyen pondéré des articles interactés par l'utilisateur.
#     """
#     total_weight = sum(user_click_counts)
#     weighted_average_embedding = np.zeros_like(articles_emb[0])
#     for article_id, click_count in zip(user_articles, user_click_counts):
#         weighted_average_embedding += articles_emb[article_id] * (
#             click_count / total_weight
#         )
#     return weighted_average_embedding


# def getnArticles(user_id, articles_emb, df_cliks, n=5):
#     """
#     Retourne une liste de n articles recommandés pour un utilisateur spécifié,
#     basée sur la similarité des embeddings des articles pondéré par le nombre de click de l'utilisateur.
#     Si l'utilisateur n'a lu aucun article, la fonction retourne les n articles les plus populaires.

#     Parameters:
#     userId (int): ID de l'utilisateur pour lequel les recommandations doivent être générées.
#     n (int): Nombre d'articles recommandés à récupérer.
#     articles_emb (array-like): Tableau 2D où chaque sous-liste
#                                 représente l'embedding d'un article.
#     df_cliks (DataFrame): Un DataFrame pandas contenant les données de clics des
#                           utilisateurs.

#     Return:
#     list of int: Liste des IDs des n articles recommandés.
#     """
#     user_articles = df_cliks.loc[df_cliks["user_id"] == user_id]["article_id"].tolist()
#     logging.info(print(user_articles))
#     user_click_counts = df_cliks.loc[df_cliks["user_id"] == user_id][
#         "article_user_clicks"
#     ].tolist()

#     if not user_articles:
#         print(f"User {user_id} has not read any articles yet!")
#         return

#     weighted_average_embedding = get_weighted_average_embeddings(
#         user_articles, user_click_counts, articles_emb
#     )
#     unread_articles = list(set(range(len(articles_emb))) - set(user_articles))
#     cos_distance = distance.cdist(
#         [weighted_average_embedding], articles_emb[unread_articles], "cosine"
#     )[0]
#     similarity_scores = 1 - cos_distance

#     recommended_indices = np.argsort(similarity_scores)[::-1][
#         :n
#     ]  # Tri décroissant pour avoir les scores les plus élevés en premier
#     recommended_articles_scores = [
#         (unread_articles[i], similarity_scores[i]) for i in recommended_indices
#     ]

#     return recommended_articles_scores


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


# def hybrid_recommendation(
#     user_id, articles_emb, df_clicks, all_article_ids, algo, n=5, w=0.5
# ):
#     """
#     Retourne une liste d'articles recommandés basée sur une combinaison linéaire des scores
#     de similarité de CBF et SVD++.
#     """

#     # Obtenir les recommandations de CBF
#     cbf_recommendations = getnArticles(user_id, articles_emb, df_clicks)
#     print(f"CBF recommandations : \n {cbf_recommendations}")

#     # Obtenir les recommandations de SVD++
#     svd_recommendations = svd_function(user_id, all_article_ids, algo, n=10)
#     print(f"SVD++ recommandations : \n {svd_recommendations}")

#     # Créer un dictionnaire pour stocker les scores combinés
#     combined_scores = {}
#     if cbf_recommendations:
#         for article_id, score in cbf_recommendations:
#             combined_scores[article_id] = w * score

#     for article_id, score in svd_recommendations:
#         if article_id in combined_scores:
#             combined_scores[article_id] += (1 - w) * score
#         else:
#             combined_scores[article_id] = (1 - w) * score

#     # Trier les articles par score combiné de manière décroissante
#     recommended_articles = sorted(
#         combined_scores.items(), key=lambda x: x[1], reverse=True
#     )
#     top_recommended = recommended_articles[:n]
#     print(f"Hybrid recommandations : \n {top_recommended}")

#     return top_recommended
