import azure.functions as func
import logging
import pickle
import heapq

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


def load_pickle_file(file_stream):
    file_content = file_stream.read()
    return pickle.loads(file_content)


def load_all_article_ids():
    try:
        with open("all_article_ids.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logging.error(f"Failed to load all_article_ids: {str(e)}")
        return []


@app.function_name(name="httpTrigger")
@app.route(route="users/{user_id:int?}", methods=[func.HttpMethod.GET])
@app.blob_input(
    arg_name="svdModel",
    path="model-articledata/svd++_algo.pkl",
    data_type="binary",
    connection="AzureWebJobsStorage",
)
def recommender_function(
    req: func.HttpRequest, svdModel: func.InputStream
) -> func.HttpResponse:
    try:
        user_id = req.route_params.get("user_id")
        logging.info(f"Received User_id: {user_id}")
        if not user_id:
            return func.HttpResponse(
                "User ID is required for recommendations", status_code=400
            )

        algo = load_pickle_file(svdModel)
        all_article_ids = load_all_article_ids()
        top_recommended = svd_function(user_id, all_article_ids, algo, n=5)

        return func.HttpResponse(
            body=f"For user_id: {user_id}, top recommendations: {top_recommended}",
            status_code=200,
        )
    except Exception as e:
        logging.error(f"Error in recommender_function: {str(e)}")
        return func.HttpResponse(
            "An error occurred while processing your request", status_code=500
        )


def svd_function(user_id, all_article_ids, algo, n=5):
    predictions = [
        (algo.predict(uid=user_id, iid=article_id).est, article_id)
        for article_id in all_article_ids
    ]

    return [
        (article_id, -score) for score, article_id in heapq.nlargest(n, predictions)
    ]
