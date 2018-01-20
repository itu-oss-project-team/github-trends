from bigquery import get_client

from github_trends import secret_config


class BigQueryService:
    def __init__(self):
        self.__bigquery_config = secret_config['bigquery-api']
        self.__bigquery_client = None

    def get_bigquery_client(self, use_legacy_sql=False):
        if self.__bigquery_client is None:
            json_key = self.__bigquery_config['json_key']
            self.__bigquery_client = get_client(json_key=json_key, readonly=True)
        return self.__bigquery_client

    def execute_bigquery_select(self, query):
        client = self.get_bigquery_client()

        # Submit an async query.
        job_id, _results = client.query(query, use_legacy_sql=False)

        retry_count = 100
        complete = False
        while retry_count > 0 and not complete:
            complete, _ = client.check_job(job_id)
            retry_count -= 1
            time.sleep(10)

        # Retrieve the results.
        return client.get_query_rows(job_id)
