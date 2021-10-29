import os

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

# context = create_default_context(cafile="path/to/cert.pem")


if not os.getenv('ELASTICSEARCH_URL', None):
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env.gcp'), verbose=True)

# print( "os.getenv('ELASTICSEARCH_URL')", os.getenv('ELASTICSEARCH_URL') )
es = Elasticsearch(
    [ os.getenv('ELASTICSEARCH_URL') ],
    # sniff_on_start=True,
    timeout=60
)
