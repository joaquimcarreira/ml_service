import os
from django.core.wsgi import get_wsgi_application
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')
application = get_wsgi_application()

# ML registry
import inspect
from apps.ml.registry import MLRegistry
from apps.ml.movie_classifier.random_forest import RandomForestClassifier 

try:
    registry = MLRegistry() # create ML registry
    # Random Forest classifier
    rf = RandomForestClassifier()
    # add to ML registry
    registry.add_algorithm(endpoint_name="movie_classifier",
                            algorithm_object=rf,
                            algorithm_name="random forest",
                            algorithm_status="testing",
                            algorithm_version="2",
                            owner="Joaquim",
                            algorithm_description="Random Forest with simple pre- and post-processing",
                            algorithm_code=inspect.getsource(RandomForestClassifier))
    registry.add_algorithm(endpoint_name="movie_classifier",
                            algorithm_object=rf,
                            algorithm_name="rf",
                            algorithm_status="production",
                            algorithm_version="1",
                            owner="Joaquim",
                            algorithm_description="Random Forest with simple pre- and post-processing",
                            algorithm_code=inspect.getsource(RandomForestClassifier))
except Exception as e:
    print("Exception while loading the algorithms to the registry,", str(e))
