import sys
from apispec import APISpec
from apispec.ext.marshmallow import MarshmallowPlugin
import importlib.util
import json
import copy

mcquic_config = importlib.util.spec_from_file_location("mcquic.config", sys.argv[1])
config = importlib.util.module_from_spec(mcquic_config)
mcquic_config.loader.exec_module(config)

spec = APISpec(title="", version="1.0.0", openapi_version="3.0.2", plugins=[MarshmallowPlugin()])
spec.components.schema("result", schema=config.ConfigSchema)

result = spec.to_dict()["components"]["schemas"]

result.update({key: value for key, value in copy.deepcopy(result["result"]).items()})

del result["result"]


with open(sys.argv[2], "w") as fp:
    json.dump(result, fp)
