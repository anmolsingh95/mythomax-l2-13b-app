import os

from fastapi_poe import make_app
from modal import Image, Secret, Stub, asgi_app

from mythomax import MythoMaxL213BBot

image = Image.debian_slim().pip_install_from_requirements("requirements.txt")
stub = Stub("mythomax-l2-13b-app")


@stub.function(image=image, secret=Secret.from_name("mythomax-l2-13b-secret"))
@asgi_app()
def fastapi_app():
    bot = MythoMaxL213BBot(
        model=os.environ["INFERENCE_ENDPOINT"],
        token=os.environ["HUGGINGFACE_ACCESS_TOKEN"],
    )
    app = make_app(bot, access_key=os.environ["POE_ACCESS_KEY"])
    return app
