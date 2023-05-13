from gradio_client import Client

from PIL import Image


def test_api():
    client = Client("http://127.0.0.1:7860/")
    path = "imgs/002.jpg"
    print(client)
    # img = Image.open(path) 
    result = client.predict(
        path,
        # str representing input in 'Original Image' Image component
        0,  # int | float representing input in 'Tagging Confidence Threshold' Slider component
        128,  # int | float representing input in 'Image for Recognition' Slider component
        api_name="/run_pre"
    )
    # result = client.predict(
    #     "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png",
    #     # str representing input in 'Original Image' Image component
    #     0,  # int | float representing input in 'Tagging Confidence Threshold' Slider component
    #     128,  # int | float representing input in 'Image for Recognition' Slider component
    #     api_name="/run_pre"
    # )
    print(result)


test_api()
