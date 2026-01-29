import base64
from io import BytesIO
from langchain_core.messages import HumanMessage


def get_image_response(image_obj, prompt, llms):

    buffered = BytesIO()
    image_obj.save(buffered, format="PNG")

    image_base64 = base64.b64encode(buffered.getvalue()).decode()

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                },
            },
        ]
    )

    response = llms["llm_vision"].invoke([message])

    return response.content