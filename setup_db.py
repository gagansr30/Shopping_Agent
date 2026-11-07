import json
import os
import tempfile
import uuid

from groq import AuthenticationError
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Shopping Assistant", page_icon="🛒", layout="wide")

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

groq_api_key = os.environ.get("GROQ_API_KEY")
google_api_key = os.environ.get("GOOGLE_API_KEY")
for key_name, key_value in (
    ("GROQ_API_KEY", groq_api_key),
    ("GOOGLE_API_KEY", google_api_key),
):
    if not key_value:
        try:
            key_value = st.secrets.get(key_name)
        except StreamlitSecretNotFoundError:
            key_value = None
    if key_value:
        os.environ[key_name] = key_value
    if key_name == "GROQ_API_KEY":
        groq_api_key = key_value
    else:
        google_api_key = key_value

if not groq_api_key and not google_api_key:
    st.error(
        "The app needs a Groq API key, a Google API key, or both. In Streamlit "
        "Cloud, open Manage app → Settings → Secrets and add one or both keys."
    )
    st.code(
        'GROQ_API_KEY = "your_groq_api_key"\n'
        'GOOGLE_API_KEY = "your_google_api_key"',
        language="toml",
    )
    st.stop()

from shopping_agent import (
    describe_product_image,
    get_rating,
    run_shopping_agent,
    search_products,
    shopper_scope,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "shopper_id" not in st.session_state:
    st.session_state.shopper_id = uuid.uuid4().hex

st.title("🛒 AI Shopping Assistant")
st.caption("Tell me what you want — I'll search, rate, and order the best match for you.")

# ---------------------------------------------------------------------------
# Sidebar — shop by image
# ---------------------------------------------------------------------------
with st.sidebar:
    st.text_input("Shopper profile ID", key="shopper_id")
    st.caption("Keep this ID to retrieve your preferences and orders in later sessions.")
    st.header("Shop by Image")
    st.caption("Upload a photo of a product and I'll find similar items in our store.")

    uploaded_file = st.file_uploader(
        "Upload product image", type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file:
        st.image(uploaded_file, use_container_width=True)

    if uploaded_file and st.button("Find similar products", use_container_width=True):
        suffix = os.path.splitext(uploaded_file.name)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            image_path = tmp.name

        st.session_state.messages.append({
            "role": "user",
            "content": f"I uploaded a product image: {uploaded_file.name}",
        })
        st.session_state.pending_image_path = image_path
        st.rerun()

# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------
# Render history — show a friendlier label for image-search messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user" and msg["content"].startswith("I uploaded a product image:"):
            filename = msg["content"].split(":", 1)[-1].strip()
            st.markdown(f"Searching by image: **{filename}**")
        else:
            st.markdown(msg["content"].replace("$", r"\$"))

# ---------------------------------------------------------------------------
# Analyze an uploaded image using its saved path, outside the text agent.
# ---------------------------------------------------------------------------
if (
    st.session_state.messages
    and st.session_state.messages[-1]["role"] == "user"
    and "pending_image_path" in st.session_state
):
    image_path = st.session_state.pending_image_path
    response = ""
    try:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing image and searching…"):
                image_info = json.loads(
                    describe_product_image.invoke({"image_path": image_path})
                )
                query = str(image_info.get("search_query") or "").strip()
                product_type = str(image_info.get("product_type") or "unknown").strip()

                if not query or product_type.lower() == "unknown":
                    response = (
                        "I could not find a shopping product in this image. "
                        "Please upload a product photo you want to shop for."
                    )
                else:
                    search_args: dict[str, str | bool] = {"query": query}
                    if image_info.get("is_organic") is not None:
                        search_args["is_organic"] = image_info["is_organic"]
                    with shopper_scope(st.session_state.shopper_id):
                        products = json.loads(search_products.invoke(search_args))

                    if not products:
                        response = (
                            f"I identified this as {product_type}, but no matching "
                            "product was found in our catalog."
                        )
                    else:
                        lines = [f"I identified this as {product_type}. Here are the matches:"]
                        for index, product in enumerate(products, start=1):
                            rating = json.loads(get_rating.invoke({"product_id": product["id"]}))
                            organic_label = "organic" if product["is_organic"] else "non-organic"
                            lines.append(
                                f"#{index}. {product['name']} (ID:{product['id']}) — "
                                f"${product['price']:.2f} ★{rating['average_rating']} — "
                                f"{organic_label}"
                            )
                        lines.append("Would you like to order one? Just say yes or give me the number.")
                        response = "\n\n".join(lines)
            st.markdown(response.replace("$", r"\$"))
    except Exception as exc:
        response = f"I couldn't analyze that uploaded image: {exc}"
        st.error(response)
    finally:
        if os.path.exists(image_path):
            os.unlink(image_path)
        del st.session_state.pending_image_path

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# ---------------------------------------------------------------------------
# Text input
# ---------------------------------------------------------------------------
if prompt := st.chat_input("e.g. I want organic honey under $15 with 4+ rating"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            messages = [
                {
                    **message,
                    "content": "I uploaded a product image earlier."
                    if message["role"] == "user"
                    and message["content"].startswith("I uploaded a product image. Please analyze it")
                    else message["content"],
                }
                for message in st.session_state.messages
            ]
            try:
                result = run_shopping_agent(messages, st.session_state.shopper_id)
                response = result["messages"][-1].content.replace("`", "")
            except AuthenticationError:
                response = (
                    "Groq rejected its API key, and no working Google fallback was "
                    "available. Add a valid `GOOGLE_API_KEY` to Streamlit Cloud "
                    "Secrets (or fix `GROQ_API_KEY`) and reboot the app."
                )
                st.error(response)
            else:
                st.markdown(response.replace("$", r"\$"))

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
