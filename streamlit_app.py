import streamlit as st
import requests
import time
import pandas as pd
from pymongo import MongoClient
import subprocess
import os
import tempfile

FASTAPI_URL = "http://127.0.0.1:8080"

st.set_page_config(page_title="APS Failure Classification", page_icon="🚨")

st.title("🚨 APS Failure Classification")


# Function to start FastAPI in the background
def start_fastapi():
    if "fastapi_process" not in st.session_state:
        st.session_state.fastapi_process = subprocess.Popen(["python", "main.py"])
        time.sleep(3)  # Wait for FastAPI to start


# Start FastAPI automatically
start_fastapi()

# -------------------- 📌 SIDEBAR --------------------
st.sidebar.header("🔍 Choose Input Method")
option = st.sidebar.radio("", ["Upload File", "MongoDB URL"])

# ----------------- 📂 FILE UPLOAD OPTION -----------------
if option == "Upload File":
    st.sidebar.subheader("📤 Upload CSV File")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file and st.sidebar.button("🔍 Predict"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(f"{FASTAPI_URL}/predict/", files=files)

        if response.status_code == 200:
            result = response.json()
            if "file_path" in result:
                file_path = result["file_path"]
                st.success(f"Predictions saved: {file_path}")

                # Load and display predictions
                df = pd.read_csv(file_path)
                st.write("### Predictions (First 5 Rows):")
                st.write(df.head())

                # Provide download link
                with open(file_path, "rb") as file:
                    st.download_button("📥 Download Predictions", file, file_name="predictions.csv")

                # Remove temporary file
                os.remove(file_path)
            else:
                st.error("Unexpected response from FastAPI")
        else:
            st.error(f"Error: {response.json().get('error', 'Unknown error')}")

# ----------------- 🛢️ MONGODB INPUT OPTION -----------------
elif option == "MongoDB URL":
    st.sidebar.subheader("🛢️ MongoDB Connection")

    mongodb_url = st.sidebar.text_input("Enter MongoDB Connection URL", "")

    if mongodb_url:
        try:
            client = MongoClient(mongodb_url)
            databases = client.list_database_names()

            if databases:
                selected_db = st.sidebar.selectbox("📂 Select Database", databases)

                if selected_db:
                    db = client[selected_db]
                    collections = db.list_collection_names()

                    if collections:
                        selected_collection = st.sidebar.selectbox("📑 Select Collection", collections)

                        if selected_collection and st.sidebar.button("🔍 Predict"):
                            collection = db[selected_collection]

                            # Fetch data from MongoDB, excluding "_id"
                            data = list(collection.find({}, {"_id": 0}))

                            if not data:
                                st.error("No data found in the selected collection!")
                            else:
                                df = pd.DataFrame(data)

                                # Show MongoDB records in Streamlit UI
                                st.write(f"### Preview of `{selected_collection}` Collection:")
                                st.write(df.head())

                                # Create a temporary CSV file
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
                                    temp_filename = temp_file.name
                                    df.to_csv(temp_filename, index=False)

                                # Send temp file to FastAPI
                                with open(temp_filename, "rb") as f:
                                    files = {"file": f}
                                    response = requests.post(f"{FASTAPI_URL}/predict/", files=files)

                                # Delete temporary file after sending
                                os.remove(temp_filename)

                                if response.status_code == 200:
                                    result = response.json()

                                    if "file_path" in result:
                                        file_path = result["file_path"]
                                        st.success(f"Predictions saved: {file_path}")

                                        # Load and display predictions
                                        df = pd.read_csv(file_path)
                                        st.write("### Predictions (First 5 Rows):")
                                        st.write(df.head())

                                        # Provide download link
                                        with open(file_path, "rb") as file:
                                            st.download_button("📥 Download Predictions", file,
                                                               file_name="predictions.csv")

                                        # Remove the result file
                                        os.remove(file_path)
                                    else:
                                        st.error("Unexpected response from FastAPI")
                                else:
                                    st.error(response.json().get("error", "Prediction failed"))
        except Exception as e:
            st.error(f"MongoDB Connection Error: {e}")
