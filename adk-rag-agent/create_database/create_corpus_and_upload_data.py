# create_corpus_and_upload_data.py
from __future__ import annotations

import os
import tempfile
from typing import Optional

from dotenv import load_dotenv, set_key
from google.auth import default as google_auth_default
from google.api_core.exceptions import ResourceExhausted, Forbidden, NotFound
from google.cloud import storage
import vertexai
from vertexai.preview import rag


# ---------- Configuration (editable) ----------
# Display name & description for the RAG corpus
CORPUS_DISPLAY_NAME = "AI_Act_Regulation_2024"
CORPUS_DESCRIPTION = "Corpus containing EU AI Act (Regulation (EU) 2024/1689)"

# GCS source info (bucket + object). You can also set these via .env if you prefer.

# Path to your project's .env, relative to this file. Adjust if your structure differs.
ENV_FILE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".env")
)


class DocRAGIngestor:
    """
    End-to-end helper to:
      1) Initialize Vertex AI with ADC credentials.
      2) Create or retrieve a RAG corpus configured with the embedding model.
      3) Download a PDF from GCS to a local temp file.
      4) Upload the file into the RAG corpus.
      5) Persist the corpus name into .env for downstream use.
      6) List files currently in the corpus (sanity check).
    """

    def __init__(self, project_id: str, location: str):
        """
        Args:
            project_id: GCP project ID (e.g., "my-project-123").
            location: Vertex AI location/region (e.g., "us-central1", "europe-west1").
        """
        self.project_id = project_id
        self.location = location
        self.credentials = None
        self.corpus = None  # Will hold the rag.Corpus once created/retrieved

    def initialize_vertex_ai_client(self) -> None:
        """
        Initializes Vertex AI SDK with Application Default Credentials (ADC).
        Raises if credentials cannot be found.
        """
        self.credentials, _ = google_auth_default()
        vertexai.init(
            project=self.project_id,
            location=self.location,
            credentials=self.credentials,
        )
        print(f"✔ Vertex AI initialized for project '{self.project_id}' in '{self.location}'")

    def get_or_create_rag_corpus(self) -> rag.Corpus:
        """
        Gets an existing RAG corpus by display name or creates a new one using
        the specified embedding model.
        """
        embedding_model_config = rag.EmbeddingModelConfig(
            publisher_model="publishers/google/models/text-embedding-004"
        )

        print(f"ℹ Searching for an existing corpus named '{CORPUS_DISPLAY_NAME}'...")
        for existing in rag.list_corpora():
            if existing.display_name == CORPUS_DISPLAY_NAME:
                self.corpus = existing
                print(f"✔ Found existing corpus: display_name='{existing.display_name}', name='{existing.name}'")
                return existing

        print("⚠ No existing corpus found; creating a new one...")
        self.corpus = rag.create_corpus(
            display_name=CORPUS_DISPLAY_NAME,
            description=CORPUS_DESCRIPTION,
            embedding_model_config=embedding_model_config,
        )
        print(f"✔ Created corpus: display_name='{self.corpus.display_name}', name='{self.corpus.name}'")
        return self.corpus

    def _parse_gcs_uri(self, gcs_uri: str) -> tuple[str, str]:
        """
        Parses a gs://bucket/object URI into (bucket_name, blob_name).
        """
        if not gcs_uri.startswith("gs://"):
            raise ValueError("GCS URI must start with 'gs://'")
        path = gcs_uri[5:]  # strip gs://
        parts = path.split("/", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"Invalid GCS URI: {gcs_uri}")
        return parts[0], parts[1]

    def download_gcs_file(self, gcs_uri: str, local_path: str) -> str:
        """
        Downloads a single file from GCS to a local path using google-cloud-storage.

        Args:
            gcs_uri: Full GCS URI (e.g., gs://my-bucket/path/to/file.pdf)
            local_path: Local file path to write to.

        Returns:
            The local path of the downloaded file.

        Raises:
            google.api_core.exceptions.Forbidden: if permissions are insufficient.
            google.api_core.exceptions.NotFound: if bucket/blob doesn't exist.
        """
        bucket_name, blob_name = self._parse_gcs_uri(gcs_uri)
        print(f"⬇ Downloading from GCS: bucket='{bucket_name}', object='{blob_name}' -> '{local_path}'")

        client = storage.Client(credentials=self.credentials, project=self.project_id)
        try:
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(local_path)
        except Forbidden as e:
            print("⛔ Forbidden: Check IAM permissions for the service account/ADC.")
            print("   Required roles usually include 'Storage Object Viewer' on the bucket.")
            raise e
        except NotFound as e:
            print("❓ Not found: Verify bucket/object name and region.")
            raise e

        print("✔ GCS download complete.")
        return local_path

    def upload_file_into_corpus(self, local_path: str, display_name: str, description: str) -> Optional[rag.File]:
        """
        Uploads a local file into the current corpus.

        Args:
            local_path: local file path to upload.
            display_name: human-readable name shown in RAG console.
            description: short description for the file.

        Returns:
            The created rag.File, or None if upload failed due to quota or other errors.
        """
        if not self.corpus:
            raise RuntimeError("Corpus not initialized. Call get_or_create_rag_corpus() first.")

        print(f"⬆ Uploading '{display_name}' into corpus '{self.corpus.display_name}'...")
        try:
            rag_file = rag.upload_file(
                corpus_name=self.corpus.name,
                path=local_path,
                display_name=display_name,
                description=description,
            )
            print(f"✔ Upload successful: file_name='{rag_file.name}'")
            return rag_file
        except ResourceExhausted as e:
            print(f"Quota exceeded while uploading '{display_name}': {e}")
            print("   Tip: Request a higher quota for text-embedding-004 in Cloud Console.")
            return None
        except Exception as e:
            print(f"⚠ Unexpected error uploading '{display_name}': {e}")
            return None

    def persist_corpus_name_to_env(self, env_file_path: str) -> None:
        """
        Writes/updates RAG_CORPUS in the given .env file so other scripts can use it.
        """
        if not self.corpus:
            raise RuntimeError("Corpus not initialized. Cannot persist to .env.")
        try:
            set_key(env_file_path, "RAG_CORPUS", self.corpus.name)
            print(f"✔ Updated RAG_CORPUS in {env_file_path} -> {self.corpus.name}")
        except Exception as e:
            print(f"⚠ Failed to update .env file: {e}")

    def print_corpus_file_inventory(self) -> None:
        """
        Lists files currently in the corpus (for sanity checking).
        """
        if not self.corpus:
            raise RuntimeError("Corpus not initialized. Call get_or_create_rag_corpus() first.")

        files = list(rag.list_files(corpus_name=self.corpus.name))
        print(f"📦 Files in corpus '{self.corpus.display_name}' ({len(files)} total):")
        for f in files:
            print(f"  • {f.display_name}  —  {f.name}")


def main():
    # --- Load environment variables from .env and validate required config ---
    load_dotenv()

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT not set in .env")

    location = os.getenv("GOOGLE_CLOUD_LOCATION")
    if not location:
        raise ValueError("GOOGLE_CLOUD_LOCATION not set in .env")

    # --- Instantiate the ingestor and run the sequence ---
    ingestor = DocRAGIngestor(project_id=project_id, location=location)

    ingestor.initialize_vertex_ai_client()
    ingestor.get_or_create_rag_corpus()
    ingestor.persist_corpus_name_to_env(ENV_FILE_PATH)

    # Use a temporary directory for clean local storage
    with tempfile.TemporaryDirectory() as tmpdir:
        # Get variables values from env
        local_temp_filename = os.getenv("LOCAL_TEMP_FILENAME")
        gcs_uri = os.getenv("GCS_URI")
        local_pdf_path = os.path.join(tmpdir, local_temp_filename)

        # 1) Download the PDF from GCS
        ingestor.download_gcs_file(gcs_uri, local_pdf_path)

        # 2) Upload the PDF into the corpus
        ingestor.upload_file_into_corpus(
            local_path=local_pdf_path,
            display_name=local_temp_filename,
            description="EU AI Act official consolidated PDF (Regulation (EU) 2024/1689).",
        )

    # 3) Show a corpus inventory for verification
    ingestor.print_corpus_file_inventory()


if __name__ == "__main__":
    main()
