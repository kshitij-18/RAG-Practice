from storage import StoragePipeline
import os

path = os.path.join(os.getcwd(), "src", "docs")

storage_pipeline = StoragePipeline(collection_name="company-policy-docs", docs_path=path)
storage_pipeline.store()