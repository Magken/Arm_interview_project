import os
import tarfile

class Functionalities:
    """Handles retrieval data extraction."""
    
    RETRIEVAL_DIR = "retrieval_data"
    ARCHIVE_PATH = "retrieval_data.tar.gz"

    @staticmethod
    def extract_retrieval_data():
        """Extracts retrieval_data.tar.gz if not already extracted."""
        if not os.path.exists(Functionalities.RETRIEVAL_DIR):  
            print("🔍 Extracting retrieval data from archive...")
            with tarfile.open(Functionalities.ARCHIVE_PATH, "r:gz") as archive:
                archive.extractall(".")
            print("✅ Retrieval data extracted!")
        else:
            print("✅ Retrieval data already exists.")
