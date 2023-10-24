import os
from dotenv import load_dotenv

load_dotenv()
PROJECT_WORKING_DIRECTORY = os.path.expanduser(os.getenv('PROJECT_WORKING_DIRECTORY'))