import firebase_admin
from firebase_admin import credentials, auth
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Use environment variables
service_account_info = {
    "type": os.getenv('type'),
    "project_id": os.getenv('project_id'),
    "private_key_id": os.getenv('private_key_id'),
    "private_key": os.getenv('private_key').replace('\\n', '\n'),
    "client_email": os.getenv('client_email'),
    "client_id": os.getenv('client_id'),
    "auth_uri": os.getenv('auth_uri'),
    "token_uri": os.getenv('token_uri'),
    "auth_provider_x509_cert_url": os.getenv('auth_provider_x509_cert_url'),
    "client_x509_cert_url": os.getenv('client_x509_cert_url'),
    "universe_domain": os.getenv('universe_domain')
}

# Initialize the Firebase Admin SDK with the environment variables
cred = credentials.Certificate(service_account_info)
firebase_admin.initialize_app(cred, {
    'projectId': os.getenv('project_id'),
})

# Get the most recently signed-in user
def get_latest_signed_in_user():
    try:
        # Iterate through all users
        users = auth.list_users().iterate_all()
        latest_user = None

        for user in users:
            if not latest_user or user.user_metadata.last_sign_in_timestamp > latest_user.user_metadata.last_sign_in_timestamp:
                latest_user = user

        if (latest_user):
            print('Most recently signed-in user UID:', latest_user.uid)
        else:
            print('No users found.')
    except Exception as e:
        print('Error getting users:', e)

if __name__ == '__main__':
    get_latest_signed_in_user()
