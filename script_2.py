# Save the private key to a file in /tmp
echo "-----BEGIN RSA PRIVATE KEY-----
MIIEogIBAAKCAQEAodPwYNUIP8eYQgkW3ew838wneke/wnFPLq27qLRfB1jtZO4c
k2K/qZOpfNRsY+NRoItj/Fi5RiD5tiRMhjLk8+9S38lVOzopbZAuyr0bSZtFNfCr
MsCPjIVI/tnhtEUcnR0HDDA8iIXjatz8vIGrmzrIQ3H36hJ78pvETrjqeEh9mEKL
1kw96wo+E1wMP0EH+xUtdPrAR775z3sjzWs6YjdB0rdU1eXvcnZCqzNKgRSJkd1/
xZKWJl3zgAHRQFa+L4v8kJESmb2OJ/nDZPRdwO7p//aUIAp6ehp0i93h8K5rMt88
amYjp9lVktyS2Z3tsrm0s4wmeldJFNRRuVV1fwIDAQABAoIBAGi3ZEzEjVn4JiZV
C17/h0SqLsGJvCRuffaOpWg9j2CJwtQU4JjexXdN/daw/pKEfkWM7V5Va05d+DhM
tNgCBvVyYAY/mc+Bi11Cd4TpacbvmpGU8rMct8cOWHXRN6QXNLTWG8FfVUxilw6H
gWPZ5NnF5+D81pe47kvHT/hA4DiEaDK2IwWryf0q1RUsB4/YK0MWADDCB0bTfrzu
Z+O9oW5f/3SBZ4AAEADYABBp7Tty2Pc2NtKGtkjjLEn5i4bCmZ/nUT3aPf5qazfO
XiKRRUe4wEJvavQXrKzaJEzGLE5KMeuyKl2OX4pQvERo2sfGPXExcFnIGkfRxBdH
qCuMJgECgYEA1ud56Je95CULgOcf240DCOW5/kYi9PSH6WZgayGBI2RVQdC/YBOq
s8ctk378ZZKNUa0nOySgcBQ1HxJyGkMSne3jWUpA+2mzRSHKLxlDVXz8+OIi3UZM
RQgbMpzmtDXQLxbF9FAwnnaO0cm+3VUjkvYoaW+fG6El/Rx1ldXOFp8CgYEAwMYj
oouXF/4twzWY4j7S9fwXaOZ8qmJcv0awWgE25HQL7mblMb53mVoB1VJOL4L6zfmU
VIM2Yaqoxj9C/h9roI4mGHM0QC8rHU5HA4Ie/vBct6D9/HexW7bSSj8Rfd54C8gL
bu5zi7U5q05r5mLPzRjo+/9Y88PSnotmIEUmlSECgYBmmcMJOhEN8GXKmA4Mqwks
4UjoTiH4YxrUYu1bmHZoKEnQD1KfEySnikuHJNRpxgs0WH/na7gxamRmPk89nJIx
1lZ51cqqfa96LQSzcdNE6FR6mrRcgmh9eL5Lbr9ygFKxeKTv3K0pqp7LKA+46iH+
0wk+NW14KnrRmhnFfHtVLwKBgC5W8vX97EI+Pa3xUmEAjSHIibATx+AFaRop2fao
2BMuujeC0JGWoZVhh5NKB7VwHO4qkreGGyp7JbsSNf50eyDLSukRuHu9WvXefT+g
pebOzNrvfl6UPzQ2zHJAmunQ8raSTf1KoMfytnwxi8qD8kPxOwBor29ZoWWvSMt1
FANBAoGAMeD0VKCqTDQhxSan/bV+/RhTDSmLy3xCiItVflCfRx53VF/uzRV9q/xr
8qNO3b8S4+pmRXvXblpm5yEpQvscS3FY7rI8kLDE7ZYsHOm6KESrJ96bBwFBbUl9
K6yEEBiSQ1xGG5Y2Cgndc2Xw2VkHIZv2WH1XefNeZZjgcBeD1ig=
-----END RSA PRIVATE KEY-----" > /tmp/my_key.pem

chmod 600 /tmp/my_key.pem

#Set the path to your Jenkins workspace
JENKINS_WORKSPACE=${WORKSPACE}
EC2_DEST_DIR="/home/ec2-user/internProject/app/padb"  # New directory 'padb'

#Ensure the new directory 'padb' exists
ssh -i /tmp/my_key.pem -o StrictHostKeyChecking=no ec2-user@44.243.32.161 "mkdir -p $EC2_DEST_DIR"

mkdir -p ${JENKINS_WORKSPACE}/transferred_scripts

scp -i /tmp/my_key.pem ec2-user@44.243.32.161:/home/ec2-user/internProject/app/ingest.py ${JENKINS_WORKSPACE}/transferred_scripts/

# Verify the file transfer
if [ -f "${JENKINS_WORKSPACE}/transferred_scripts/ingest.py" ]; then
    echo "File transfer successful"
else
    echo "File transfer failed"
    exit 1
fi

# Set up SSH key and transfer steps...
# Set up SSH key and transfer steps remain the same...

# Setup pyenv
if [ -d "$HOME/.pyenv" ]; then
    echo "Pyenv already installed, updating PATH..."
else
    echo "Installing pyenv..."
    curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
fi

# Add pyenv to PATH
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Install Python 3.9.16 if not already installed
if ! pyenv versions | grep -q 3.9.16; then
    pyenv install 3.9.16
fi
pyenv global 3.9.16

# Verify Python version
python3 --version  # Should show 3.9.16

pip install --no-cache-dir llama-index==0.12.4

# Install other dependencies
pip install --no-cache-dir 'pydantic==2.10.3'
pip install --no-cache-dir 'boto3==1.35.76'

# Remove existing venv if it exists
rm -rf ${JENKINS_WORKSPACE}/venv

# Previous pyenv setup remains the same...

# Previous pyenv setup remains the same...

# Previous pyenv setup remains the same...

# Create and activate virtual environment
python3 -m venv ${JENKINS_WORKSPACE}/venv
source ${JENKINS_WORKSPACE}/venv/bin/activate

# Upgrade pip
python3 -m pip install --upgrade pip

# Install the exact versions that work in your environment
pip install --no-cache-dir 'pydantic==2.10.3'
pip install --no-cache-dir 'pydantic-core==2.27.1'
pip install --no-cache-dir 'boto3==1.35.76'
pip install --no-cache-dir 'faiss-cpu==1.7.2'
pip install --no-cache-dir 'langchain==0.3.10'
pip install --no-cache-dir 'langchain-community==0.3.10'
pip install --no-cache-dir 'langchain-core==0.3.33'
pip install --no-cache-dir 'langchain-huggingface==0.1.2'
pip install --no-cache-dir 'llama-index==0.12.4'
pip install --no-cache-dir 'sentence-transformers==3.3.1'
pip install --no-cache-dir 'torch==2.5.1+cpu' --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir 'transformers==4.47.0'
pip install --no-cache-dir 'numpy==1.26.4'
pip install --no-cache-dir 'typing_extensions==4.12.2'
pip install --no-cache-dir 'llama-index==0.12.4'
pip install --no-cache-dir 'llama-index-embeddings-huggingface==0.4.0'
pip install --no-cache-dir 'llama-index-core==0.12.4'

echo "Using Python: $(which python3)"

# Create scripts directory and copy file
mkdir -p ${JENKINS_WORKSPACE}/scripts
cp ${JENKINS_WORKSPACE}/transferred_scripts/ingest.py ${JENKINS_WORKSPACE}/scripts/

# Update the imports in the existing script
# Update the imports in the existing script
sed -i '
    s/from langchain.embeddings import HuggingFaceEmbeddings/from langchain_community.embeddings import HuggingFaceEmbeddings/g
    s/from langchain.docstore.in_memory import InMemoryDocstore/from langchain_community.docstore.in_memory import InMemoryDocstore/g
    s/from langchain.vectorstores import FAISS/from langchain_community.vectorstores import FAISS/g
    s/from llama_index.core import SimpleDirectoryReader/from llama_index import download_loader/g
    s/from llama_index.embeddings.huggingface import HuggingFaceEmbedding/from llama_index_embeddings_huggingface import HuggingFaceEmbedding/g
    s/from llama_index import Settings/from llama_index.core import Settings/g
    s/from llama_index import StorageContext/from llama_index.core import StorageContext/g
    s/from llama_index.core.node_parser import CodeSplitter, SentenceSplitter/from llama_index.node_parser import CodeSplitter, SentenceSplitter/g
' ${JENKINS_WORKSPACE}/scripts/ingest.py

# Add a line at the beginning of the script to get SimpleDirectoryReader
sed -i '1i\SimpleDirectoryReader = download_loader("SimpleDirectoryReader")' ${JENKINS_WORKSPACE}/scripts/ingest.py

# Execute the script
cd ${JENKINS_WORKSPACE}
python ${JENKINS_WORKSPACE}/scripts/ingest.py

# Deactivate virtual environment
deactivate

# Clean up
rm -f /tmp/my_key.pem
