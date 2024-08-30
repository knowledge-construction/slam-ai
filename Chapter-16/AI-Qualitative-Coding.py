#%%
import torch
import pandas as pd
from io import StringIO
from   sentence_transformers import SentenceTransformer # conda install sentence-transformer
from   sentence_transformers import util
import nltk
import regex as re
################################################################################

#%%
#
# GPU check & model download
#
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
#
# Download necessary models
#
nltk.download('punkt') # nltk tokenizer

model_name = 'sentence-transformers/all-MiniLM-L12-v2' # Change according to needs
model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
model      = SentenceTransformer(model_name)

#%%
# For your codebook, the key is to come up with a consistent labeling scheme.
# I like Construct.Dimension(.SubDimensions)*
# Always separate the label and data with a colon (:)
# Constructs, dimensions, examples can be separated by blank lines, but it's all syntax sugar
filename = 'IAM-CodeBook.txt' # CHANGE THIS TO YOUR OWN CODEBOOK
with open(filename, 'r') as file:
    raw_file = file.read()

codebook_list = raw_file.split('\n')
codebook_list = [line for line in codebook_list if line.strip()!='']
# Add any other cleaning. Here I just get rid of spaces after the :
codebook_list = [re.sub(r':[ ]+', ':', line) for line in codebook_list]

codebook_codes = [line.split(':', 1)[0] for line in codebook_list]
codebook_data  = [line.split(':', 1)[1] for line in codebook_list]


# %%
# assumes user & text columns,
filename = 'IAM42-Clean.csv'
df = pd.read_csv(filename, encoding='utf-8')

postings = df.text
# %%
# Create embeddings for codebook and postings
################################################################################
codebook_embeddings = model.encode(codebook_data)
posting_embeddings = model.encode(postings)

# Determine dimension similarities
################################################################################
all_embedding_matrix = util.cos_sim(posting_embeddings, codebook_embeddings)

# %%
labels = [item.split(':', 1)[0] for item in codebook_list]

# %%
dfe = pd.DataFrame(all_embedding_matrix, columns=labels)
dft = pd.DataFrame(df.text)
dff = pd.concat([dft, dfe], axis=1)
# %%
out_file = 'IAM42-Analysis.csv' # Change this
dff.to_csv(out_file, index = False)