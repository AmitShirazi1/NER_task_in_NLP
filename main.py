from gensim import downloader

GLOVE_PATH = 'glove-twitter-200'
GLOVE = downloader.load(GLOVE_PATH)
BIO_DICT = {"B": 1,
            "I": 2,
            "O": 3}

def create_glove_vector(word):
    if word not in GLOVE.key_to_index:
        print(f"{word} not an existing word in the model")
         # if you dont have this word - just skip it
        return False
    else:
        vec = GLOVE[word]
        return vec

if __name__ == "__main__":
    train_dataset = []
    print("hey")
    with open("data/train.tagged") as f:
        lines = f.readlines()
        print(lines)
        # for line in lines:
        #     line.split
        #     train_dataset.append(create_glove_vector())