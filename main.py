from gensim import downloader

GLOVE_PATH = 'glove-twitter-200'
GLOVE = downloader.load(GLOVE_PATH)

def create_glove_vector(word, data):
    if word not in GLOVE.key_to_index:
        print(f"{word} not an existing word in the model")
         # if you dont have this word - just skip it
        return False
    else:
        vec = GLOVE[word]
        data.append(vec)

if __name__ == "__main__":
    train_dataset = []
    
    with open("/home/student/hw2/NER_task_in_NLP/data/train.tagged") as f:
        lines = f.readlines()
        for line in lines:
            try:
                word, tag = line.rstrip().split("\t")
            except:
                print(line)
                print(line.rstrip().split("\t"))
            # print(train_dataset)
            create_glove_vector(word, train_dataset)
            # print(train_dataset)

