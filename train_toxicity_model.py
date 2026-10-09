import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

clean=[
"thank you for your help","this is a useful explanation","i appreciate your answer",
"could you explain this again","the project looks interesting","i agree with your point",
"please share the documentation","this solution works well","great job on the project",
"let us discuss this tomorrow","can you help me with this problem","the results look good",
"i understand your suggestion","this is a reasonable approach","please check the file",
"the model needs more training","we can improve the system","the presentation is ready",
"i will review the code","thanks for sharing","that is a good idea","could you provide an example"
]

toxic=[
"you are stupid","you are an idiot","shut up idiot","you are disgusting",
"nobody likes you","go away loser","you are pathetic","what a moron",
"you are a horrible person","you are worthless","i hate you",
"you are an annoying idiot","stop being stupid","you are dumb",
"this is fucking stupid","this is damn annoying","you suck",
"you are trash","get lost loser","you are a jerk","you are a fool",
"you are an asshole","stop insulting me","you are being hateful",
"i will hurt you","i am going to attack you","you are a terrible human"
]

contexts=["the comment is","the user said","someone wrote","the message says"]

clean += [f"{c} {x}" for c in contexts for x in clean.copy()]
toxic += [f"{c} {x}" for c in contexts for x in toxic.copy()]

texts=clean+toxic
labels=[0]*len(clean)+[1]*len(toxic)

vectorizer=TfidfVectorizer(
    lowercase=True,
    ngram_range=(1,2),
    sublinear_tf=True,
    max_features=20000
)
X=vectorizer.fit_transform(texts)

model=LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
model.fit(X,labels)

joblib.dump(model,"toxicity_classifier.pkl")
joblib.dump(vectorizer,"tfidf_vectorizer.pkl")
print("Created toxicity_classifier.pkl and tfidf_vectorizer.pkl")
