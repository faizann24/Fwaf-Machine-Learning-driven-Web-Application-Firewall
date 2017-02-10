'''
FWAF - Machine Learning driven Web Application Firewall
Author: Faizan Ahmad
Website: http://fsecurify.com
'''

def getNGrams(query): #tokenizer function, this will make 3 grams of each query
	tempQuery = str(query)
	ngrams = []
	for i in range(0,len(tempQuery)-3):
		ngrams.append(tempQuery[i:i+3])
	return ngrams
  
  
filename = 'badqueries.txt'
directory = str(os.getcwd())
filepath = directory + "/" + filename
data = open(filepath,'r').readlines()
data = list(set(data))
badQueries = []
validQueries = []
count = 0
for d in data:
	d = str(urllib.unquote(d).decode('utf8'))   #converting url encoded data to simple string
	badQueries.append(d)
 
filename = 'goodqueries.txt'
directory = str(os.getcwd())
filepath = directory + "/" + filename
data = open(filepath,'r').readlines()
data = list(set(data))
for d in data:
	d = str(urllib.unquote(d).decode('utf8'))
	validQueries.append(d)
  
badQueries = list(set(badQueries))
tempvalidQueries = list(set(validQueries]))
tempAllQueries = badQueries + tempvalidQueries
bady = [1 for i in range(0,len(tempXssQueries))]  #labels, 1 for malicious and 0 for clean
goody = [0 for i in range(0,len(tempvalidQueries))]
y = bady+goody
queries = tempAllQueries

vectorizer = TfidfVectorizer(tokenizer=getNGrams) #converting data to vectors
X = vectorizer.fit_transform(queries)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #splitting data

lgs = LogisticRegression()
lgs.fit(X_train, y_train) #training our model
print(lgs.score(X_test, y_test))  #checking the accuracy
