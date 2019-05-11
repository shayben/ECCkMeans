import pandas as pd
import numpy as np
filename = "BX-Book-Ratings.csv"
filename_Books = "BX-Books.csv"
filename_Users = "BX-Users.csv"
#pd.read_excel(io=file_name+".xls", sheet_name=sheet)

def GetBookID():
    books = {}
    df = pd.read_csv(filename_Books, error_bad_lines=False,
                     encoding = "ISO-8859-1",
                     names=['ISBN','title','author','year','publisher',
                            'image1','image2','image3'], delimiter=";")
    for i in range(1, len(df['ISBN'])):
        books[df['ISBN'][i]] = [df['title'][i], df['author'][i],
                                df['year'][i]]
    return books

### Reading CSV and creating graph
### Creating an edge only when the rating is large enough
def GenGraph(rating_T = 0):
    print("Reading and Generating Graph...", end="")
    G = {}
    df = pd.read_csv(filename, error_bad_lines=False,
                     encoding = "ISO-8859-1",
                     names=['user','book','rating'], delimiter=";")
    for i in range(1, len(df['user'])):
        if int(df['rating'][i]) < rating_T: continue
        if df['user'][i] not in G: G[df['user'][i]] = []
        if df['book'][i] not in G: G[df['book'][i]] = []
        G[df['book'][i]].append(df['user'][i])
        G[df['user'][i]].append(df['book'][i])
    print("done!")
    # print(len(G))
    return G

### Removing all users that have rated less than degree_T books
### Removing all books that have been rated less than degree_T times
def PreprocessDeg(G, degree_T = 11):
    print("Preprocessing...", end="")
    topop = []
    for u in G.keys():
        if len(G[u]) < degree_T:
            topop.append(u)
            for v in G[u]:
                G[v].remove(u)    
    for u in topop:
        G.pop(u)
    print("done!")
    print("NB nodes:", len(G))
    return G 


def ToMatrix(G):
    mapping = {}
    inv_mapp = {}
    count = 0
    for i in G.keys():
        mapping[i] = count
        inv_mapp[count] = i
        count+=1

    n = len(G.keys())
    X = np.matrix([[0]*n for i in range(n)])
    for i in G.keys():
        for j in G[i]:
            X[mapping[i], mapping[j]] = 1
            X[mapping[j], mapping[i]] = 1
        
    return X, mapping, inv_mapp
    

### Wrapper function 
def Wrapper(degree_T = 11, rating_T = 0):
    G = GenGraph(rating_T=rating_T)
    prep = PreprocessDeg(G, degree_T = degree_T)
    X, mapping, inv_mapp = ToMatrix(prep)
    books = GetBookID()
    return X, mapping, inv_mapp, books
    

#Wrapper()

