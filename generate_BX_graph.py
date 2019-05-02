import pandas as pd
filename = "BX-Book-Ratings.csv"
#pd.read_excel(io=file_name+".xls", sheet_name=sheet)

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
    return G

### Wrapper function 
def Wrapper(degree_T = 11, rating_T = 0):
    G = GenGraph(rating_T=rating_T)
    return G #PreprocessDeg(G, degree_T = degree_T)



