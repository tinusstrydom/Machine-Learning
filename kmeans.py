# Machine Learning
# Tinus Strydom
# Program implementing K-Means clustering

#import modules/packages
import pandas as pd
import numpy as np

#method requesting user input
#returns file, clusterinput and iterations
def userInput():
    #Welcome Message
    print('Welcome to kmeans algorithm!')

    #Request user input
    #request user to select dataset
    #request user to select amount of clusters
    #request user amount of iterations
    dataChoice = int(input("Please pick dataset to use for testing\n 1. Data of 1953\n 2. Data of 2008\n 3. Data of both years (1953 and 2008)\n"))

    #if else loop to select csv file 
    if dataChoice == 1:
        file = 'data1953.csv'
    elif dataChoice == 2:
        file = 'data2008.csv'
    elif dataChoice == 3:
        file = 'dataBoth.csv'
    else:
        print('Incorrect Choice')
        userInput()

    #assign and set input for clustersInput and iterationsInput 
    clustersInput = int(input("Please enter the number of clusters you want to test? "))
    iterationsInput = int(input("Please enter the number of iterations the algorithm should run? "))
    
    #return the file selection, amount of clusters, amount of iterations
    return file, clustersInput, iterationsInput

#method for setting dataset
#returns xy_array, data 
def dataSet(file):
    #assign to data ,panda read csv method with filename 
    data = pd.read_csv(file)
    #set column names 
    data.columns = ['Country','Birth Rate', 'Life Expectancy']

    #assign and set birth rate and life expectancy columns to x and y
    x, y = data['Birth Rate'], data['Life Expectancy']
    #assign to xy array the values from x and y
    #using zip method to sort the values
    #and then creating numpy array from that values 
    xy_array = np.array(list(zip(x,y)))

    #return xy_array and data
    return xy_array, data

def mm_normalize(data):
    # min-max
    #from the data get the shape of the matrix
    #returns the value of columns and rows
    #set to variables rows and column
    (rows, cols) = data.shape
    #set and create mins and maxs variables to
    #2 columns of zeros and set dtype for precision to float 32
    mins = np.zeros(shape=(cols), dtype=np.float32)
    maxs = np.zeros(shape=(cols), dtype=np.float32)

    #forloop loops for the amount of time value of columns
    ###data[:,j] - iterate over all rows in the columns ###
    #assign a set mins an array of the smallest number in columns
    #assign a set mins an array of the biggest number in columns
    for j in range(cols):
        mins[j] = np.min(data[:,j])
        maxs[j] = np.max(data[:,j])

    #assign and set to result copy of data to loop over
    result = np.copy(data)

    #for loop loops over each rows
    #and then nested for loop loops each column and assigning to result
    #result 
    for i in range(rows):
        for j in range(cols):
            result[i,j] = (data[i,j] - mins[j]) / (maxs[j] - mins[j])
    #return
    return (result, mins, maxs)

def distance(item, mean):
    # Euclidean distance from a data item to a mean
    #assign and set sum 
    sum = 0.0
    
    #assign to dimension the value of length of item
    dim = len(item)
    
    #for loop loops through range of dimension
    #distance((x,y),(a,b)) equals (x-a)2+(y-b)2
    #then sqrt, return the sqrt of sum
    for j in range(dim):
        sum += (item[j] - mean[j]) ** 2

    #return
    return np.sqrt(sum)

def update_clustering(norm_data, clustering, means):
    # given a (new) set of means, assign new clustering
    # return False if no change or bad clustering

    #assign to n length of norm_data
    #assign to k length of the means
    n = len(norm_data)
    k = len(means)

    #assign and set new clustering , copy clustering
    new_clustering = np.copy(clustering)
    #assign to distances the dimension of zeros array
    # from item to each mean
    distances = np.zeros(shape=(k), dtype=np.float32)

    #for loop loops through range of n, length of the data
    #nested for loop loops through range of k, length of clusters
    #checks each point to means
    for i in range(n):  # walk thru each data item
        for kk in range(k):
            distances[kk] = distance(norm_data[i], means[kk])
        new_clustering[i] = np.argmin(distances)
        #new_id = np.argmin(distances)
        #new_clustering[i] = new_id

    #if the arrays of clustering and new clustering is equal return false
    #no changes have been done
    if np.array_equal(clustering, new_clustering):  # no change so done
        return False

    # make sure that no cluster counts have gone to zero
    #assign to counts dimension of zero arrays 
    counts = np.zeros(shape=(k), dtype=np.int)

    #for loop loops for range of n (length of data)
    #assign to c_id , clustering value 
    for i in range(n):
        c_id = clustering[i]
        counts[c_id] += 1
        
    #for loops loop for the range of k (lenght of clusters)
    #bad clustering if counts equal zero
    #return false
    for kk in range(k):  # could use np.count_nonzero
        if counts[kk] == 0:  # bad clustering
            return False

    #there was a change, and no counts have gone 0
    #for loop loops through range of n (length of data)
    for i in range(n):
        # update clustering
        # update by ref
        clustering[i] = new_clustering[i]

    #return
    return True

def update_means(norm_data, clustering, means):
    # given a (new) clustering, compute new means
    # assumes update_clustering has just been called
    # to guarantee no 0-count clusters
    #assign to n and dim the shape of data
    (n, dim) = norm_data.shape
    #assign and set to k the length of means value
    k = len(means)
    #assign to counts the dimension of k, zeros array
    counts = np.zeros(shape=(k), dtype=np.int)
    #assign to new means the dimension of means shape, zeros array
    new_means = np.zeros(shape=means.shape, dtype=np.float32)  # k x dim

    #for loop loops through each row
    for i in range(n):
        #assign to c id the value of clustering each value
        ###clustering to the clusters
        c_id = clustering[i]
        #count the amount of cluster points to cluster
        counts[c_id] += 1
        #for loop loop range of dim
        for j in range(dim):
            #assign and set to new means the acumulated sum
            new_means[c_id,j] += norm_data[i,j]  # accumulate sum

    #for loop loop range of k (clusters)
    #nested loop loop range of dimension
    #each mean
    #assumes not zero
    for kk in range(k):  # each mean
        for j in range(dim):
            new_means[kk,j] /= counts[kk]

    #for loop loops through range of k
    # each mean  
    for kk in range(k):
        for j in range(dim):
            # update means by ref
            means[kk,j] = new_means[kk,j]

#initialize
def initialize(norm_data, k):
     #assign to n and dim the values of shape example(196,2)(column,rows)
    (n, dim) = norm_data.shape
    #assign to clustering the dimenson of n (196)
    clustering = np.zeros(shape=(n), dtype=np.int)  # index = item, val = cluster ID

    #for loop loops throug range of k(amount of clusters)
    #create cluster full of zeros
    for i in range(k):
        clustering[i] = i

    #for loop loops through range of value of clusters to value of rows
    #set random number from low value 0 to value of k and assign that to
    #each cluster    
    for i in range(k, n):
        clustering[i] = np.random.randint(0, k)
        
    #assign to means the dimensions of the array k and dim of zeros
    means = np.zeros(shape=(k,dim), dtype=np.float32)
    #calls update means function 
    update_means(norm_data, clustering, means)
    #return clustering and means
    return(clustering, means) 

#create cluster
#cluster() calls helpers initialize(),update_clustering(), update_means()  
def cluster(norm_data, k, iterate):
    #assign and set to clustering and means the value of initialize returns
    (clustering, means) = initialize(norm_data, k)

    #assign to ok the value of True
    #if a change was made and no bad clustering
    ok = True
    #assign and set max_iteration to iterations value(from user Input)
    max_iter = iterate

    #sanity count check
    #assign to sanity ct the value of 1
    sanity_ct = 1

    #while loop check if sanity count is less or eqaul to max iteration
    while sanity_ct <= max_iter:
        #assign to ok the new means
        #use new means
        ok = update_clustering(norm_data, clustering, means)
        #if the new means is False then break from loop 
        if ok == False:
            #done
            break  

        #update means     
        update_means(norm_data, clustering, means)
        #add one to sanity count
        sanity_ct += 1

    #return clustering
    return clustering

#display method 
def display(data, clustering):

    #create cluster column
    data['Clusters'] = clustering

    #data grouped by countries of cluster
    data_grouped = data[['Country', 'Clusters']].groupby('Clusters')
    #count groups of countries of cluster
    count_groups = data[['Country', 'Clusters']].groupby('Clusters').count()
    #data grouped birthrate of cluster
    birth_group = data[['Birth Rate','Clusters']].groupby('Clusters')
    #birth average for birth grouped (use mean method)
    birth_avg = birth_group.mean()
    #data grouped birthrate of cluster
    life_group = data[['Life Expectancy','Clusters']].groupby('Clusters')
    #birth average for birth grouped (use mean method)
    life_avg = life_group.mean()
    
    #for loop print items(names of countries) of grouped data
    for key, items in data_grouped:
        print(items)

    #return counted groups and the average of birth rate
    return count_groups, birth_avg, life_avg, data_grouped

###cluster() calls helpers initialize(),update_clustering(), update_means()
###update_clustering() calls distance()
###

def main():
    np.set_printoptions(precision=4, suppress=True)
    np.random.seed(2)
    #set variables for returned values from userInput
    (file, clustersInput, iterationsInput) = userInput()
    
    #set variables for returned values from dataset
    #returned xy_array(columns for Birth Rate and life expectancy)
    #returned data(all columns)
    (xy_array, data) = dataSet(file)

    #set variables for returned values from normalize
    #n_data is normalization of the xy_array
    #mins is array of smallest numbers in column - from birth rate and life expectancy
    #max is array of biggest numbers in columns - from birth rate and life expectancy
    (norm_data, mins, maxs) = mm_normalize(xy_array)

    #assign and set amount of clusters using amount set by user
    k = clustersInput

    #assign to clustering the return value of cluster
    clustering = cluster(norm_data, k, iterationsInput)

    print("\nCountries and clusters: ")
    count_groups, birth_avg, life_avg, data_grouped= display(data,clustering)

    #print counted groups for each cluster
    print('\nCounted groups:')
    print(count_groups)
    #print birth rate average for each cluster 
    print('\nBirth Rate average:')
    print(birth_avg)
    #print Life Expectancy rate average for each cluster 
    print('\nLife Expectancy average:')
    print(life_avg)
    print("\nKmeans Done and dusted! \n")
    
    #used this part to save to csv the full list of clusters    
    #dt = data_grouped.apply(lambda _df: _df.sort_values(by=['Clusters']))
    #dt.to_csv('result.csv')
    main()
    
if __name__ == "__main__":
  main()
