#!.project/bin/
# we are implementing the quick select deterministic median of medians algorithm: -> finding the kth smallest element of an array A.

#1. divide the array of length n into n/5 chunks of 5 elements
#2. sort the chunks and find the medians
#3. find the median of medians
#4. use that median as partition pivot on array
#5. recursively call the quick select on the appropriate portion of the array A
import matplotlib.pyplot as plt
import random
import time



seed = 42
random.seed(seed)

def partition(A, low, high, pivot):
    """Partially sort the array around the pivot and return its position.

    Args:
        A (array): list or array of numbers
        low (int): start index
        high (int): end index
        pivot (int): index of the partition number

    Returns:
        int: partially sort A and returns the right position of the pivot when sorted.
    """
    # swap the pivot with the last element in the array
    A[high], A[pivot] = A[pivot], A[high]

    i, j = low, high-1 # iterate from both ends of the array
    while i<j:
        while A[i]<A[high]: # increase i until A[i] is larger than the pivot
            i+=1
        while A[j]>A[high]: # decrease j until A[j] is smaller than the pivot
            j-=1
        if i<j:
            A[i], A[j] = A[j], A[i] # swap the positions to keep the smaller elements to the left and the larger element to the right.
            i+=1
            j-=1
        # print(A)

    A[j+1], A[high] = A[high], A[j+1] # put the pivot to its position when sorted.
    return j+1 # the position of the pivot when sorted.

def insertion_sort(A, low, high):
    """ Insertion sort on the small sized array to return the index of median. with respect to low and high.

    Args:
        A (array): array of numbers
        low (int): starting index
        high (int): ending index

    Returns:
        int: index of the median
    """
    for i in range(low+1, high+1):
        key = A[i]
        j = i-1
        while j>=low and A[j] > key:
            A[j+1]=A[j]
            j -= 1
        A[j+1] = key

    return low + (high-low+1)//2 # the index median of A[low:high+1]; #TODO: key part of Algorithm = low + (high-low+1)//2

def medianOfMedians(A, low, high):
    """ Return the index of the median of medians

    Args:
        A (array): list or array of numbers
        low (int): start index
        high (int): end index

    Returns:
        int: median of median index (pivot index)
    """
    n = high-low+1 # length of A
    if n<=5: # base case, sort the small array and return the index of the median
        return insertion_sort(A, low, high)

    medians = [] # collect the index of the medians of the small array
    for i in range(low, high+1, 5):
        j = i + 4
        if j>high:
            j = high
        median = insertion_sort(A, i, j)
        medians.append(median)

    # call a quick select on the list of median indexes to return the median which is an index
    return quickselect(medians, 0, len(medians)-1, len(medians)//2)


def quickselect(A, low, high, k):
    """ Returns the kth smallest element of A.

    Args:
        A (array): array or list of numbers
        low (int): starting index
        high (int): ending index
        k (int): the kth smallest element to find.
        k=0 means finding the smallest element and k=high-low means searching for the largest element

    Returns:
        float: The kth smallest element of A.
    """

    # safeguard for values of k that do not make sense.
    if k>=high-low+1:
        raise Exception("k should be in [0 n-1], n is the size of the array")

    while True:
        if low == high: # base case.
            return A[low]

        p_idx = medianOfMedians(A, low, high) # pivot index corresponding to the median of medians
        p_idx = partition(A, low, high, p_idx) # partition the array around the pivot.

        if k == p_idx: # kth element is the pivot.
            return A[k]
        elif k < p_idx: # kth element is below the pivot.
            #return quickselect(A, low, p_idx-1, k)
            high = p_idx - 1
        else: # the kth element is above the pivot.
            # return quickselect(A, p_idx+1, high, k-p_idx)
            low = p_idx + 1

def theoretical_time_complexity(ns):
    # theoretically, dt = n
    times = [n for n in ns]
    return times


if __name__ == '__main__':
    ns = [10**i for i in range(7)]
    ks = [n//2 for n in ns] # kth

    dts = []
    for k, n in zip(ks, ns):
        A = random.sample(range(1, 2*n), n)

        start = time.time()
        quickselect(A, 0, n-1, k)
        end = time.time()

        dt = (end - start)*1e9 # nanoseconds
        dts.append(dt)

    theo_dts = theoretical_time_complexity(ns)

    avg_dts = sum(dts)/len(dts)
    avg_theo_dts = sum(theo_dts)/len(theo_dts)
    c = avg_dts/avg_theo_dts

    print("coef=", c)
    adj_theo_dts = [c*theo_dt for theo_dt in theo_dts]

    for n, dt, theo_dt, adj_theo_dt in zip(ns, dts, theo_dts, adj_theo_dts):
        print("n={}-> dt={:.2f}ns -> theo_dt={:.0f}-> adj_theo_dt={:.2f}".format(n, dt, theo_dt, adj_theo_dt))

    fig, ax = plt.subplots()

    ax.plot(ns, dts, label="experimental results")
    ax.plot(ns, adj_theo_dts, label="theoretical results")

    plt.legend(loc="lower left")
    plt.grid()
    plt.show()
    # print(arr)
    # start = time.time()
    # print("The {}th smallest element of A is {}".format(k,quickselect(arr, 0, len(arr)-1, 5)))
    # end = time.time()
    # dt = (end - start)* 1e9 # nanoseconds
    # print(dt)
    # print(sorted(arr)[5])
    # print(sorted(arr)[k])

    # print(arr)