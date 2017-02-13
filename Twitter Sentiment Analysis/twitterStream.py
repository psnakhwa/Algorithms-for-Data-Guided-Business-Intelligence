from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import operator
import numpy as np
import matplotlib.pyplot as plt


def main():
    conf = SparkConf().setMaster("local[2]").setAppName("Streamer")
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 10)   # Create a streaming context with batch interval of 10 sec
    ssc.checkpoint("checkpoint")

    pwords = load_wordlist("positive.txt")
    nwords = load_wordlist("negative.txt")
   
    counts = stream(ssc, pwords, nwords, 100)
    make_plot(counts)


def make_plot(counts):
    """ 
    Plot the counts for the positive and negative words for each timestep.
    Use plt.show() so that the plot will popup.
    """
    # YOUR CODE HERE
    pos = []
    neg = []
    x = []
    # Appending the values to positive and negative lists
    i = 1
    for val in counts[1:]:
        pos.append(val[0][1])
        neg.append(val[1][1])
        x.append(i)
        i += 1
    # Plotting the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel("Time Step")
    plt.ylabel("Word Count")
    plt.plot(x, pos, 'bo-', x, neg, 'go-')
    ax.legend(('positive', 'negative'),loc ='upper left')
    plt.axis([0,i,0,(max(pos)+50)])
    plt.show()

def load_wordlist(filename):
    """ 
    This function should return a list or set of words from the given filename.
    """
    # YOUR CODE HERE
    file = open(filename,'r')
    word_list = set()
    for line in file.read():
    	word_list.add(line)
    return word_list



def stream(ssc, pwords, nwords, duration):
    kstream = KafkaUtils.createDirectStream(
        ssc, topics = ['twitterstream'], kafkaParams = {"metadata.broker.list": 'localhost:9092'})
    tweets = kstream.map(lambda x: x[1].encode("ascii","ignore"))

    # Each element of tweets will be the text of a tweet.
    # You need to find the count of all the positive and negative words in these tweets.
    # Keep track of a running total counts and print this at every time step (use the pprint function).
    # YOUR CODE HERE
    tweets.pprint()
    # get all the words in the tweets into the words list
    words = tweets.flatMap(lambda line:line.split(" "))
    # Only the words which are in the pwords or nwords are put into the filtered_words list
    filtered_words = words.filter(lambda word:word in pwords or word in nwords)
    # For each word in filtered words, mark it positive or negative
    pairs = filtered_words.map(lambda x:("positive",1) if x in pwords else ("negative",1))
    # Get the total word count
    total_word_count = pairs.reduceByKey(lambda x,y: x+y)
    #Print the total_word_count
    total_word_count.pprint()

    #Prints the first 10 elements of each RDD to the console
    def updateFunc(newValues,runningCount):
    	if runningCount is None:
    		runningCount = 0
    	return sum(newValues,runningCount)
    
    runningCount = pairs.updateStateByKey(updateFunc)
    runningCount.pprint()

    # Let the counts variable hold the word counts for all time steps
    # You will need to use the foreachRDD function.
    # For our implementation, counts looked like:
    #   [[("positive", 100), ("negative", 50)], [("positive", 80), ("negative", 60)], ...]
    counts = []
    # YOURDSTREAMOBJECT.foreachRDD(lambda t,rdd: counts.append(rdd.collect()))
    total_word_count.foreachRDD(lambda t,rdd: counts.append(rdd.collect()))
    ssc.start()                         # Start the computation
    ssc.awaitTerminationOrTimeout(duration)
    ssc.stop(stopGraceFully=True)

    return counts


if __name__=="__main__":
    main()
